
__all__ = [
          'PolygonObj' 
          ]

import scipy.spatial, scipy.linalg
import numpy
import numpy.linalg
import copy
import time

import cvxopt # the base.matrix class
from cvxopt import matrix, spdiag, log, mul, div # random shit
from cvxopt import solvers, lapack, blas # things to solve stuff
from cvxopt.modeling import variable, op

from directUtil import findLowestObjIndex, findHighestObjIndex
from directUtil import identifyPotentialOptimalObjectPareto

from pygot.gradient.simplexGradient import closestVector

# make sure we do not get crap
solvers.options['show_progress'] = False
# solvers.options['reltol'] = 1e-8
# solvers.options['abstol'] = 1e-8

class PolygonObj(object):
    '''
    Polygon object
    
    Parameters
    ----------
    func: callable
        objective function
    A: array like, optional
        matrix of A in Ax<=b 
    b: array like, optional
        vector of b in Ax<=b
    hullorV: :class:`scipy.spatial.ConvexHull` or :class:`numpy.ndarray`, optional
        Either a convex hull or the vertices that define our convex hull.  If this
        is not None, then information here has a higher priority than the linear 
        inequalities
        
    '''
    def __init__(self,func,A,b,hullorV=None):
        
        if hullorV is None:
            # find the center
            self._location,sol,A,b = findAnalyticCenter(A,b,full_output=True)
        # beware that the output here of A,b is of the form cvxopt.base.matrix
            self._A = numpy.array(A)
            self._b = numpy.array(b)
            # find our vertices
            self._V = constraintToVertices(A,b,self._location,full_output=False)
        elif type(hullorV) is scipy.spatial.qhull.ConvexHull:
            # we assume that the hull input is correct rather
            # than the set of inequalities
            self._A,self._b,self._V = extractAbVFromHull(hullorV,False)
            self._location = findAnalyticCenter(self._A,self._b)
        elif type(hullorV) is numpy.ndarray:
            # t0 = time.time()
            hull = scipy.spatial.ConvexHull(hullorV,False)

            # t1 = time.time()
            self._A,self._b,self._V = extractAbVFromHull(hull)

            # t2 = time.time()
            # verticesToConstraint(hullorV)
            # t3 = time.time()
            # print "time taken for qhull"
            # print t1-t0
            # print "time taken for internal"
            # print t3-t2
            self._location = findAnalyticCenter(self._A,self._b)
        elif type(hullorV) is cvxopt.base.matrix:
            hull = scipy.spatial.ConvexHull(numpy.array(hullorV),False)
            self._A,self._b,self._V = extractAbVFromHull(hull)
            self._location = findAnalyticCenter(self._A,self._b)
        else:
            raise Exception("Input for convex hull is of an unknown type")

        # now we compute the distance
        self._simplicesDistance = self._computeDistanceToSimplices()
        self._verticesDistance = self._computeDistanceToVertices()
        self._maxSimplicesDistance = numpy.max(self._simplicesDistance)
        self._maxVerticesDistance = numpy.max(self._verticesDistance)

        self._dimension = len(self._A[0,:])
        self._fx = func(self._location)
        
        self._hasSplit = False
        self._simplexGrad = None
        self._decreaseDirectionFromParent = False

    def getFx(self):
        '''
        Returns the objective value evaluated at the center of the polygon
    
        Returns
        -------
        float
        '''
        return copy.deepcopy(self._fx)
    
    def getLocation(self):
        '''
        Returns the center of the polygon
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''

        return copy.deepcopy(self._location)

    def getMeasure(self):
        '''
        Returns the measure of the polygon in terms of size
    
        Returns
        -------
        float
        '''
        return copy.deepcopy(self._maxVerticesDistance)

    def getVertices(self):
        '''
        Returns the set of vertices
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return self._V

    def getDistanceToVertices(self):
        '''
        Returns the set of distances from center to vertices
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return copy.deepcopy(self._verticesDistance)

    def getDistanceToSimplices(self):
        '''
        Returns the set of distances from center to the simplices
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return copy.deepcopy(self._simplicesDistance)

    def getMaxDistanceToVertices(self):
        '''
        Returns the maximum distance from the center to the vertices
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return copy.deepcopy(self._maxVerticesDistance)

    def getMaxDistanceToSimplices(self):
        '''
        Returns the maximum distance from the center to the simplicies
    
        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return copy.deepcopy(self._maxSimplicesDistance)
    
    def getInequality(self):
        '''
        Returns the set of inequalities
    
        Returns
        -------
        A: :class:`numpy.ndarray`
            matrix A in Ax<=b
        b: :class:`numpy.ndarray`
            vector b in Ax<=b
        '''
        return self._A, self._b
    
    def setDirectionFromParent(self):
        '''
        Denote this polygon as a child which it's center to the parent's 
        center forms the smallest angle (which is guarantee to be acute)

        Returns
        -------
        self
        '''
        self._decreaseDirectionFromParent = True
        return self

    def isDirectionFromParent(self):
        '''
        Return the information from :func:`setDirectionFromParent`

        Returns
        -------
        bool
        '''
        return self._decreaseDirectionFromParent

    def getGrad(self):
        '''
        Return the simplex gradient obtained given the child (after
        the split)

        Returns
        -------
        :class:`numpy.ndarray`
        '''
        return self._simplexGrad

    def hasSplit(self):
        '''
        If this polygon has been split already, i.e. if this is 
        a parent or a child

        Returns
        -------
        bool
        '''
        return self._hasSplit
    
    def splitThis(self,childObjList=None):
        '''
        Denote this polygon has splited.  Also find the gradient if it
        exist

        Parameters
        ----------
        childObjList: list, optional
            list of :class:`PolygonObj` who are the child of this polygon

        Returns
        -------
        self
        '''

        if childObjList is None:
            # a simple split
            self._hasSplit = True
        else:
            # our split also involves finding out the simplex gradient
            numChild = len(childObjList)
            X = numpy.zeros((numChild,self._dimension))
            fx = numpy.zeros(numChild)
            for i in range(0,numChild):
                childObj = childObjList[i]
                X[i] = childObj.getLocation()
                fx[i] = childObj.getFx()

            # solve the linear system
            beta,resid,r,s = scipy.linalg.lstsq(X - self.getLocation(),fx-self.getFx())
            self._simplexGrad = beta
            self._hasSplit = True
        return self
    
    def _checkDimension(self,value):
        if type(value) is numpy.ndarray:
            if len(value) != self._dimension:
                raise Exception("Dimension of input do not conform to the object")
        else:
            raise Exception("Expecting type numpy.ndarray")

    def _computeDistanceToSimplices(self):
        # only need to worry about the binding ones
        return self._b - self._A.dot(self._location)

    def _computeDistanceToVertices(self):
        V = self.getVertices()
        D = V - self._location
        numVertices = len(V[:,0])
        # distance holder
        d = numpy.ones(numVertices)
        # Euclidean distance
        for i in range(0,len(D[:,0])):
            d[i] = numpy.linalg.norm(D[i,:])

        return d

def triangulatePolygon(func,polyObj):
    '''
    Triangulate a polygon object given objective function
    
    Parameters
    ----------
    func: callable
        objective function
    polyObj: :class:`PolygonObj`
        the polygon which we want to split

    Returns
    -------
    list:
        the set of new simplex as well as the origin
    '''

    # TODO: think about the case where we only divide a single
    # dimension.

    # each polyObj carries it's own little polygon
    # defined by the inequalities
    A,b = polyObj.getInequality()
        
    polyList = list()

    ###
    # 
    # HERE, we divide the polygon into little triangles
    # 
    ### 

    # We obtain our convex hull and the points at the vertices
    V, dualHull, G, h, x0 = constraintToVertices(A,b,x0=None,full_output=True)
    # and the convex hull in the primal space
    # print "Reached Here"
    # t0 = time.time()
    hull = scipy.spatial.ConvexHull(V,False)

    # t1 = time.time()
    # print t1 - t0
    numFace = len(b)
    numSimplices = len(hull.simplices[:,0])
    numDimension = len(V[0,:])
    numVertices = len(V[:,0])

    # now we split it up given the center
    
    # we are going to find the distance between the vertices and the
    # hyperplanes defined by the inequalities
    dToPlane = distanceToPlane(V,A,b)
    #distanceToPlane = abs(A.dot(V.T) - numpy.reshape(b,(numFace,1)))
    indexList = numpy.linspace(0,numVertices-1,numVertices).astype(int)

    # print indexList
    X = numpy.zeros((numFace,numDimension))
    fx = numpy.zeros(numFace)
    for i in range(0,numFace):
        # print "face number " +str(i)
        # building our new triangle/tetrahedron
        # print distanceToPlane[i]
        # find points not on the plane, subject to machine precision
        # notOnPlane = abs(dToPlane[i])>=1e-8
        onPlane = abs(dToPlane[i])<=1e-8
        # print "\n Face number " +str(i)
        # print notOnPlane
        #indexNotOnPlane = indexList[notOnPlane==False]
        indexOnPlane = indexList[onPlane]
        #print indexList[indexNotOnPlane]
        newV = V[indexList[indexOnPlane],:]
        newV = numpy.append(newV,numpy.reshape(numpy.array(x0),(1,numDimension)),axis=0)

        # newA,newb = verticesToConstraint(newV)
        # print "Object " +str(i)
        # print newV
        # add our new object
        # polyList.append(copy.deepcopy(PolygonObj(func,newA,newb)))

        polyList.append(copy.deepcopy(PolygonObj(func,None,None,newV)))
        X[i] = polyList[i].getLocation()
        fx[i] = polyList[i].getFx()

    #TODO: add in closest polygon
    newPolyObj = polyObj.splitThis(polyList)
    y,r,j = closestVector(X,newPolyObj.getGrad())
    polyList[j].setDirectionFromParent()

    # add in the original and declare it has already been split
    polyList.append(copy.deepcopy(newPolyObj))
        
    return polyList


def divideGivenPolygon(func,polyObj):
    '''
    Divide a polygon object given objective function
    
    Parameters
    ----------
    func: callable
        objective function
    polyObj: :class:`PolygonObj`
        the polygon which we want to split

    Returns
    -------
    list:
        the set of new polygons as well as the origin
    '''

    # TODO: think about the case where we only divide a single
    # dimension.

    # each polyObj carries it's own little polygon
    # defined by the inequalities
    A,b = polyObj.getInequality()
        
    polyList = list()

    ###
    # 
    # HERE, we divide the polygon into little triangles
    # 
    ### 

    # We obtain our convex hull and the points at the vertices
    V, dualHull, G, h, x0 = constraintToVertices(A,b,x0=None,full_output=True)
    # and the convex hull in the primal space
    hull = scipy.spatial.ConvexHull(V,False)
    # random information
    numFace = len(b)
    numSimplices = len(hull.simplices[:,0])
    numDimension = len(V[0,:])
    numVertices = len(V[:,0])

    ###
    #
    # Find the center simplex
    #
    ###
    
    # we are going to find the distance between the vertices and the
    # hyperplanes defined by the inequalities
    dToPlane = distanceToPlane(V,A,b)
    #distanceToPlane = abs(A.dot(V.T) - numpy.reshape(b,(numFace,1)))
    indexList = numpy.linspace(0,numVertices-1,numVertices).astype(int)

    # holder for the new vertices
    X = numpy.zeros((numFace,numDimension))
    for i in range(0,numFace):
        # building our new triangle/tetrahedron
        # find points not on the plane, subject to machine precision
        onPlane = abs(dToPlane[i])<=1e-8
        indexOnPlane = indexList[onPlane]
        newV = V[indexOnPlane,:]
        # take the average
        X[i] = numpy.mean(newV,axis=0)

    # we are going to create our new simplex that sits in the middle
    # of the original simplex
    newPolyObj = PolygonObj(func,None,None,X)
    centerA,centerB = newPolyObj.getInequality()
    
    ###
    #
    # Find the corner simplex
    #
    ###
    
    # we need to find out the closest plane from the middle simplex to each vertex
    # and also the planes that connects to the vertex
    ## getting the transpose because we want to go down the list of vertices
    
    
    centerDToPlane = distanceToPlane(X,centerA,centerB)
    # the number of faces doesn't change for a n+1-simplex
    centerX = numpy.zeros((numFace,numDimension))
    verticesList = list()
    for i in range(0,numFace):
        # building our new triangle/tetrahedron
        # find points not on the plane, subject to machine precision
        onPlane = abs(centerDToPlane[i])<=1e-8
        indexOnPlane = indexList[onPlane]
        newV = X[indexOnPlane,:]
        # take the average
        centerX[i] = numpy.mean(newV,axis=0)
        verticesList.append(indexOnPlane)

    # the distance from the middle of the hyperplane that defines the center 
    # simplex and the hyperplanes.  only used for identifying which point
    # sits on which plane
    centerDToCenterPlane = (centerA.dot(centerX.T) - centerB).T

    # create the set of index for our hyperplanes
    planeIndexList = numpy.linspace(0,numFace-1,numFace).astype(int)
    # holders
    simplexFromCenter = numpy.zeros((numFace,numDimension))
    for i in range(0,numVertices):
        # going to find the planes where vertex i sits on
        planeMeetVertices = abs(dToPlane.T[i])<=1e-8
        # then the index of those planes
        planesMeetVertex = planeIndexList[planeMeetVertices]
        Atemp = A[planesMeetVertex,:]
        btemp = b[planesMeetVertex]
        # then the new plane that completes the convex hull/simplex
        # centerPlaneIndex = numpy.argmin(dVToS[i])
        # print "Vertex " +str(i)
        # print Atemp
        # print btemp
        # Atemp = numpy.append(Atemp,numpy.reshape(centerA[centerPlaneIndex],(1,numDimension)),axis=0)
        # btemp = numpy.append(btemp,centerB[centerPlaneIndex])

        # the set of distance between vertex i and the points lying on the hyperplane
        # of the center simplex
        dVToCenterX = V[i] - centerX
        # index with the minimum distance
        index = numpy.argmin(numpy.linalg.norm(dVToCenterX,axis=1))
        # index of center simplex that completes the corner simplex
        index = numpy.argmin(abs(centerDToCenterPlane[index]))
        # print "Location of V[i]"
        # print V[i]
        # print "dVToCenterX"
        # print dVToCenterX
        # print "Actual distance from vertices to the center simplex mid point"
        # print numpy.linalg.norm(dVToCenterX,axis=1)
        # print "index of center simplex that completes the corner simplex"
        # print index
        # print "and the location of that is "
        # print centerX[index]
        # print "vertices of center simplex"
        # print X[verticesList[index]]

        XTemp = numpy.append(X[verticesList[index]],numpy.reshape(V[i],(1,numDimension)),axis=0)
        # Atemp = numpy.append(Atemp,numpy.reshape(centerA[index],(1,numDimension)),axis=0)
        # btemp = numpy.append(btemp,centerB[index])

        
        # print V[i]
        # print planeMeetVertices
        # print A
        # print b
        # add the object to the list
        # print centerPlaneIndex
        polyList.append(PolygonObj(func,None,None,XTemp))
        # find the location for simplex i
        simplexFromCenter[i] = polyList[i].getLocation()
        
    ###
    #
    # Consolidate information
    #
    ###

    # add in the original and declare it has already been split
    #originalPolyObj = polyObj.splitThis()
    #polyList.append(copy.deepcopy(originalPolyObj))
    # this should be a deepcopy because we remove the original
    # object later using their hash
    #TODO: add in closest polygon
    #polyObj = polyObj.splitThis(polyList[1::])
    newPolyObj = newPolyObj.splitThis(polyList)
    y,r,j = closestVector(simplexFromCenter,newPolyObj.getGrad())
    polyList[j].setDirectionFromParent()
    polyList.append(newPolyObj)

    polyList.append(copy.deepcopy(polyObj.splitThis()))

    return polyList

def identifyPotentialOptimalPolygonPareto(polyObjList,EPSILON=1e-4,includeMin=False):
    '''
    Divide a polygon object given objective function
    
    Parameters
    ----------
    polyObjList: list
        list of :class:`PolygonObj`
    EPSILON: numeric, optional
        control on how local the search is
    includeMin: bool, optional
        

    Returns
    -------
    list:
        index of polygons which we want to divide
    '''

    # only operate on the unsplit polygon
    polyList = _findUnsplitPolygon(polyObjList)

    # find our Pareto front
    listPotentialOptimalIndex = identifyPotentialOptimalObjectPareto(polyList,EPSILON=1e-4,uniqueDecimal=4,includeMin=includeMin)

    # need to convert the index back to the original set which
    # also include polygons that has been split
    listPotentialOptimalOriginal = list()
    for i in listPotentialOptimalIndex:
        listPotentialOptimalOriginal.append(copy.copy(polyObjList.index(polyList[i])))

    return listPotentialOptimalOriginal

def lbubToBox(lb,ub):
    '''
    Convert a set of lower and upper bounds vectors to a matrix
    
    Parameters
    ----------
    lb: array like
        lower bound vector
    ub: array like
        upper bound vector

    Returns
    -------
    :class:`numpy.ndarray`
        our box constraints
    '''

    if type(lb) is not numpy.ndarray:
        lb = numpy.array(lb)
    if type(ub) is not numpy.ndarray:
        ub = numpy.array(ub)
    
    return numpy.array([lb,ub]).T

def addBoxToInequality(box=None,A=None,b=None):
    '''
    Add the box constraints to inequalities
    
    Parameters
    ----------
    box: array like
        matrix of box constraints
    A: array like
        matrix A in Ax<=b
    b: array like
        vector b in Ax<=b

    Returns
    -------
    tuple: shape(2,)
        our new Ax<=b with A our first element and b the second
        
    See Also
    --------
    :func:`addBoxToInequalityLBUB`
    
    '''
    if A is None and b is None and box is None:
        # Stop smoking
        raise Exception("No Center")

    if box is not None:
        if type(box) is list:
            box = numpy.array(box)
    
        p = len(box[:,0])
        G = numpy.append(numpy.eye(p),-numpy.eye(p),axis=0)
        h = numpy.append(box[:,1],-box[:,0],axis=0)
    else:
        G = None

    # now the inequality constraints
    if A is not None:
        if b is not None:
            if type(A) is not numpy.ndarray:
                A = numpy.array(A)
            if type(b) is not numpy.ndarray:
                if type(b) in (int,float):
                    b = numpy.array([b])
                else:
                    b = numpy.array(b)

            # don't have to further check dimension because 
            # it will be checked in cvxopt anyway
            if G is None:
                G = A
                h = b
            else:
                # bind
                G = numpy.append(G,A,axis=0)
                h = numpy.append(h,b,axis=0)

        else:
            raise Exception("Have A in Ax=b but not b")
    else: # A is None
        if b is not None:
            raise Exception("Have b in Ax=b but not A")

    return G,h

def addBoxToInequalityLBUB(lb=None,ub=None,A=None,b=None):
    '''
    Add the box constraints to inequalities
    
    Parameters
    ----------
    lb: array like
        lower bounds
    ub: array like
        upper bounds
    A: array like
        matrix A in Ax<=b
    b: array like
        vector b in Ax<=b

    Returns
    -------
    tuple: shape(2,)
        our new Ax<=b with A our first element and b the second
        
    See Also
    --------
    :func:`addBoxToInequality`
    
    '''
    if lb is not None and ub is not None:
        box = lbubToBox(lb,ub)
    else:
        raise Exception("Require both lb and ub")
    
    return addBoxToInequality(box,A,b)

def _findChebyshevCenter(G,h,full_output=False):
    # return the binding constraints
    bindingIndex,dualHull,G,h,x0 = bindingConstraint(G,h,None,True)
    # Chebyshev center
    R = variable()
    xc = variable(2)
    m = len(h)
    op(-R, [ G[k,:]*xc + R*blas.nrm2(G[k,:]) <= h[k] for k in range(m) ] + [ R >= 0] ).solve()
    R = R.value
    xc = xc.value

    if full_output:
        return numpy.array(xc).flatten(), G, h
    else:
        return numpy.array(xc).flatten()

    ## If we want to check and see the difference between the center  
    ## points obtained using a set of binding constraints and a 
    ## solution with all the constraints.
    # print sol['x']
    # G = newG
    # h = newh
    # print solvers.cp(F)['x']
    
def findAnalyticCenter(G,h,full_output=False):
    '''
    Find the analytic center of a polygon given a set of 
    inequalities Gx<=h.  Solves the problem
    
    min_{x} \sum_{i}^{n} -log(h - Gx)_{i}
    
    subject to the satisfaction of the inequalities
    
    Parameters
    ----------
    G: array like
        matrix A in Gx<=h, shape (p,d)
    h: array like
        vector h in Gx<=h  shape (p,)
    full_output: bool, optional
        whether the full output is required.  Only the center *x* is return
        if False, which is the default.

    Returns
    -------
    x: array like
        analytic center of the polygon
    sol: dict
        solution dictionary from cvxopt 
    G: array like
        binding matrix G in Gx<=h
    h: array like
        binding vector h in G<=h
        
    See Also
    --------
    :func:`findAnalyticCenter`
        
    '''
    # we use the cvxopt convention here in that 
    # Gx \preceq h is our general linear inequality constraints
    # where Ax \le b is our linear inequality
    # and box is our box constraints
#     if type(G) is numpy.ndarray:
#         G = matrix(G)
#     if type(h) is numpy.ndarray:
#         h = matrix(h)
# 
#     if G is None or h is None:
#         raise Exception("Expecting input for G and h")
#     else:
#         if len(G[:,0])!=len(h):
#             raise Exception("Number of rows in G must equal the number of values in h")

    ## note that the new set of G and h are the binding constraints
    bindingIndex,dualHull,G,h,x0 = bindingConstraint(G,h,None,True)


    # define our objective function along with the 
    # gradient and the Hessian
    def F(x=None, z=None):
        if x is None: return 0, matrix(x0)
        y = h-G*x
        # we are assuming here that the center can sit on an edge
        if min(y) <= 0: return None 
        # pretty standard log barrier 
        f = -sum(log(y))
        Df = (y**-1).T * G
        if z is None: return matrix(f), Df
        H =  G.T * spdiag(y**-2) * G
        return matrix(f), Df, z[0]*H

    # then we solve the non-linear program
    try:
        # print "Find the analytic center"
        sol = solvers.cp(F)
    except:
        print feasiblePoint(numpy.array(x0),numpy.array(G),numpy.array(h))
        print "G"
        print G
        print "h"
        print h
        print "Distance to face"
        print h - G*x0
        print "starting location"        
        print x0
        h1 = h
        lapack.gels(G,h1)
        print "LS version"
        print h1[:len(G[0,:])]
        print "Re-solve the problem"
        sol = solvers.conelp(matrix(numpy.ones(len(G[0,:]))),G,h)
        print sol
        print h - G*sol['x']
        sol = solvers.conelp(matrix(-numpy.ones(len(G[0,:]))),G,h)
        print sol
        print h - G*sol['x']
        raise Exception("FUCK!")

   

    # this is rather confusing because part of the outputs are
    # in cvxopt matrix format while the first is in numpy.ndarray
    # also note that G and h are the binding constraints
    # rather than the full set of inequalities
    if full_output:
        return numpy.array(sol['x']).flatten(), sol, G, h
    else:
        return numpy.array(sol['x']).flatten()

def findAnalyticCenterBox(box=None,A=None,b=None,full_output=False):
    '''
    Find the analytic center of a polygon given lower and upper bounds
    in matrix form and inequalities Ax<=b
    
    Parameters
    ----------
    box: array like
        lower bounds and upper bounds of dimension (p,2) where
        p is the number of variables
    A: array like
        matrix A in Ax<=b, shape (p,d)
    b: array like
        vector b in Ax<=b  shape (p,)
    full_output: bool
        whether the full output is required

    Returns
    -------
    x: array like
        analytic center of the polygon
        
    See Also
    --------
    :func:`findAnalyticCenter`
        
    '''
    # we use the cvxopt convention here in that 
    # Gx \preceq h is our general linear inequality constraints
    # where Ax \le b is our linear inequality
    # and box is our box constraints

    # first, we have to convert box to inequality
    G,h = addBoxToInequality(box,A,b)

    return findAnalyticCenter(G,h,full_output)

def findAnalyticCenterLBUB(lb=None,ub=None,A=None,b=None,full_output=False):
    '''
    Find the analytic center of a polygon given lb_{i} <= x_{i} <= ub_{i}
    and inequalities Ax<=b
    
    Parameters
    ----------
    lb: array like
        lower bounds
    ub: array like
        upper bounds
    A: array like
        matrix A in Ax<=b, shape (p,d)
    b: array like
        vector b in Ax<=b  shape (p,)
    full_output: bool
        whether the full output is required

    Returns
    -------
    x: array like
        analytic center of the polygon
        
    See Also
    --------
    :func:`findAnalyticCenter`
        
    '''

    box = lbubToBox(lb,ub)
        
    return findAnalyticCenter(box,A,b,full_output)

def distanceToPlane(x,A,b):
    '''
    Find out the distance between all the points and hyperplane, i.e.
    the set of inequalities defined by Ax<=b
    
    Parameters
    ----------
    X: array like
        input locations, shape (v,d)
    A: array like
        matrix A in Ax<=b, shape (p,d)
    b: array like
        vector b in Ax<=b  shape (p,)

    Returns
    -------
    array like
        distance to the hyperplanes, shape (p, v)

    '''

    numFace = len(b)
    distanceToPlane = abs(A.dot(x.T) - numpy.reshape(b,(numFace,1)))
    return distanceToPlane

def feasiblePoint(x,A,b,allowQuasiBoundary=False):
    '''
    Determine where x is a feasible point given the set
    of inequalities Ax<=b
    
    Parameters
    ----------
    x: array like
        input location
    A: array like
        matrix A in Ax<=b
    b: array like
        vector b in Ax<=b
    allowQuasiBoundary: bool
        whether we allow points to sit on the hyperplanes.  This 
        also acts as a mechanism to accommodate machine precision
        problem

    Returns
    -------
    bool
        True is x is a feasbiel point

    '''

    # type checking
    if type(A) is cvxopt.base.matrix:
        A = numpy.array(A)
    if type(b) is cvxopt.base.matrix:
        b = numpy.array(b)
    if type(x) is cvxopt.base.matrix:
        x = numpy.array(x)

    # can our point sit on the hyperplane?
    # false means no
    # also better as it prevents stupid stuff happening simply
    # due to numerical error
    if allowQuasiBoundary==False:
        b = b - numpy.sqrt(numpy.finfo(numpy.float).eps)

    # an infeasible point means that one of the inequality is violated
    return numpy.any(A.dot(x)-b>=0)==False

def feasibleStartingValue(A,b,allowQuasiBoundary=False,isMin=True):

    # equal weight for each dimension
    p = len(A[0,:])
    if isMin:
        x0 = matrix(numpy.ones(p))
    else:
        x0 = matrix(-numpy.ones(p))
    # change the type to make it suitable for cvxopt
    if type(A) is numpy.ndarray:
        A = matrix(A)
    if type(b) is numpy.ndarray:
        b = matrix(b)

    # aka do we allow for some numerical error
    if allowQuasiBoundary==False:
        b = b - numpy.sqrt(numpy.finfo(numpy.float).eps)

    # find our feasible point
    b1 = copy.deepcopy(b)
    # Note that lapack.gels expects A to be full rank
    lapack.gels(+A,b1)
    x = b1[:p]
    # test feasibility
    if feasiblePoint(numpy.array(x),numpy.array(A),numpy.array(b),allowQuasiBoundary):
        # print "YES!, our shape is convex"
        return numpy.array(x)
    else:
        # print "Solving lp to get starting value"
        sol = solvers.conelp(x0,A,b)
        # check if see if the solution exist
        if sol['status']=="optimal":
            pass
        elif sol['status']=="primal infeasible":
            raise Exception("The interior defined by the set of inequalities is empty")
        elif sol['status']=="dual infeasible":
            raise Exception("The interior is unbounded")
        else:
            if feasiblePoint(numpy.array(sol['x']),numpy.array(A),numpy.array(b),allowQuasiBoundary):
                pass
            else:
                # print A
                # print b
                # print sol
                # print A * sol['x'] - b
                # print "Solution?"
                # print sol['x']
                # b1 = b
                # lapack.gels(A,b)
                # print "LS Solution"
                # print b[:p]
                # print A * b[:p] - b1
                raise Exception("Something went wrong, I have no idea")
    
        return numpy.array(sol['x'])

def bindingConstraint(A,b,x0=None,full_output=False):
    # checking thata input is sane
    if A is None or b is None:
        raise Exception("Expecting input for both A and b")
    else:
        if len(A[:,0])!=len(b):
            raise Exception("Number of rows in A must equal the number of values in b")

    # first, we find a feasible value
    if type(A) is numpy.ndarray:
        A = matrix(A)
    if type(b) is numpy.ndarray:
        b = matrix(b)
    if x0 is None:
        x01 = feasibleStartingValue(A,b,isMin=False)
        x02 = feasibleStartingValue(A,b,isMin=True)
        # we are going to try and see whether the average of the 
        # two point is valid
        x0 = (x01 + x02) / 2
        if feasiblePoint(x0,A,b):
            # valid, happy and this must be a better guess!
            x0 = matrix(x0)
        else:
            # holy cow, we may have a problem
            raise Warning("May not be a convex hull")
            x0 = matrix(x01)
    else:
        # first test on the dimension
        if len(x0) != A[0,:]:
            raise Exception("The center should have the same dimension as the inequality")
        # second test on feasibility
        if feasiblePoint(x0,A,b):
            if type(x0) is numpy.ndarray:
                x0 = matrix(x0)
            elif type(x0) is list:
                x0 = matrix(x0)
        else:
            raise Warning("Input feasible point does not belong to the interior of the feasible set")
            # input not valid, we find a new point
            x01 = feasibleStartingValue(A,b,isMin=False)
            x02 = feasibleStartingValue(A,b,isMin=True)
            x0 = (x01 + x02) / 2
            if feasiblePoint(x0,A,b):
                x0 = matrix(x0)
            else:
                x0 = matrix(x01)

    # find our dual points
    D = constraintToDualVertices(A,b,x0)
    
    # construct our convex hull
    hull = scipy.spatial.ConvexHull(D,False)

    # give them the actual convex hull
    # and also the set of inequalities, which we will want if 
    # we have converted the box constraints to inequality
    if full_output:
        return hull.vertices, hull, A[hull.vertices.tolist(),:], b[hull.vertices.tolist()], x0
    else:
        # the index of the binding constraints
        return hull.vertices

def bindingConstraintBox(box,A,b,x0=None,full_output=False):
    G,h = addBoxToInequality(box,A,b)
    return bindingConstraint(G,h,x0,full_output)

def bindingConstraintLBUB(lb,ub,A,b,x0=None,full_output=False):
    G,h = addBoxToInequality(lb,ub,A,b)
    return bindingConstraint(G,h,x0,full_output)

def redundantConstraint(A,b,x0=None,full_output=False):
    # Total
    numConstraints = len(A)
    # Total = redundant + binding
    index = bindingConstraint(A,b,x0,full_output)
    # the full set of index
    totalIndex = numpy.linspace(0,numConstraints-1,numConstraints)
    # delete the binding ones
    return numpy.delete(totalIndex,index)

def redundantConstraintBox(box,A,b,x0=None,full_output=False):
    G,h = addBoxToInequality(box,A,b)
    return redundantConstraint(G,h,x0,False)

def redundantConstraintLBUB(lb,ub,A,b,x0=None,full_output=False):
    G,h = addBoxToInequalityLBUB(lb,ub,A,b)
    return redundantConstraint(G,h,x0,False)

def constraintToDualVertices(A,b,x0=None):
    if x0 is None:
        x0 = feasibleStartingValue(A,b)
        x0 = matrix(x0)
    else:
        if type(x0) is numpy.ndarray:
            x0 = matrix(x0)
    d = b - A * x0
    # the dual polytope vertices
    # div is the elementwise operation aka Hadamard product 
    # in terms of division
    D = div(A,d[:,matrix(0,(1,len(x0)))])
    return D

def constraintToVertices(A,b,x0=None,full_output=False):
    # find the center
    if x0 is None:
        x0, sol, A, b = findAnalyticCenter(A,b,True)
        x0 = matrix(x0)
    else:
        # convert it to cvxopt.base.matrix
        x0 = matrix(x0)
    
    # find the dual points
    D = constraintToDualVertices(A,b,x0)
    
    # construction of our convex hull
    hull = scipy.spatial.ConvexHull(D,False)

    # information (duh!)
    numSimplex = len(hull.simplices[:,0])
    numDimension = len(D[0,:])
    # holder for the vertices 
    G = numpy.zeros((numSimplex,numDimension))
    
    # find out the intersection between the simplices
    # print "total number of simplices = " +str(numSimplex)
    totalRow = 0
    for i in range(0,numSimplex):
        F = D[hull.simplices[i].tolist(),:]
        #G[i,:],e,rank,s = scipy.linalg.lstsq(F,numpy.ones(len(F[:,0])))
        y = matrix(1.0,(len(F[:,0]),1))
        # solve the least squares problem
        beta,e,rank,s = scipy.linalg.lstsq(numpy.array(F),numpy.array(y))        
        if rank==numDimension:
            # only add row if the previous linear system is of full rank
            G[i,:] = beta.flatten()
            totalRow += 1

        # try:
        #     lapack.gels(+F,y)
        # except:
        #     print "Error"
        #     print numpy.array(F).shape
        #     #print y
        #     print numpy.linalg.matrix_rank(numpy.array(F))
        #     beta,e,rank,s = numpy.linalg.lstsq(numpy.array(F),numpy.array(y))
        #     print "Solve via LS"
        #     print beta
        #     print rank
        #     print s
        #     print "And the rank of D"
        #     print numpy.linalg.matrix_rank(numpy.array(D))
        #     #raise Exception("WTF")
        # G[i,:] = numpy.array(y[0:numDimension]).flatten()

    G = G[:totalRow,:]

    # center them to the original coordinate
    V = G + x0.T

    if full_output:
        return V, hull, A, b, x0
    else:
        return V

def verticesToConstraint(V,full_output=False):

    # Note that we are operating in numpy.ndarray format
    # for some part while doing the rest in cvxopt.base.matrix
    # rather different to other functions because they all 
    # mainly operate in cvxopt.base.matrix

    # type checking
    if type(V) is not numpy.ndarray:
        if type(V) is list:
            V = numpy.array(V)
        elif type(V) is cvxopt.base.matrix:
            V = numpy.array(V)
        else:
            raise Exception("Input type not recognized")

    # construct the convex hull
    try:
        hull = scipy.spatial.ConvexHull(V,False)
    except Warning:
        raise Exception("Caught a warning")
    # information
    numSimplex = len(hull.simplices[:,0])
    numDimension = len(V[0,:])

    # centering, we find the column mean
    c = numpy.mean(V[hull.vertices,:],axis=0)
    # actual centering
    V = V - c
    # holder
    A = numpy.zeros((numSimplex,numDimension))
    totalRow = 0
    # going through the set of simplices
    for i in range(0,numSimplex):
        F = V[hull.simplices[i],:]
        #A[i,:],e,rank,s = scipy.linalg.lstsq(F,numpy.ones(len(F[:,0])))
        #beta,e,rank,s = scipy.linalg.lstsq(numpy.array(F),numpy.array(y))        
        beta,e,rank,s = scipy.linalg.lstsq(F,numpy.ones(len(F[:,0])))
        if rank==numDimension:
            # only add row if the previous linear system is of full rank
            A[i,:] = beta.flatten()
            totalRow += 1

    A = A[:totalRow,:]
    #b = numpy.ones(numSimplex)
    b = numpy.ones(totalRow)
    b += A.dot(c.T)
    
    return A,b

def extractAbVFromHull(hull):
    A = hull.equations[:,:-1]
    b = -hull.equations[:,-1:].flatten()
    V = hull.points
    return A,b,V

def _findUnsplitPolygon(polyList):
    
    if len(polyList)==1:
        # there is nothing to do here
        return polyList
    else:
        # make our new list
        newPolyList = list()
        for o in polyList:
            if o.hasSplit()==False:
                # not a copy because we want the same reference
                # to the objects
                newPolyList.append(o)

        return newPolyList

