
__all__ = [
          'PolygonObj',
          'triangulatePolygon',
          'divideGivenPolygon',
          'identifyPotentialOptimalPolygonPareto'
          ]

import scipy.spatial
import numpy
import numpy.linalg
import copy
import time

import cvxopt # the base.matrix class

from directUtil import findLowestObjIndex, findHighestObjIndex
from directUtil import identifyPotentialOptimalObjectPareto

from pygotools.gradient.simplexGradient import closestVector
from pygotools.optutils.consMani import *


# make sure we do not get crap

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
            beta,resid,r,s = numpy.linalg.lstsq(X - self.getLocation(),fx-self.getFx())
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

