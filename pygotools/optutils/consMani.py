
__all__ = [
    'lbubToBox',
    'addBoxToInequality',
    'addLBUBToInequality',
    'feasiblePoint',
    'feasibleStartingValue',
    'findAnalyticCenter',
    'findAnalyticCenterBox',
    'findAnalyticCenterLBUB',
    'bindingConstraint',
    'bindingConstraintBox',
    'bindingConstraintLBUB',
    'redundantConstraint',
    'redundantConstraintBox',
    'redundantConstraintLBUB',
    'constraintToDualVertices',
    'constraintToVertices',
    'verticesToConstraint',
    'extractAbVFromHull',
    'distanceToPlane'
    ]

import copy

import scipy.spatial, scipy.linalg
import numpy

import cvxopt # the base.matrix class
from cvxopt import matrix, spdiag, log, mul, div # random shit
from cvxopt import solvers, lapack, blas # things to solve stuff
from cvxopt.modeling import variable, op

solvers.options['show_progress'] = False

def lbubToBox(lb, ub):
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

def addBoxToInequality(box=None, G=None, h=None):
    '''
    Add the box constraints to inequalities
    
    Parameters
    ----------
    box: array like
        matrix of box constraints
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h

    Returns
    -------
    tuple: shape(2,)
        our new Gx<=h with G our first element and h the second
        
    See Also
    --------
    :func:`addLBUBToInequality`
    
    '''
    if G is None and h is None and box is None:
        # Stop smoking
        raise Exception("No input!")

    if box is not None:
        if type(box) in (list, tuple):
            box = numpy.array(box)
    
        p = len(box[:,0])
        A = numpy.append(numpy.eye(p), -numpy.eye(p), axis=0)
        b = numpy.append(box[:,1], -box[:,0], axis=0)
    else:
        A = None

    # now the inequality constraints
    if G is not None:
        if h is not None:
            if type(G) is not numpy.ndarray:
                G = numpy.array(G)
            if type(h) is not numpy.ndarray:
                if type(h) in (int, float):
                    h = numpy.array([h])
                else:
                    h = numpy.array(h)

            # don't have to further check dimension because 
            # it will be checked later anyway
            if A is not None:
                G = numpy.append(G, A, axis=0)
                h = numpy.append(h, b, axis=0)

        else:
            raise Exception("Have G in Gx<=h but not h")
    else: # G is None
        if h is not None:
            raise Exception("Have h in Gx<=h but not G")
        
        G = A
        h = b

    return G, h

def addLBUBToInequality(lb=None, ub=None, G=None, h=None):
    '''
    Add the box constraints to inequalities
    
    Parameters
    ----------
    lb: array like
        lower bounds
    ub: array like
        upper bounds
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h

    Returns
    -------
    tuple: shape(2,)
        our new Gx<=h with G our first element and h the second
        
    See Also
    --------
    :func:`addBoxToInequality`
    
    '''
    if lb is not None and ub is not None:
        box = lbubToBox(lb,ub)
    else:
        raise Exception("Require both lb and ub")
    
    return addBoxToInequality(box, G, h)


def feasiblePoint(x, G, h, allowQuasiBoundary=False):
    '''
    Determine where x is a feasible point given the set
    of inequalities Gx<=h
    
    Parameters
    ----------
    x: array like
        input location
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
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
    if type(G) is cvxopt.base.matrix:
        G = numpy.array(G)
    if type(h) is cvxopt.base.matrix:
        h = numpy.array(h)
    if type(x) is cvxopt.base.matrix:
        x = numpy.array(x)

    # can our point sit on the hyperplane?
    # false means no
    # also better as it prevents stupid stuff happening simply
    # due to numerical error
    if allowQuasiBoundary==False:
        h = h - numpy.sqrt(numpy.finfo(numpy.float).eps)

    # an infeasible point means that one of the inequality is violated
    return numpy.any(G.dot(x)-h>=0)==False

def feasibleStartingValue(G, h, allowQuasiBoundary=False, isMin=True):
    '''
    Find a feasible starting value given a set of inequalities Gx<=h
    
    Parameters
    ----------
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    allowQuasiBoundary: bool, optional
        whether we allow points to sit on the hyperplanes.  This 
        also acts as a mechanism to accommodate machine precision
        problem
    isMin: bool, optional
        whether we want to solve the problem min x s.t. Gx<=h or
        max x s.t. Gx>=h
        
    Returns
    -------
    x:
        a feasible point
    '''
    # equal weight for each dimension
    p = len(G[0,:])
    if isMin:
        f = matrix(numpy.ones(p))
    else:
        f = matrix(-numpy.ones(p))
    # change the type to make it suitable for cvxopt
    if type(G) is numpy.ndarray:
        G = matrix(G)
    if type(h) is numpy.ndarray:
        h = matrix(h)

    # aka do we allow for some numerical error
    if allowQuasiBoundary==False:
        h = h - numpy.sqrt(numpy.finfo(numpy.float).eps)

    # find our feasible point
    h1 = copy.deepcopy(h)
    # Note that lapack.gels expects A to be full rank
    lapack.gels(+G,h1)
    x = h1[:p]
    # test feasibility
    if feasiblePoint(numpy.array(x),numpy.array(G),numpy.array(h),allowQuasiBoundary):
        # print "YES!, our shape is convex"
        return numpy.array(x)
    else:
        # print "Solving lp to get starting value"
        sol = solvers.conelp(f,G,h)
        # check if see if the solution exist
        if sol['status']=="optimal":
            pass
        elif sol['status']=="primal infeasible":
            raise Exception("The interior defined by the set of inequalities is empty")
        elif sol['status']=="dual infeasible":
            raise Exception("The interior is unbounded")
        else:
            if feasiblePoint(numpy.array(sol['x']),numpy.array(G),numpy.array(h), allowQuasiBoundary):
                pass
            else:
                raise Exception("Something went wrong, I have no idea")
    
        return numpy.array(sol['x'])

def _findChebyshevCenter(G, h, full_output=False):
    # return the binding constraints
    bindingIndex,dualHull,G,h,x0 = bindingConstraint(G, h, None, True)
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
    
def findAnalyticCenter(G, h, full_output=False):
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

    ## note that the new set of G and h are the binding constraints
    bindingIndex, dualHull, G, h, x0 = bindingConstraint(G, h, None, True)


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
        raise Exception("Problem! Unidentified yet")

    # this is rather confusing because part of the outputs are
    # in cvxopt matrix format while the first is in numpy.ndarray
    # also note that G and h are the binding constraints
    # rather than the full set of inequalities
    if full_output:
        return numpy.array(sol['x']).flatten(), sol, G, h
    else:
        return numpy.array(sol['x']).flatten()

def findAnalyticCenterBox(box=None, G=None, h=None, full_output=False):
    '''
    Find the analytic center of a polygon given lower and upper bounds
    in matrix form and inequalities Gx<=h
    
    Parameters
    ----------
    box: array like
        lower bounds and upper bounds of dimension (p,2) where
        p is the number of variables
    G: array like
        matrix G in Gx<=h, shape (p,d)
    h: array like
        vector b in Gx<=h  shape (p,)
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
    G,h = addBoxToInequality(box,G,h)

    return findAnalyticCenter(G,h,full_output)

def findAnalyticCenterLBUB(lb=None, ub=None, G=None, h=None, full_output=False):
    '''
    Find the analytic center of a polygon given lb_{i} <= x_{i} <= ub_{i}
    and inequalities Gx<=h
    
    Parameters
    ----------
    lb: array like
        lower bounds
    ub: array like
        upper bounds
    G: array like
        matrix G in Gx<=h, shape (p,d)
    h: array like
        vector h in Gx<=h  shape (p,)
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

    box = lbubToBox(lb, ub)
        
    return findAnalyticCenter(box, G, h, full_output)

def bindingConstraint(G, h, x0=None, full_output=False):
    '''
    Find the binding constraints given Gx<=h and a guess (optional) x0
    
    Parameters
    ----------
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    x0: array like, optional
        a point in the interior of Gx<=h
    full_output: bool, optional
        if full set of output is desired

    Returns
    -------
    i:
        set of index for the binding constraints
    
    '''

    # checking thata input is sane
    if G is None or h is None:
        raise Exception("Expecting input for both G and h")
    else:
        if len(G[:,0])!=len(h):
            raise Exception("Number of rows in G must equal the number of values in h")

    # first, we find a feasible value
    if type(G) is numpy.ndarray:
        G = matrix(G)
    if type(h) is numpy.ndarray:
        h = matrix(h)
    if x0 is None:
        x01 = feasibleStartingValue(G, h, isMin=False)
        x02 = feasibleStartingValue(G, h, isMin=True)
        # we are going to try and see whether the average of the 
        # two point is valid
        x0 = (x01 + x02) / 2
        if feasiblePoint(x0, G, h):
            # valid, happy and this must be a better guess!
            x0 = matrix(x0)
        else:
            # holy cow, we may have a problem
            raise Warning("May not be a convex hull")
            x0 = matrix(x01)
    else:
        # first test on the dimension
        if len(x0) != G[0,:]:
            raise Exception("The center should have the same dimension as the inequality")
        # second test on feasibility
        if feasiblePoint(x0, G, h):
            if type(x0) is numpy.ndarray:
                x0 = matrix(x0)
            elif type(x0) is list:
                x0 = matrix(x0)
        else:
            raise Warning("Input feasible point does not belong to the interior of the feasible set")
            # input not valid, we find a new point
            x01 = feasibleStartingValue(G, h, isMin=False)
            x02 = feasibleStartingValue(G, h, isMin=True)
            x0 = (x01 + x02) / 2
            if feasiblePoint(x0, G, h):
                x0 = matrix(x0)
            else:
                x0 = matrix(x01)

    # find our dual points
    D = constraintToDualVertices(G, h, x0)
    
    # construct our convex hull
    hull = scipy.spatial.ConvexHull(D,False)

    # give them the actual convex hull
    # and also the set of inequalities, which we will want if 
    # we have converted the box constraints to inequality
    if full_output:
        return hull.vertices, hull, G[hull.vertices.tolist(),:], h[hull.vertices.tolist()], x0
    else:
        # the index of the binding constraints
        return hull.vertices

def bindingConstraintBox(box, G, h, x0=None, full_output=False):
    '''
    Find a feasible starting value given a set of inequalities Gx<=h
    
    Parameters
    ----------
    box: array like
        box constraints of dimension (d,p) where p = len(x0)
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    full_output: bool, optional
        if additional output is required
        
    Returns
    -------
    i:
        set of index of the binding constraints
    '''
    G,h = addBoxToInequality(box, G, h)
    return bindingConstraint(G, h, x0, full_output)

def bindingConstraintLBUB(lb, ub, A, b, x0=None, full_output=False):
    '''
    Same as :func:`bindingConstraintBox' but requires the input
    to be a set of lower and upper bounds
    '''
    G, h = addBoxToInequality(lb, ub, A, b)
    return bindingConstraint(G, h, x0, full_output)

def redundantConstraint(G, h, x0=None, full_output=False):
    '''
    Find the redundant constraints given a set of inequalities Gx<=h
    
    Parameters
    ----------
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    x0: array like, optional
        a point in the interior of Gx<=h
    full_output: bool, optional
        if additional output is required
        
    Returns
    -------
    i:
        set of index of the redundant constraints
    '''

    # Total
    numConstraints = len(G)
    # Total = redundant + binding
    index = bindingConstraint(G, h, x0, full_output)
    # the full set of index
    totalIndex = numpy.linspace(0, numConstraints-1, numConstraints)
    # delete the binding ones
    return numpy.delete(totalIndex, index)

def redundantConstraintBox(box, G, h, x0=None, full_output=False):
    '''
    Find the redundant constraints given a set of inequalities Gx<=h
    and the box constraints
    
    Parameters
    ----------
    box: array like
        lower bounds and upper bounds of dimension (p,2) where
        p is the number of variables
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    x0: array like, optional
        a point in the interior of Gx<=h
    full_output: bool, optional
        if additional output is required
        
    Returns
    -------
    i:
        set of index of the redundant constraints
    '''
    G,h = addBoxToInequality(box, G, h)
    return redundantConstraint(G, h, x0, False)

def redundantConstraintLBUB(lb, ub, G, h, x0=None, full_output=False):
    '''
    Find the redundant constraints given a set of inequalities Gx<=h
    and the lower and upper bounds
    
    Parameters
    ----------
    lb: array like
        lower bounds
    ub: array like
        upper bounds
    G: array like
        matrix G in Gx<=h
    h: array like
        vector h in Gx<=h
    x0: array like, optional
        a point in the interior of Gx<=h
    full_output: bool, optional
        if additional output is required
        
    Returns
    -------
    i:
        set of index of the redundant constraints
    '''

    G, h = addLBUBToInequality(lb, ub, G, h)
    return redundantConstraint(G, h, x0, False)

def constraintToDualVertices(G, h, x0=None):
    '''
    Convert the set of constraints Gx<=h to the
    verticies in the dual space
    
    Parameters
    ----------
    G: array like
        matrix G in Gx<=h, shape (p,d)
    h: array like
        vector h in Gx<=h  shape (p,)
    x0: array like, optional
        a point in the interior of Gx<=h

    Returns
    -------
    D: array like
        vertices in the dual space
    '''
    if x0 is None:
        x0 = feasibleStartingValue(G, h)
        x0 = matrix(x0)
    else:
        if type(x0) is numpy.ndarray:
            x0 = matrix(x0)
    d = h - G * x0
    # the dual polytope vertices
    # div is the elementwise operation aka Hadamard product 
    # in terms of division
    D = div(G,d[:,matrix(0,(1,len(x0)))])
    return numpy.array(D)

def constraintToVertices(G, h, x0=None, full_output=False):
    '''
    Convert the set of constraints Gx<=h to vertices
    
    Parameters
    ----------
    G: array like
        matrix G in Gx<=h, shape (p,d)
    h: array like
        vector h in Gx<=h  shape (p,)
    x0: array like, optional
        a point in the interior of Gx<=h
    full_output: bool
        if more output is desired

    Returns
    -------
    D: array like
        vertices given by the constraints
    '''
    # find the center
    if x0 is None:
        x0, sol, G, h = findAnalyticCenter(G,h,True)
        x0 = matrix(x0)
    else:
        # convert it to cvxopt.base.matrix
        x0 = matrix(x0)
    
    # find the dual points
    D = constraintToDualVertices(G, h, x0)
    
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
        y = matrix(1.0, (len(F[:,0]),1))
        # solve the least squares problem
        beta,e,rank,s = scipy.linalg.lstsq(numpy.array(F), numpy.array(y))
        if rank==numDimension:
            # only add row if the previous linear system is of full rank
            G[i,:] = beta.ravel()
            totalRow += 1

    G = G[:totalRow,:]

    # center them to the original coordinate
    V = G + x0.T

    if full_output:
        return V, hull, G, h, x0
    else:
        return V

def verticesToConstraint(V,full_output=False):
    '''
    Convert the set of vertices V into constraints of the form Gx<=h 
    
    Parameters
    ----------
    V: array like
        Vertices coordinates
    full_output: bool
        if more output is desired

    Returns
    -------
    G: array like
        G matrix in Gx<=h
    h: array like
        h vector in Gx<=h
    '''

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
    G = numpy.zeros((numSimplex,numDimension))
    totalRow = 0
    # going through the set of simplices
    for i in range(0,numSimplex):
        F = V[hull.simplices[i],:]
        #A[i,:],e,rank,s = scipy.linalg.lstsq(F,numpy.ones(len(F[:,0])))
        #beta,e,rank,s = scipy.linalg.lstsq(numpy.array(F),numpy.array(y))        
        beta,e,rank,s = scipy.linalg.lstsq(F,numpy.ones(len(F[:,0])))
        if rank==numDimension:
            # only add row if the previous linear system is of full rank
            G[i,:] = beta.flatten()
            totalRow += 1

    G = G[:totalRow,:]
    #b = numpy.ones(numSimplex)
    h = numpy.ones(totalRow)
    h += G.dot(c.T)
    
    return G, h

def extractAbVFromHull(hull):
    '''
    Extract the equations A,b,V from a hull object where the
    last output V is the vertices and
    Ax<=b defines the first two output.
    '''
    A = hull.equations[:,:-1]
    b = -hull.equations[:,-1:].flatten()
    V = hull.points
    return A,b,V

def distanceToPlane(x,G,h):
    '''
    Find out the distance between all the points and hyperplane, i.e.
    the set of inequalities defined by Gx<=h
    
    Parameters
    ----------
    X: array like
        input locations, shape (v,d)
    G: array like
        matrix G in Gx<=h, shape (p,d)
    h: array like
        vector h in Gx<=h  shape (p,)

    Returns
    -------
    array like
        distance to the hyperplanes, shape (p, v)

    '''

    numFace = len(h)
    distanceToPlane = abs(G.dot(x.T) - numpy.reshape(h,(numFace,1)))
    return distanceToPlane

def pointInSCO(x, A, b=None, c=None, d=1.0):
    if b is None:
        r = numpy.dot(A,x)
    else:
        r = numpy.dot(A,x) + b

    if c is None:
        rho = d
    else:
        rho = numpy.dot(c,x) + d

    return numpy.dot(r,r) <= rho

def polytopeInSCO(G, h, A, b=None, c=None, d=1.0):
    # it is sufficient to find out whether all the vertices
    # are inside the ball
    V = constraintToVertices(G, h)
    
    t = numpy.array([pointInSCO(v, A, b, c, d) for v in V])
    return numpy.all(t == True)

def polytopeIntersectSOC(G, h, A, b=None, c=None, d=1.0):
    '''
    Test whether the inequality :math:`Gx \le h` and the second order cone
    (SOC) :math:`\| Ax + b \|^{2} \le cx + d`, which defines a ball with
    radius c centered at b, intersects. 

    Parameters
    ----------
    G: :class:`numpy.ndarray`
        G in inequality Gx \le h
    h: :class:`numpy.ndarray`
        h in inequality Gx \le h
    A: :class:`numpy.ndarray`
        A in \| Ax + b \|^{2} \le cx + d
    b: :class:`numpy.ndarray`
        Defaults to None. b in \| Ax + b \|^{2} \le cx + d
    c: :class:`numpy.ndarray`
        Defaults to None.  c in \| Ax + b \|^{2} \le cx + d. 
    d: :class:`numpy.ndarray`
        Defaults to 1, i.e. the unit ball.  d in \| Ax + b \|^{2} \le cx + d

    '''
    if b is None:
        b = numpy.zeros(len(A))
    if c is None:
        c = numpy.zeros((1, len(A)))
        if d <= 0.0:
            raise Exception("d must be greater than 0 when c is None")

    # need to rework the second order cone into format that is
    # accepted by cvxopt 
    A1 = numpy.append(c, A, axis=0)
    b1 = numpy.append(d, -b)
    
    # combining the constraints into one
    G1 = numpy.append(G, A1, axis=0)
    h1 = numpy.append(h, b1)

    # define the dimension of the constraints
    dims = {'l': len(G), 'q': [len(A1)], 's': []}
    # we wish to find the solution to
    # \min x s.t. constraints
    sol = solvers.conelp(matrix(numpy.ones(len(A))),
                         matrix(G1), matrix(h1), 
                         dims)

    if sol['status'] == 'optimal':
        return True
    else:
        return False


