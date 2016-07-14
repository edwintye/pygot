import functools

import numpy
from scipy.optimize import approx_fprime

from pygotools.optutils.consMani import addLBUBToInequality, feasiblePoint, constraintToVertices, polytopeInSCO
from pygotools.gradient import forwardGradCallHessian

eps = numpy.sqrt(numpy.finfo(float).eps)

def LipschitzGradient(boxes, func, fStar=None, grad=None, hess=None, locationFunc=None, epsilon=1.0, LType='conservative', full_output=False):
    '''
    Performs underestimation with the Lipschitz gradient
    '''
    
    fi = numpy.array([box.getFx() for box in boxes])
    if fStar is None:
        fStar = min(fi)
    else:
        if fStar > min(fi):
            fStar = min(fi)

    if grad is None:
        grad = functools.partial(approx_fprime, f=func, epsilon=eps)
    if hess is None:
        hess = functools.partial(forwardGradCallHessian, grad, h=eps)

    # gList = [grad(box.getLocation()) for box in boxes]
    # HList = [hess(box.getLocation()) for box in boxes]
    
    ## computing the inequality
    exclusionList = list()
    kList = list()
    hList = list()
    rList = list()
    LList = list()

    for i, box in enumerate(boxes):
        xi = box.getLocation()
        lb = box.getLB()
        ub = box.getUB()
        if locationFunc is not None:
            xi = locationFunc(xi)
            lb = locationFunc(lb)
            ub = locationFunc(ub)

        g = box.getGradient()
        if g is None:
            g = grad(xi)
            box.setGradient(g)

        H = box.getHessian()
        if H is None:
            H = hess(xi)
            box.setHessian(H)

        # e = numpy.linalg.eigvalsh(HList[i])
        e = numpy.linalg.eigvalsh(H)
        LMax = max(abs(e))
        if LType.lower() == 'exact':
            Li = min(e)
            LList.append(Li)
            if Li < 0:
                Li = abs(Li)
                t, h, k, r = _convexExclusion(fi[i], fStar, Li, g, xi, lb, ub, epsilon)
            else:
                # print "Box "+str(i)
                # print numpy.linalg.eig(HList[i])[0]
                t, h, k, r = _concaveExclusion(fi[i], fStar, Li, g, xi, lb, ub, epsilon)
        elif LType.lower() == 'conservative':
            Li = LMax
            LList.append(Li)
            t, h, k, r = _convexExclusion(fi[i], fStar, Li, g, xi, lb, ub, epsilon)
        else:
            raise Exception("Type of calculation for Lipschitz gradient not recognized")

        # We may also want to know if the underestimator is good
        # additional check on whether the whole box is within the radius define by a full Newton step
        # only need to check if we want to eliminate the box
        if t == True:
            t = _boxWithinNewtonStep(lb, ub, xi, g, LMax)

        exclusionList.append(t)
        kList.append(k)
        hList.append(h)
        rList.append(r)

    if full_output:
        return exclusionList, hList, kList, rList, LList
    else:
        return exclusionList

def _boxWithinNewtonStep(lb, ub, c, g, H):
    # we define the Newton Step (or more correctly the trust region) 
    # as a ball with length \| H^{-1}g \|, the full step
    if isinstance(H, numpy.ndarray):
        r = numpy.linalg.solve(H, g)
    else:
        r = numpy.linalg.norm(g/H)

    A, b = addLBUBToInequality(lb, ub)
    return polytopeInSCO(A, b, numpy.eye(len(lb)), b=-c, c=None, d=numpy.dot(r,r))

def _convexExclusion(f, fStar, L, g, x, lb, ub, epsilon=1.0):
    '''
    Convex in the sense that area within a ball \| x - h \|^{2} + k \le 0
    is not the area of interest
    '''
    # The notation of \| x - h \|^{2} \le k arise from 
    # completing the squrae x^{T}Ax + bx + c <=> \| x - h \|^{2} \le k
    k = 2.0/L * (fStar - epsilon - f - numpy.dot(g,g)/(2.0*L))
    h = x + g/L
    # radius of the ball
    if k > 0:
        # the inequality cannot be satisfied
        return False, h, k, 0.0
    else:
        # find the radius of the second order cone
        r = numpy.sqrt(-k)
    
        A, b = addLBUBToInequality(lb, ub)
        return polytopeInSCO(A, b, numpy.eye(len(lb)), b=-h, c=None, d=-k), h, k, r
        # check if the centroid is within the box
        # feasible = feasiblePoint(h, A, b)
        # feasible = True
        # if feasible:
        #     # now we check whether all the vertices are within the ball 
        #     V = constraintToVertices(A, b)

        #     return numpy.all(map(numpy.linalg.norm, V - h) < r), h, k, r
        # else:
        #     return False, h, k, r

def _concaveExclusion(f, fStar, L, g, x, lb, ub, epsilon=1.0):
    '''
    Convex in the sense that area outside the ball \| x - h \|^{2} + k \ge 0
    is not the area of interest
    '''
    k = 2.0/L * (f - fStar + epsilon - numpy.dot(g,g)/(2.0*L))
    h = x - g/L

    if k >= 0.0:
        # the inequality is always satisfied.  
        return True, h, k, 0.0
    else: 
        # radius of the ball that tells us the size of non-exlucsion area
        r = numpy.sqrt(-k)
        A, b = addLBUBToInequality(lb, ub)
        return not polytopeInSCO(A, b, numpy.eye(len(lb)), b=-h, c=None, d=-k), h, k, r
        
def underestimateFunc(box, grad, hess, LType="exact"):
    f = box.getFx()
    c = box.getLocation()

    g = _checkFuncAndEvaluate(grad, c)
    H = _checkFuncAndEvaluate(hess, c)

    if LType.lower() == 'exact':
        L = min(numpy.linalg.eigvalsh(H))
        # L = _exact(H)
    elif LType.lower() == 'conservative':
        # L = -_conservative(H)
        L = max(abs(numpy.linalg.eigvalsh(H)))
    else:
        raise Exception("Type of calculation for Lipschitz gradient not recognized")

    return quadApproxWrapper(c, f, g, L)

def quadApproxWrapper(c, func, grad, hess):
    f = _checkFuncAndEvaluate(func, c)
    g = _checkFuncAndEvaluate(grad, c)
    H = _checkFuncAndEvaluate(hess, c)
    return functools.partial(quadApprox, c=c, f=f, g=g, H=H)

def quadApprox(x, c, f, g, H):
    y = x - c
    return f + numpy.dot(g, y) + 0.5*numpy.dot(y,numpy.dot(H,y))

def _exact(H):
    e = numpy.linalg.eigvalsh(H)
    Li = min(e)
    return Li

def _conservative(H):
    e = numpy.linalg.eigvalsh(H)
    Li = max(abs(e))
    return Li

def _checkFuncAndEvaluate(func, x):
    if hasattr(func, '__call__'):
        return func(x)
    else:
        return func



