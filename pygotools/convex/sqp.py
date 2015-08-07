
__all__ = [
    'sqp'
    ]

from pygotools.optutils.optCondition import sufficientNewtonDecrement
from pygotools.convex.qpUpdate import _updateLineSearch, _updateTrustRegionSOCP, _updateTrustRegion
from pygotools.optutils.disp import Disp
from pygotools.optutils.checkUtil import _checkFunction2DArray
from pygotools.gradient.finiteDifference import forward
from .approxH import BFGS

from .convexUtil import _setup, _checkInitialValue, _checkFuncGradHessian

import numpy

EPSILON = 1e-8

def sqp(func, grad=None, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        method='trust',
        disp=0, full_output=False):

    if method.lower()=='trust' or method.lower()=='line':
        pass
    else:
        raise Exception("Input method not recognized")
    
    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)
    x = _checkInitialValue(x0, G, h, A, b)
    p = len(x)
    
    # print hessian(x.ravel())

    func, grad, hessian, approxH = _checkFuncGradHessian(x, func, grad, hessian)
#     if hessian is None:
#         approxH = BFGS
#     else:
#         hessian = _checkFunction2DArray(hessian, x)
#     if grad is None:
#         def finiteForward(x,func,p):
#             def finiteForward1(x):
#                 return forward(func,x.ravel())
#             return finiteForward1
#         grad = finiteForward(x,func,p)
#     else:
#         grad = _checkFunction2DArray(grad, x)
        
    g = numpy.zeros((p,1))
    H = numpy.zeros((p,p))

    fx = func(x)
    oldFx = numpy.inf
    oldOldFx = numpy.inf
    oldGrad = None
    update = True
    deltaX = numpy.zeros((p,1))
    # print x
    # print func(x)
        
    output = dict()
    if disp is None:
        disp = 0
    dispObj = Disp(disp)
    i = 0
    innerI = 0
    step = 1.0
    radius = 1.0

    if hessian is None:
        H = numpy.eye(p)

    while maxiter>=i:

        g = grad(x).reshape(p,1)

        if hessian is None:
            if oldGrad is None:
                H = numpy.eye(p)
            else:
                if update: # update is always true for line search
                    diffG = (g - oldGrad).ravel()
                    H = approxH(H, diffG, step * deltaX.ravel())
        else:
            H = hessian(x)
       
        if method=='trust':
            if hessian is None:
                # we assume that the Hessian is always a PSD
                # x, update, radius, deltaX, z, y, fx, oldFx, oldGrad, innerIter = _updateTrustRegion(x, fx, oldFx, deltaX, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b)
                x, update, radius, deltaX, z, y, fx, oldFx, oldGrad, innerIter = _updateTrustRegionSOCP(x, fx, oldFx, deltaX, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b)
            else:
                x, update, radius, deltaX, z, y, fx, oldFx, oldGrad, innerIter = _updateTrustRegionSOCP(x, fx, oldFx, deltaX, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b)
        else:
            x, deltaX, z, y, fx, oldFx, oldOldFx, oldGrad, step, innerIter = _updateLineSearch(x, fx, oldFx, oldOldFx, deltaX, g, H, func, grad, z, G, h, y, A, b)

        innerI += innerIter
        # print qpOut
        # print "b Temp" 
        # print bTemp
        # print "b" 
        # print b - A.dot(x)

        i += 1
        dispObj.d(i, x , fx, deltaX.ravel(), g.ravel(), radius)

        # print "s"
        # print h - G.dot(x)
        # print "z"
        # print numpy.array(qpOut['z']).ravel()
        if update:
            if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
                output['message'] = "Sufficient Newton decrement smaller than epsilon"
                break
        
            if abs(fx-oldFx)<=EPSILON:
                output['message'] = "Change in objective function smaller than epsilon"
                break
        else:
            if radius<=EPSILON:
                output['message'] = "Radius is too small = "+str(radius)
                break

    # TODO: full_output- dual variables
    if full_output:        
        output['H'] = H
        output['g'] = g.flatten()

        output['radius'] = radius
        output['fx'] = fx
        output['iter'] = i
        output['innerIter'] = innerI
        if G is not None:
            output['z'] = z.flatten()
            output['s'] = (h - G.dot(x)).flatten()
        if A is not None:
            output['y'] = y.flatten()

        return x.flatten(), output
    else:
        return x.flatten()

