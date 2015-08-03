
__all__ = [
    'ipPDC'
    ]

from pygotools.optutils.checkUtil import checkArrayType, _checkFunction2DArray
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian, forward

from .convexUtil import _setup, _checkInitialValue, _checkFuncGradHessian
from .ipUtil import  _logBarrier, _findInitialBarrier
from .ipUtil import _surrogateGap, _rDualFunc, _rCentFunc, _rPriFunc, _checkDualFeasible
from .ipUtil import _solveKKTAndUpdatePDC, _maxStepSizePDC, _residualLineSearchPDC

import numpy

import scipy.linalg, scipy.sparse

from cvxopt import solvers

solvers.options['show_progress'] = True

EPSILON = 1e-6

def ipPDC(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        disp=0, full_output=False):

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)
    x = _checkInitialValue(x0, G, h, A, b)
    p = len(x)
   
    func, grad, hessian, approxH = _checkFuncGradHessian(x, func, grad, hessian)
    fx = func(x)
    print "new version"
    
    g = numpy.zeros((p,1))
    gOrig = g.copy()

    oldFx = numpy.inf
    oldGrad = None
    deltaX = None
    deltaY = 0
    deltaZ = 0
    H = numpy.zeros((p,p))
    Haug = H.copy()

    dispObj = Disp(disp)
    i = 0
    mu = 5.0
    step = 1.0
    t = 1.0
    # because we determine the size of the back tracking
    # step on the fly, we don't give it a maximum.  At the
    # same time, because we are only evaluating the residuals
    # of the KKT system, there are times where we may want to
    # give the descent a nudge
    #step0 = 1.0  # back tracking search step maximum value

    if G is not None:
        s = h - G.dot(x)
        z = 1.0/s
        m = G.shape[0]
        eta = _surrogateGap(x, z, G, h, y, A, b)
        t = mu * m / eta

    while maxiter>=i:

        gOrig[:] = grad(x).reshape(p,1)
        g[:] = gOrig.copy()
        
        if hessian is None:
            if oldGrad is None:
                H = numpy.eye(len(x))
            else:
                diffG = (gOrig - oldGrad).ravel()
                H = approxH(H, diffG, step * deltaX.ravel())
        else:
            H = hessian(x)

        Haug[:] = H.copy()

        x, y, z, fx, step, oldFx, oldGrad, deltaX = _solveKKTAndUpdatePDC(x, func, grad,
                                                                          fx, g, gOrig, Haug,
                                                                          z, G, h, y,
                                                                          A, b, t)

        i += 1
        dispObj.d(i, x , fx, deltaX.ravel(), g.ravel(), step)

        if G is not None:
            feasible, t = _checkDualFeasible(x, z, G, h, y, A, b, grad, m, mu)
        else:
            if abs(fx-oldFx)<=EPSILON:
                break

        if feasible:
            break

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t

        if G is not None:
            gap = _surrogateGap(x, z, G, h, y, A, b)
        else:
            gap = 0

        output['dgap'] = gap
        output['fx'] = func(x)
        output['H'] = H
        output['g'] = gOrig.ravel()

        if G is not None:
            output['s'] = s.ravel()
            output['z'] = z.ravel()
            output['rDual'] = _rDualFunc(x, grad, z, G, y, A).ravel()
        if A is not None:
            output['rPri'] = _rPriFunc(x, A, b).ravel()
            output['y'] = y.ravel()

        return x.ravel(), output
    else:
        return x.ravel()
