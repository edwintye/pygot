from cvxopt.coneprog import coneqp
from cvxopt.base import matrix

__all__ = [
    'ipPDandPDC'
    ]

from pygotools.optutils.optCondition import lineSearch, exactLineSearch2,exactLineSearch, backTrackingLineSearch
from pygotools.optutils.disp import Disp
from pygotools.optutils.checkUtil import _checkFunction2DArray
from pygotools.gradient.finiteDifference import forward 
from .approxH import *
from .convexUtil import _setup, _checkInitialValue, _checkFuncGradHessian
from .ipUtil import _logBarrier, _findInitialBarrier, _surrogateGap, _checkDualFeasible
from .ipUtil import _rDualFunc, _rCentFunc, _rCentFunc2, _rCentFuncCorrect, _rPriFunc, _deltaZFunc
from .ipUtil import _residualLineSearchPD, _residualLineSearchPDC
from .ipUtil import  _findStepSize, _solveSparseAndRefine, _updateVar
from .ipUtil import  _updateVarSimultaneous, _dualityGap

import numpy
import copy

import scipy.linalg, scipy.sparse

EPSILON = 1e-6
maxiter = 100

def ipPDandPDC(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        method="pd",
        disp=0, full_output=False):

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)
    x = _checkInitialValue(x0, G, h, A, b)
    p = len(x)

    func, grad, hessian, approxH = _checkFuncGradHessian(x, func, grad, hessian)

    if method.lower()=='pd' or method.lower()=='pdc':
        updateFunc = _solveKKTAndUpdatePD
    else:
        raise Exception("interior point update method not recognized")

    g = numpy.zeros((p,1))
    gOrig = g.copy()

    oldOldFx = None
    oldFx = None
    fx = None

    oldGrad = None
    deltaX = None
    deltaY = None
    deltaZ = None
    H = scipy.sparse.eye(len(x), format='csc')
    # print "Type of matrix = "+str(type(H))

    output = dict()
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
        # print "eta="+str(eta)

    while maxiter>i:
        gOrig[:] = grad(x).reshape(p,1)
        g[:] = gOrig.copy()
        
        if hessian is None:
            if oldGrad is None:
                H = scipy.sparse.eye(len(x), format='csc')
            else:
                diffG = (gOrig - oldGrad).ravel()
                # print diffG
                # print deltaX.ravel()
                H = approxH(H, diffG, step * deltaX.ravel())
        else:
            H = hessian(x)

#         eigenVal = scipy.linalg.eig(H)
#         if numpy.any(eigenVal<=0):
#             H += numpy.eye(p) * min(eigenVal)+1e-8
                
        oldOldFxTemp = oldFx
        # print "fx = " +str(fx)+ " and oldFx = "+str(oldFx)+ " and oldOldFx = "+str(oldOldFx)

        try:
            oldX = x.copy()
            if G is not None:
                oldZ = z.copy()
            
            x, y, z, fx, step, oldFx, oldGrad, deltaX = updateFunc(x, func, grad,
                                                                   fx, oldFx, oldOldFx,
                                                                   g, gOrig,
                                                                   H,
                                                                   z, G, h,
                                                                   y, A, b, t, method)
            
            # print g
            # print z
#             print x-oldX
#             print z-oldZ
#             print H
#             print g
            
            if step<=1e-15:
                if hessian is None:
                    H = scipy.sparse.eye(len(x), format='csc')
                    oldGrad = None
        except Exception as e:
            print "Positive descent direction, reset Hessian"
            # print hessian
            if hessian is None:
                H = scipy.sparse.eye(len(x), format='csc')
            else:
                raise Exception("User supplied Hessian is not PSD")
            x, y, z, fx, step, oldFx, oldGrad, deltaX = updateFunc(x, func, grad,
                                                                   fx, oldFx, oldOldFx,
                                                                   g, gOrig,
                                                                   H,
                                                                   z, G, h,
                                                                   y, A, b, t, method)

        oldOldFx = oldOldFxTemp
        
        i += 1
        dispObj.d(i, x , func(x), deltaX.ravel(), g.ravel(), step)
        # print "step = "+str(step)+ ", gap = "+str(_surrogateGap(x, z, G, h, y, A, b))+ " and t = "+str(t)+ "\n"

        feasible = False
        if G is not None:
            feasible, t = _checkDualFeasible(x, z, G, h, y, A, b, grad, m, mu)
            output['message'] = 'Dual feasibility less than epsilon'
        else:
            if oldFx is not None:
                if abs(fx-oldFx)<=EPSILON:
                    output['message'] = 'Absolute change in objective function less than epsilon'
                    break
            elif numpy.all(abs(oldX-x)<=1e-8):
                output['message'] = 'Change in x less than epsilon'
                break

        dGap = _dualityGap(func, x,
                           z, G, h,
                           y, A, b)
        # print "dual gap = "+str(dGap)+ ", surrogate gap = "+str(_surrogateGap(x, z, G, h, y, A, b))+ ", m/t = " +str(m/t)+ ", step = "+str(step)+"\n"

        if feasible:
            break
        
    # TODO: full_output- dual variables
    if full_output:
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
            output['s'] = (h - G.dot(x)).ravel()
            output['z'] = z.ravel()
            output['rDual'] = _rDualFunc(x, grad, z, G, y, A).ravel()
        if A is not None:
            output['rPri'] = _rPriFunc(x, A, b).ravel()
            output['y'] = y.ravel()

        return x.ravel(), output
    else:
        return x.ravel()


def _solveKKTAndUpdatePD(x, func, grad, fx, oldFx, oldOldFx, g, gOrig, Haug, z, G, h, y, A, b, t, method='pd'):
    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()
 
    # deltaX1, deltaY1, deltaZ1  = _solveKKTSystemPD(x.copy(), func, grad, g.copy(), Haug.copy(), z, G, h, y, A, b, t)

    ## TODO: there are slight numerical differences between PD and PDC which in turns leads to
    ## a non-convergence pd implementation
    if method=='pd':
        deltaX, deltaZ, deltaY  = _solveKKTSystemPD(x, func, grad, g.copy(), Haug.copy(), z, G, h, y, A, b, t)
    else:
        deltaX, deltaZ, deltaY = _solveKKTSystemPDC(x, func, grad, g, Haug, z, G, h, y, A, b, t)
 
#     print deltaX
#     print g
#     print deltaX.ravel().dot(g.ravel())
    if deltaX.ravel().dot(g.ravel())>0:
        # print HERE
        raise Exception("Positive descent direction")

    step, fx = _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx)
    x, y, z = _updateVar(x, deltaX, z, deltaZ, y, deltaY, step)

    return x, y, z, fx, step, oldFx, oldGrad, deltaX

def _solveKKTSystemPD(x, func, grad, g, Haug, z, G, h, y, A, b, t):
    p = len(x)
    deltaX = None
    deltaZ = None
    deltaY = None

    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
        
        Haug += numpy.einsum('ji,ik->jk', G.T, G*zs)
        Dphi = Gs.sum(axis=0).reshape(p, 1)
        g += Dphi / t

    # find the solution to a Newton step to get the descent direction
    if A is not None:
        bTemp = _rPriFunc(x, A, b)
        g += A.T.dot(y)
        # print "here"
        LHS = scipy.sparse.bmat([
                [Haug, A.T],
                [A, None]
                ],'csc')
        RHS = numpy.append(g, bTemp, axis=0)
    else:
        LHS = Haug
        RHS = g

    deltaTemp = _solveSparseAndRefine(LHS, -RHS)
    deltaX = deltaTemp[:p]
    if A is not None:
        deltaY = deltaTemp[p::]

    if G is not None:
        deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)

    return deltaX, deltaZ, deltaY

def _solveKKTSystemPDC(x, func, grad, g, Haug, z, G, h, y, A, b, t):
    p = len(x)
    deltaY = None
    deltaZ = None

    RHS = _rDualFunc(x, grad, z, G, y, A)
    
    if G is not None:
        s = h - G.dot(x)
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        # print rCent
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        rPri = _rPriFunc(x, A, b)
        RHS = numpy.append(RHS, rPri, axis=0)

        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T, A.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0), None],
                    [A, None, None]
                    ],'csc')
        else: # G is None
            LHS = scipy.sparse.bmat([
                    [Haug, A.T],
                    [A, None]
                    ],'csc')
    else: # A is None
        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0)],
                    ],'csc')
        else:
            LHS = Haug

    deltaTemp = _solveSparseAndRefine(LHS, -RHS)
    
    deltaX = deltaTemp[:p]
    if A is not None:
        if G is not None:
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else:
            deltaY = deltaTemp[p::]
    else:
        if G is not None:
            deltaZ = deltaTemp[p::]

    # print numpy.append(deltaX,deltaX1,axis=1)
    # print numpy.append(deltaZ,deltaZ1,axis=1)
    # print numpy.append(deltaY,deltaY1,axis=1)

    return deltaX, deltaZ, deltaY


