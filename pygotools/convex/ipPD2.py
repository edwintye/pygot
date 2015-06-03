
__all__ = [
    'ipPD2'
    ]

from pygotools.optutils.optCondition import lineSearch, exactLineSearch2,exactLineSearch, backTrackingLineSearch, sufficientNewtonDecrement

from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forward, forwardGradCallHessian
from .approxH import *
from .convexUtil import _setup, _logBarrier, _findInitialBarrier, _surrogateGap, _checkInitialValue
from .convexUtil import _rDualFunc, _rCentFunc, _rPriFunc

import numpy

import scipy.linalg, scipy.sparse

from cvxopt import solvers

solvers.options['show_progress'] = True

EPSILON = 1e-6
maxiter = 100

def ipPD2(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        method="pd",
        disp=0, full_output=False):

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)
    x = _checkInitialValue(x0, G, h, A, b)
    p = len(x)

    if hessian is None:
        approxH = BFGS
    elif type(hessian) is str:
        if hessian.lower()=='bfgs':
            approxH = BFGS
        elif hessian.lower()=='sr1':
            approxH = SR1
        elif hessian.lower()=='dfp':
            approxH = DFP
        else:
            raise Exception("Input name of hessian is not recognizable")
        hessian = None

    if grad is None:
        def finiteForward(func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(func,p)

    if method=='pd':
        updateFunc = _solveKKTAndUpdatePD
    elif method=='pdc':   
        updateFunc = _solveKKTAndUpdatePDC
    else:
        raise Exception("interior point update method not recognized")


    g = numpy.zeros((p,1))
    gOrig = g.copy()

    oldOldFx = numpy.inf
    oldFx = numpy.inf
    oldGrad = None
    deltaX = None
    deltaY = 0
    deltaZ = 0
    H = numpy.zeros((p,p))
    Haug = H.copy()

    fx = func(x)

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
        #z = numpy.ones(s.shape)
        # print G.dot(x)
        # print s
        # print z
        m = G.shape[0]
        eta = _surrogateGap(x, z, G, h, y, A, b)
        # print eta
        t = mu * m / eta
        # print t

    while maxiter>i:

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
        oldOldFxTemp = oldFx
        x, y, z, fx, step, oldFx, oldGrad, deltaX = updateFunc(x, func, grad,
                                                               fx, oldFx, oldOldFx,
                                                               g, gOrig,
                                                               Haug,
                                                               z, G, h,
                                                               y, A, b, t)
        oldOldFx = oldOldFxTemp
        
        i += 1
        dispObj.d(i, x , fx, deltaX.ravel(), g.ravel(), step)

        feasible = False
        if G is not None:
            feasible = True
            eta = _surrogateGap(x, z, G, h, y, A, b)
            if eta >= EPSILON:
                feasible = False
            if G is not None:
                r = _rDualFunc(x, grad, z, G, y, A)
                if scipy.linalg.norm(r) >= EPSILON:
                    feasible = False
            if A is not None:
                r = _rPriFunc(x, A, b)
                if scipy.linalg.norm(r) >= EPSILON:
                    feasible = False

            t = mu * m / eta
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


def _deltaZFunc(x, deltaX, t, z, G, h):
    s = h - G.dot(x)
    rC = _rCentFunc(z, s, t)
    return z/s * G.dot(deltaX) + rC/-s

def _maxStepSizePD(z, x, deltaX, t, G, h):
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    index = deltaZ<0
    step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    
    s = h - G.dot(x + step * deltaX)
    while numpy.any(s<=0):
        step *= 0.8
        s = h - G.dot(x + step * deltaX)

    return step

def _residualLineSearchPD(x, deltaX,
                       gradFunc, t,
                       z, deltaZFunc, G, h,
                       y, deltaY, A, b):
    
    def F(step):
        newX = x + step * deltaX
        if G is not None:
            newZ = z + step * _deltaZFunc(x, deltaX, t, z, G, h)
        else:
            newZ = None

        r1 = _rDualFunc(newX, gradFunc, newZ, G, y, A).ravel()

        if y is not None:
            #newY = y + step * deltaY
            r2 = numpy.array(_rPriFunc(newX, A, b)).ravel()
        else:
            r2 = numpy.zeros(1)

        r = numpy.append(r1,r2,axis=0)
        return scipy.linalg.norm(r)
    return F

def _solveKKTAndUpdatePD(x, func, grad, fx, oldFx, oldOldFx, g, gOrig, Haug, z, G, h, y, A, b, t):
    p = len(x)
    step = 1.0
    deltaX = None
    deltaZ = None
    deltaY = None

    # standard log barrier, \nabla f(x) / -f(x)
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
        # now find the matrix/vector of our qp
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
        # print LHS
        # print RHS
        # if the total number of elements (in sparse format) is
        # more than half total possible elements, it is a dense matrix
        if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
            deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        else:
            deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        # print deltaTemp
        deltaX = deltaTemp[:p]
        deltaY = deltaTemp[p::]
    else:
        deltaX = scipy.linalg.solve(Haug, -g).reshape(p, 1)

    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()

    if G is None:
        maxStep = 1
        barrierFunc = _logBarrier(x, func, t, G, h)
        lineFunc = lineSearch(x, deltaX, barrierFunc)
        searchScale = deltaX.ravel().dot(g.ravel())
    else:
        maxStep = _maxStepSizePD(z, x, deltaX, t, G, h)
        lineFunc = _residualLineSearchPD(x, deltaX,
                                      grad, t,
                                      z, _deltaZFunc, G, h,
                                      y, deltaY, A, b)

        searchScale = -lineFunc(0.0)

    # perform a line search.  Because the minimization routine
    # in scipy can sometimes be a bit weird, we assume that the
    # exact line search can sometimes fail, so we do a
    # back tracking line search if that is the case
    step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)

    # step, fx =  exactLineSearch(maxStep, lineFunc)
    # if fx >= oldFx or step <=0:
    #     step, fx =  backTrackingLineSearch(maxStep, lineFunc, searchScale)

    # found one iteration, now update the information
    if z is not None:
        z += step * _deltaZFunc(x, deltaX, t, z, G, h)
    if y is not None:
        y += step * deltaY

    x += step * deltaX

    return x, y, z, fx, step, oldFx, oldGrad, deltaX


def _maxStepSizePDC(z, deltaZ, x, deltaX, G, h):
    index = deltaZ<0
    step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    
    s = h - G.dot(x + step * deltaX)
    while numpy.any(s<=0):
        step *= 0.8
        s = h - G.dot(x + step * deltaX)

    return step

def _residualLineSearchPDC(x, deltaX,
                       gradFunc, t,
                       z, deltaZ, G, h,
                       y, deltaY, A, b):
    
    def F(step):
        newX = x + step * deltaX
        if G is not None:
            newZ = z + step * deltaZ
        else:
            newZ = z

        if A is not None:
            newY = y + step * deltaY
        else:
            newY = y

        r1 = _rDualFunc(newX, gradFunc, newZ, G, newY, A)
        if G is not None:
            s = h - G.dot(newX)
            r2 = _rCentFunc(newZ, s, t)
            r1 = numpy.append(r1,r2,axis=0)

        if A is not None:
            r2 = _rPriFunc(newX, A, b)
            r1 = numpy.append(r1,r2,axis=0)

        return scipy.linalg.norm(r1)
    return F

def _solveKKTAndUpdatePDC(x, func, grad, fx, oldFx, oldOldFx, g, gOrig, Haug, z, G, h, y, A, b, t):
    p = len(x)
    step = 1
    deltaX = None
    deltaZ = None
    deltaY = None

    rDual = _rDualFunc(x, grad, z, G, y, A)
    RHS = rDual
    # RHS = g      
    
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
    
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        bTemp = b - A.dot(x)
        #g += A.T.dot(y)

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
        
    if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2.0:
        deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
        # print "here"
        invA = scipy.sparse.linalg.splu(LHS)
        deltaTemp1 = invA.solve(-RHS).reshape(len(RHS), 1)
        # print numpy.append(deltaTemp,deltaTemp1,axis=1)
    else:
        # print "there"
        # deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        lu,piv = scipy.linalg.lu_factor(LHS.todense())
        deltaTemp = scipy.linalg.lu_solve((lu,piv),-RHS).reshape(len(RHS), 1)
    # deltaTemp1 = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
    # print numpy.append(deltaTemp,deltaTemp1,axis=1)

    if G is not None:
        if A is not None:
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else:
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p::]
    else:
        if A is not None:
            deltaX = deltaTemp[:p]
            deltaY = deltaTemp[p::]
        else:
            deltaX = deltaTemp
    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()

    if G is None:
        # print "obj"
        maxStep = 1
        barrierFunc = _logBarrier(x, func, t, G, h)
        lineFunc = lineSearch(x, deltaX, barrierFunc)
        searchScale = deltaX.ravel().dot(g.ravel())
    else:
        maxStep = _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)
        lineFunc = _residualLineSearchPDC(x, deltaX,
                                          grad, t,
                                          z, deltaZ, G, h,
                                          y, deltaY, A, b)
        searchScale = -lineFunc(0.0)

    # perform a line search.  Because the minimization routine
    # in scipy can sometimes be a bit weird, we assume that the
    # exact line search can sometimes fail, so we do a
    # back tracking line search if that is the case

    step, fx =  backTrackingLineSearch(maxStep, lineFunc, searchScale, alpha=0.0001, beta=0.8)
    
    # step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)
    if z is not None:
        z += step * deltaZ
    if y is not None:
        y += step * deltaY
        
    x += step * deltaX

    return x, y, z, fx, step, oldFx, oldGrad, deltaX

def _solveKKTSystemPDC(x, func, grad, g, Haug, z, G, h, y, A, b, t):
    rDual = _rDualFunc(x, grad, z, G, y, A)
    RHS = rDual
    
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
    
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        bTemp = b - A.dot(x)
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
        
    if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2.0:
        deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
        # print "here"
        invA = scipy.sparse.linalg.splu(LHS)
        deltaTemp1 = invA.solve(-RHS).reshape(len(RHS), 1)
        # print numpy.append(deltaTemp,deltaTemp1,axis=1)
    else:
        # print "there"
        # deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        lu,piv = scipy.linalg.lu_factor(LHS.todense())
        deltaTemp = scipy.linalg.lu_solve((lu,piv),-RHS).reshape(len(RHS), 1)
    # deltaTemp1 = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
    # print numpy.append(deltaTemp,deltaTemp1,axis=1)

    if G is not None:
        if A is not None:
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else:
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p::]
    else:
        if A is not None:
            deltaX = deltaTemp[:p]
            deltaY = deltaTemp[p::]
        else:
            deltaX = deltaTemp

    return deltaX, deltaZ, deltaY
        

def _solveKKTAndUpdatePDC2(x, func, grad, fx, oldFx, oldOldFx, g, gOrig, Haug, z, G, h, y, A, b, t):
    p = len(x)
    step = 1
    deltaX = None
    deltaZ = None
    deltaY = None

    rDual = _rDualFunc(x, grad, z, G, y, A)
    RHS = rDual
    # RHS = g      
    
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
        # Dphi = Gs.sum(axis=0).reshape(p, 1)
        # RHS += Dphi
        # if A is not None:
        #     RHS += A.T.dot(y)
    
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        bTemp = b - A.dot(x)
        #g += A.T.dot(y)

        rPri = _rPriFunc(x, A, b)
        RHS = numpy.append(RHS, rPri, axis=0)

        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T, A.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0), None],
                    [A, None, None]
                    ],'csc')
            if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            else:
                deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
            # print deltaTemp
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else: # G is None
            LHS = scipy.sparse.bmat([
                    [Haug, A.T],
                    [A, None]
                    ],'csc')
            if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            else:
                deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS),1)
            # print deltaTemp
            deltaX = deltaTemp[:p]
            deltaY = deltaTemp[p::]
    else: # A is None
        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0)],
                    ],'csc')
            if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            else:
                deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
            # print deltaTemp
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p::]
        else:
            deltaX = scipy.linalg.solve(Haug, -RHS).reshape(len(RHS), 1)

    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()

    if G is None:
        # print "obj"
        maxStep = 1
        barrierFunc = _logBarrier(x, func, t, G, h)
        lineFunc = lineSearch(x, deltaX, barrierFunc)
        searchScale = deltaX.ravel().dot(g.ravel())
    else:
        maxStep = _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)
        lineFunc = _residualLineSearchPDC(x, deltaX,
                                          grad, t,
                                          z, deltaZ, G, h,
                                          y, deltaY, A, b)
        searchScale = -lineFunc(0.0)

    # perform a line search.  Because the minimization routine
    # in scipy can sometimes be a bit weird, we assume that the
    # exact line search can sometimes fail, so we do a
    # back tracking line search if that is the case
    
    # step, fx =  exactLineSearch(maxStep, lineFunc)
    # if fx >= oldFx or step <= 0 or step>=maxStep:
    #     step, fx =  backTrackingLineSearch(maxStep, lineFunc, searchScale, oldFx)
    step, fx =  backTrackingLineSearch(maxStep, lineFunc, searchScale, alpha=0.0001, beta=0.8)
    
    # step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)
    if z is not None:
        z += step * deltaZ
    if y is not None:
        y += step * deltaY
        
    x += step * deltaX

    # step, fc, gc, fx, old_fval, new_slope = scipy.optimize.line_search(func,
    #                                                                    grad,
    #                                                                    x.ravel(),
    #                                                                    maxStep * deltaX.ravel(),
    #                                                                    gOrig.ravel(),
    #                                                                    oldFx,
    #                                                                    oldOldFx
    #                                                                    )
    # if z is not None:
    #     z += maxStep * step * deltaZ
    # if y is not None:
    #     y += maxStep * step * deltaY
        
    # x += maxStep * step * deltaX

    return x, y, z, fx, step, oldFx, oldGrad, deltaX


