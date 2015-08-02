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
from .convexUtil import _setup, _logBarrier, _findInitialBarrier, _surrogateGap, _checkInitialValue
from .convexUtil import _rDualFunc, _rCentFunc, _rCentFunc2, _rCentFuncCorrect, _rPriFunc

import numpy

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
    
    func = _checkFunction2DArray(func, x)
    
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
    else:
        hessian = _checkFunction2DArray(hessian, x)

    if grad is None:
        def finiteForward(func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(func,p)
    else:
        grad = _checkFunction2DArray(grad, x)

    if method.lower()=='pd' or method.lower()=='pdc':
        updateFunc = _solveKKTAndUpdatePD
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

    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    mu = 1.0
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
            
        eigenVal = scipy.linalg.eig(H)
        if numpy.any(eigenVal<=0):
            H += numpy.eye(p) * min(eigenVal)+1e-8
                
        oldOldFxTemp = oldFx

        x, y, z, fx, step, oldFx, oldGrad, deltaX = updateFunc(x, func, grad,
                                                               fx, oldFx, oldOldFx,
                                                               g, gOrig,
                                                               H,
                                                               z, G, h,
                                                               y, A, b, t, method)

        oldOldFx = oldOldFxTemp
        
        i += 1
        dispObj.d(i, x , func(x), deltaX.ravel(), g.ravel(), step)

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

            #print "eta = " +str(eta)
            #print "r = " +str(r)
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

    if method=='pd':
        deltaX, deltaY, deltaZ  = _solveKKTSystemPD(x, func, grad, g.copy(), Haug.copy(), z, G, h, y, A, b, t)
    else:
        deltaX, deltaY, deltaZ = _solveKKTSystemPDC(x, func, grad, g.copy(), Haug.copy(), z, G, h, y, A, b, t)

    
    # deltaX1, deltaY1, deltaZ1 = _solveKKTSystemPDC(x, func, grad, g.copy(), Haug.copy(), z, G, h, y, A, b, t)

    # if deltaZ is not None:
    #     print "deltaZ, pd and pdc"
    #     print numpy.append(deltaZ,deltaZ1,axis=1)
    # print "deltaX, pd and pdc"
    # print numpy.append(deltaX,deltaX1,axis=1)
    # out = coneqp(matrix(Haug),matrix(g),matrix(G),matrix(h))
    # print out['x']
    # print z
    # print scipy.linalg.solve(Haug,-g)
    # print numpy.append(deltaX,deltaX1,axis=1)

    # the only difference is in solving the linear system
    step, fx = _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx)
    
    if step<1e-15:
        # switch to pure gradient update
        H = numpy.eye(len(x.ravel()))
        if method=='pd':
            deltaX, deltaY, deltaZ  = _solveKKTSystemPD(x, func, grad, g.copy(), H, z, G, h, y, A, b, t)
        else:
            deltaX, deltaY, deltaZ = _solveKKTSystemPDC(x, func, grad, g.copy(), H, z, G, h, y, A, b, t)
        
        step, fx = _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx)
    
    x, y, z = _updateVar(x, deltaX, y, deltaY, z, deltaZ, step)    
    
    return x, y, z, fx, step, oldFx, oldGrad, deltaX

def _solveKKTSystemPD(x, func, grad, g, Haug, z, G, h, y, A, b, t):
    p = len(x)
    
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        zs = z/s
        # now find the matrix/vector of our qp
        HaugOrig = Haug.copy()
        gOrig = g.copy()

        for i in range(len(z)):
            HaugOrig += z[i]/s[i] * numpy.outer(G[i],G[i])
        
        Haug += numpy.einsum('ji,ik->jk', G.T, G*zs)
        Dphi = Gs.sum(axis=0).reshape(p, 1)
        
        DphiA = numpy.zeros(p)
        # print DphiA
        for i in range(len(z)):
            # print G[i]/(s[i]*t)
            DphiA += G[i]/(s.ravel()[i]*t)

        g += Dphi / t

        # print DphiA.reshape(p,1) - Dphi
        # print HaugOrig - Haug

        # print scipy.linalg.solve(HaugOrig,-gOrig)
        # print scipy.linalg.solve(Haug,-g)

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
        # if the total number of elements (in sparse format) is
        # more than half total possible elements, it is a dense matrix
        if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2.0:
            deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        else:
            # deltaTemp = _solveRefine(LHS.todense(), -RHS)
            deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
        # print deltaTemp
        deltaX = deltaTemp[:p]
        deltaY = deltaTemp[p::]
    else:
        deltaX = _solveRefine(Haug, -g)
        deltaY = None

    if G is not None:
        deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    else:
        deltaZ = None

    return deltaX, deltaY, deltaZ

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
                    [-z*G, scipy.sparse.diags(s.ravel(),0), None],
                    [A, None, None]
                    ],'csc')
        else: # G is None
            LHS = scipy.sparse.bmat([
                    [Haug, A.T],
                    [A, None]
                    ],'csc')
        # solve
        deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
    else: # A is None
        if G is not None:
            LHS = scipy.bmat([
                    [Haug, G.T],
                    [-z*G, scipy.sparse.diags(s.ravel(),0).toarray()],
                    ])
        else:
            LHS = Haug
        # solve
        deltaTemp = scipy.linalg.solve(LHS, -RHS).reshape(len(RHS), 1)     

    deltaX = deltaTemp[:p]
    if G is not None:
        if A is not None:
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else:
            deltaZ = deltaTemp[p::]
    else:
        if A is not None:
            deltaY = deltaTemp[p::]

    return deltaX, deltaY, deltaZ

def _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx):

    if G is None:
        maxStep = 1
        barrierFunc = _logBarrier(func, t, G, h)
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
    step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)
    #step, fx = backTrackingLineSearch(maxStep, lineFunc, searchScale,
    #                                  alpha=0.0001, beta=0.8)
    step *= 0.5

    # print "step = "+str(step)+ " step1 = "+str(step1)
    # print "fx = "+str(fx)+ " fx1 = "+str(fx1)
    return step, fx


def _maxStepSizePDC(z, deltaZ, x, deltaX, G, h):
    index = deltaZ<0

    if any(index==True):
        step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    else:
        step = 1.0
        
#     print "deltaZ"
#     print deltaZ
#     print "step = " +str(step)
#     print "z"
#     print z
#     print "After step"
#     print z + step*deltaZ

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

def _deltaZFunc(x, deltaX, t, z, G, h):
    s = h - G.dot(x)
    rC = _rCentFunc(z, s, t)
    return z/s * G.dot(deltaX) - rC/s

def _maxStepSizePD(z, x, deltaX, t, G, h):
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    return _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)

def _residualLineSearchPD(x, deltaX,
                       gradFunc, t,
                       z, deltaZFunc, G, h,
                       y, deltaY, A, b):
    
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    return _residualLineSearchPDC(x, deltaX,
                                  gradFunc, t,
                                  z, deltaZ, G, h,
                                  y, deltaY, A, b)


def _updateVar(x, deltaX, y, deltaY, z, deltaZ, step):
    # found one iteration, now update the information
    if z is not None:
        z += step * deltaZ
    if y is not None:
        y += step * deltaY

    x += step * deltaX

    return x, y, z

def _solveRefine(A,b):
    lu, piv = scipy.linalg.lu_factor(A)
    x = scipy.linalg.lu_solve((lu, piv), b).reshape(len(b), 1)
    r = A.dot(x) - b
    d = scipy.linalg.lu_solve((lu, piv), r).reshape(len(b), 1)
    #print numpy.append(x,x-d,axis=1)
    return x - d
