
__all__ = [
    'ipBar'
    ]

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch2, lineSearch, sufficientNewtonDecrement
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian

from .convexUtil import _logBarrier, _logBarrierGrad, _findInitialBarrier, _dualityGap, _setup, _rDualFunc, _checkInitialValue
from .approxH import *

import numpy
import scipy.sparse, scipy.linalg

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

EPSILON = 1e-6
atol = 1e-6
rtol = 1e-4
maxiter = 100

def ipBar(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
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
        def finiteForward(x,func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(x,func,p)

    if G is not None:
        m = G.shape[0]
    else:
        m = 1

    fx = None
    oldFx = None
    oldOldFx = None 
    oldGrad = None
    deltaX = numpy.zeros((p,1))
    g = numpy.zeros((p,1))
    H = numpy.zeros((p,p))
    Haug = numpy.zeros((p,p))

    dispObj = Disp(disp)
    i = 0
    t = 0.01
    mu = 20.0
    step0 = 1.0  # back tracking search step maximum value
    step = 0.0

    j = 0
    while maxiter>=j:
        oldFx = numpy.inf
        # define the barrier function given t.  Note that
        # t is adjusted at each outer iteration
        barrierFunc = _logBarrier(func, t, G, h)
        if j==0:
            fx = barrierFunc(x)
            #print "barrier = " +str(fx)
        update = True
        #while (abs(fx-oldFx)/fx)>=rtol and abs(fx-oldFx)>=atol:
        # for i in range(1):
        while update:
            # print abs(fx-oldFx)
            # print abs(fx-oldFx)/fx
            # print fx
            # print oldFx
            gOrig = grad(x.ravel()).reshape(p,1)

            if hessian is None:
                if oldGrad is None:
                    H = numpy.eye(len(x))
                else:
                    diffG = numpy.array(gOrig - oldGrad).ravel()
                    H = approxH(H, diffG, step * deltaX.ravel())
            else:
                H = hessian(x.ravel())

            ## standard log barrier
            if G is not None:
                s = h - G.dot(x)
                Gs = G/s
                s2 = s**2
                Dphi = Gs.sum(axis=0).reshape(p,1)
                if j==0:
                    t = _findInitialBarrier(gOrig,Dphi,A)
                    # print "initial barrier = " +str(t)
                    # print "fake barrier = "+str(_findInitialBarrier(gOrig,Dphi,A))
                
                Haug = t*H + numpy.einsum('ji,ik->jk',G.T, G/s2)
                g = t*gOrig + Dphi

            else:
                Haug = t*H
                g = t*gOrig

        ## solving the QP to get the descent direction
            if A is not None:
                # re-adjust the bounds
                bTemp = b - A.dot(x)
                LHS = scipy.sparse.bmat([
                                         [Haug,A.T],
                                         [A,None]
                                         ], 'csc')
                RHS = numpy.append(g,-bTemp,axis=0)
                if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                    deltaTemp = scipy.linalg.solve(LHS.todense(),-RHS).reshape(len(RHS),1)
                else:    
                    deltaTemp = scipy.sparse.linalg.spsolve(LHS,-RHS).reshape(len(RHS),1)

                deltaX = deltaTemp[:p]
                y = deltaTemp[p::]
            else:
                deltaX = scipy.linalg.solve(Haug,-g)

            oldOldFxTemp = oldFx
            oldFx = fx
            oldGrad = gOrig

            lineFunc = lineSearch(x, deltaX, barrierFunc)
            #step, fx = exactLineSearch2(step0, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)

            # step, fx =  backTrackingLineSearch(step0, lineFunc, deltaX.ravel().dot(g.ravel()), alpha=0.0001,beta=0.8)
            
            barrierGrad = _logBarrierGrad(func, gOrig, t, G, h)
            step, fc, gc, fx, oldFx, new_slope = scipy.optimize.line_search(barrierFunc,
                                                                            barrierGrad,
                                                                            x.ravel(),
                                                                            deltaX.ravel(),
                                                                            g.ravel(),
                                                                            oldFx,
                                                                            oldOldFx
                                                                            )

            # print "fx = " +str(fx)
            # print "step= " +str(step)
            # if step is not None:
                # print "step = "+str(step)+ " with fx" +str(fx)+ " and barrier = " +str(barrierFunc(x + step * deltaX))
                # print "s"
                # print h - G.dot(x + step * deltaX)
            if step is None:
                # step, fx = exactLineSearch2(step0, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)
                step, fx =  backTrackingLineSearch(step0, lineFunc, deltaX.ravel().dot(g.ravel()), alpha=0.0001,beta=0.8)
                # print "fail wolfe = " +str(step)+ " maxStep = " +str(step0)                
                # print "fx = " +str(fx)
                # print "step= " +str(step)
                update = False
                
            oldOldFx = oldOldFxTemp
            x += step * deltaX
            # print "stepped func = "+str(func(x))
            j += 1
            # dispObj.d(j, x.ravel() , fx, deltaX.ravel(), g.ravel(), step)
            dispObj.d(j, x.ravel() , func(x.ravel()), deltaX.ravel(), g.ravel(), step)
            # end of inner iteration
        i += 1
        # obtain the missing Lagrangian multiplier
        if G is not None: 
            s = h - G.dot(x)
            z = 1.0 / (t * s)
        
        if m/t < atol:
            if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
                break
        else:
            t *= mu
        
        # print scipy.linalg.norm(_rDualFunc(x, grad, z, G, y, A))
        
        if scipy.linalg.norm(_rDualFunc(x, grad, z, G, y, A))<=EPSILON:
            break

        # end of outer iteration

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t
        output['outerIter'] = i
        output['innerIter'] = j

        if G is not None:
            s = h - G.dot(x)
            z = 1.0 / (t * s)
            output['s'] = s.ravel()
            output['z'] = z.ravel()

        if A is not None:
            y = y/t
            output['y'] = y.ravel()

        gap = _dualityGap(func, x,
                          z, G, h,
                          y, A, b)
        
        output['subopt'] = m/t
        output['dgap'] = gap
        output['fx'] = func(x)
        output['H'] = H
        output['g'] = gOrig.ravel()
        output['rDual'] = _rDualFunc(x, grad, z, G, y, A)

        return x.ravel(), output
    else:
        return x.ravel()


# we have included the infeasible step here
def _updateFeasibleNewton(x, gOrg, H, t, z, G, h, y, A, b):

    # standard log barrier
    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        s2 = s**2
        Dphi = Gs.sum(axis=0).reshape(p,1)
        if j==0:
            t = _findInitialBarrier(gOrig,Dphi,A)
            # print "initial barrier = " +str(t)
            # print "fake barrier = "+str(_findInitialBarrier(gOrig,Dphi,A))
            
        Haug = H/t + numpy.einsum('ji,ik->jk',G.T, G/s2)
        g = gOrig + Dphi
    else:
        Haug = H/t
        g = gOrig

    # solving the least squares problem to get the descent direction
    if A is not None:
        # re-adjust the bounds
        bTemp = b - A.dot(x)
        LHS = scipy.sparse.bmat([
                [Haug,A.T],
                [A,None]
                ], 'csc')
        RHS = numpy.append(g,-bTemp,axis=0)
        if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
            deltaTemp = scipy.linalg.solve(LHS.todense(),-RHS).reshape(len(RHS),1)
        else:    
            deltaTemp = scipy.sparse.linalg.spsolve(LHS,-RHS).reshape(len(RHS),1)
            
        deltaX = deltaTemp[:p]
        y = deltaTemp[p::]
    else:
        deltaX = scipy.linalg.solve(Haug,-g)

    oldOldFxTemp = oldFx
    oldFx = fx
    oldGrad = gOrig

    lineFunc = lineSearch(step0, x, deltaX, barrierFunc)
    # step, fx = exactLineSearch2(step0, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)
            
    # barrierGrad = _logBarrierGrad(x, func, gOrig, t, G, h)
    # step, fc, gc, fx, oldFx, new_slope = scipy.optimize.line_search(barrierFunc,
    #                                                                 barrierGrad,
    #                                                                 x.ravel(),
    #                                                                 deltaX.ravel(),
    #                                                                 g.ravel(),
    #                                                                 oldFx,
    #                                                                 oldOldFx
    #                                                                 )
            
    step, fx =  backTrackingLineSearch(step0, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)

    # if step is not None:
    # print "step = "+str(step)+ " with fx" +str(fx)+ " and barrier = " +str(barrierFunc(x + step * deltaX))
    # print "s"
    # print h - G.dot(x + step * deltaX)
    # if step is None:

        # step, fx = exactLineSearch2(step0, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)
        # print "fail wolfe = " +str(step)+ " maxStep = " +str(step0)
                
    oldOldFx = oldOldFxTemp
    x += step * deltaX

    return x, z, y, fx, oldFx, oldOldFx
