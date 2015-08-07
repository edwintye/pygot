
__all__ = [
    'ipBar'
    ]

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch2, lineSearch, sufficientNewtonDecrement
from pygotools.optutils.checkUtil import checkArrayType, _checkFunction2DArray
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian, forward

from .ipUtil import _logBarrier, _logBarrierGrad, _findInitialBarrier, _dualityGap, _rDualFunc
from .ipUtil import _solveSparseAndRefine, _residualLineSearchPDC
from .convexUtil import _checkInitialValue,  _setup, _checkFuncGradHessian
from .approxH import *

import numpy
import scipy.sparse, scipy.linalg, scipy.optimize

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

    func, grad, hessian, approxH = _checkFuncGradHessian(x, func, grad, hessian)
    
    if G is not None:
        m = G.shape[0]
    else:
        m = 1

    fx = None
    deltaY = None
    deltaZ = None
    oldFx = None
    oldOldFx = None 
    oldGrad = None
    oldX = None
    barSlope = None
    deltaX = numpy.zeros((p,1))
    g = numpy.zeros((p,1))
    H = numpy.zeros((p,p))
    Haug = numpy.zeros((p,p))

    output = dict()
    if disp is None:
        disp = 0
    dispObj = Disp(disp)
    t = 0.01
    mu = 10.0
    step0 = 1.0  # back tracking search step maximum value
    step = 0.0
    dGap = None  # duality gap

    if G is not None:
        s = h - G.dot(x)
        Gs = G/s
        Dphi = Gs.sum(axis=0).reshape(p,1)
        t = _findInitialBarrier(grad(x).reshape(p,1),Dphi,A)
        # print "Initial barrier = "+str(t)

    j = 0
    i = 0
    while maxiter>=j:
        # define the barrier function given t.  Note that
        # t is adjusted at each outer iteration
        barrierFunc = _logBarrier(func, t, G, h)
        #if j==0:
        fx = barrierFunc(x)
    
        if j!=0:
            oldFx = barrierFunc(oldX)
            oldOldFx = None
            
        #else:
            
        update = True
        numUpdate = 0
        while update:
            gOrig = grad(x).reshape(p,1)

            if hessian is None:
                if oldGrad is None:
                    H = numpy.eye(len(x))
                else:
                    diffG = numpy.array(gOrig - oldGrad).ravel()
                    H = approxH(H, diffG, step * deltaX.ravel())
            else:
                H = hessian(x)

            ## standard log barrier
            if G is not None:
                s = h - G.dot(x)
                Gs = G/s
                s2 = s**2
                Dphi = Gs.sum(axis=0).reshape(p,1)
                # if j==0:
                #     t = _findInitialBarrier(grad(x).reshape(p,1),Dphi,A)
                Haug = t*H + numpy.einsum('ji,ik->jk',G.T, G/s2)
                g = t*gOrig + Dphi
            else:
                Haug = t*H
                g = t*gOrig

        ## solving the QP to get the descent direction
            if A is not None:
                # re-adjust the bounds
                bTemp = A.dot(x) - b
                LHS = scipy.sparse.bmat([
                                         [Haug,A.T],
                                         [A,None]
                                         ], 'csc')
                RHS = numpy.append(g, bTemp, axis=0)
            else:
                LHS = Haug
                RHS = g
            # finish setup
            deltaTemp = _solveSparseAndRefine(LHS, -RHS)
            deltaX = deltaTemp[:p]
            if A is not None:
                y = deltaTemp[p::]
                # deltaY = deltaTemp[p::]

            oldOldFx = oldFx
            oldFx = fx

            # oldOldFxTemp = oldFx
            # oldFx = fx
            oldGrad = gOrig
            oldX = x.copy()

            if A is None:
                lineFunc = lineSearch(x, deltaX, barrierFunc)
                barrierGrad = _logBarrierGrad(func, gOrig, t, G, h)
                # print "fx = " +str(fx)+ " and oldFx = "+str(oldFx)+ " and oldOldFx = "+str(oldOldFx)
                step, fc, gc, fx, oldFx, barSlope = scipy.optimize.line_search(barrierFunc,
                                                                            barrierGrad,
                                                                            x.ravel(),
                                                                            deltaX.ravel(),
                                                                            barSlope,  # g.ravel(),
                                                                            oldFx,
                                                                            oldOldFx
                                                                            )
                
                if step is None:
                    # if we know that it is hard to find a sufficient decrease, so low alpha
                    step, fx =  backTrackingLineSearch(step0, lineFunc, None, alpha=0.0001, beta=0.8)
                    # print "backtrack, with step = "+str(step)+ " and fx = " +str(barrierFunc(x+step*deltaX))+ " and oldFx = "+str(barrierFunc(x))+"\n"
                    update = False
            else:
                lineFunc = _residualLineSearchPDC(x, deltaX,
                                          grad, t,
                                          z, deltaZ, None, h,
                                          y, 0.0, A, b)
                searchScale = None
                step, fx = exactLineSearch2(1.0, lineFunc, searchScale, oldFx)
                # y += step * deltaY
                    
            # oldOldFx = oldOldFxTemp
            x += step * deltaX
            j += 1
            # dispObj.d(j, x.ravel() , fx, deltaX.ravel(), g.ravel(), step)
            dispObj.d(j, x.ravel(), func(x.ravel()), deltaX.ravel(), g.ravel(), step)
            if abs((oldFx-fx)/fx)<=EPSILON:
                break

            ## we don't want to waste too much time on solving the intermediate problem
            if numUpdate<2:
                numUpdate += 1
            else:
                break

            ##########
            ## end of inner iteration
            ##########
        i += 1
        # obtain the missing Lagrangian multiplier
        if G is not None: 
            s = h - G.dot(x)
            z = 1.0 / (t * s)
        
        if G is not None or A is not None:
            dGap = _dualityGap(func, x,
                              z, G, h,
                              y, A, b)
            #print "dual gap = "+str(dGap)+ " and m/t = " +str(m/t)        
        
        if m/t < atol:# and dGap < atol:
            if G is None:
                if sufficientNewtonDecrement(deltaX.ravel(),gOrig.ravel()):
                    output['message'] = "Sufficient Newton decrement smaller than epsilon"
                    break
            else:
                # print scipy.linalg.norm(gOrig.ravel())**2
                if (scipy.linalg.norm(gOrig.ravel())**2)<=EPSILON:
                    output['message'] = 'Norm of gradient less tan epsilon'
                    break
#                 print "dual gap = "+str(dGap)+ " and m/t = " +str(m/t)
#                 break
            t *= mu
        else:
            t *= mu

        # print scipy.linalg.norm(_rDualFunc(x, grad, z, G, y, A))
        
        if scipy.linalg.norm(_rDualFunc(x, grad, z, G, y, A))<=EPSILON:
            output['message'] = 'Norm of dual residual less than epsilon'
            break
        
        if numpy.any(y<0):
            output['note'] = 'Negative Lagrangian multiplier'
            # break
        
        ##########
        # end of outer iteration
        ##########

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

        output['dgap'] = dGap            
        output['subopt'] = m/t
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
