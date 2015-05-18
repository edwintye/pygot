
__all__ = [
    'ip'
    ]

from pygotools.optutils.optCondition import exactLineSearch, backTrackingLineSearch, lineSearch, sufficientNewtonDecrement
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian

from .convexUtil import _logBarrier, _findInitialBarrier, _dualityGap, _setup, _rDualFunc, _checkInitialValue
from .approxH import *

import numpy
import scipy.sparse, scipy.linalg

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

EPSILON = 1e-6
maxiter = 100

def ip(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        disp=0, full_output=False):

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)
    x = _checkInitialValue(x0, G, h, A, b)
    p = len(x)

    if G is not None:
        m = G.shape[0]
    else:
        m = 1

    if hessian is None:
        approxH = BFGS

    oldFx = numpy.inf
    fx = func(x)
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
    while scipy.linalg.norm(_rDualFunc(x, grad, z, G, y, A))>=EPSILON:

        oldFx = numpy.inf
        # define the barrier function given t.  Note that
        # t is adjusted at each outer iteration
        barrierFunc = _logBarrier(x, func, t, G, h)
        while abs(fx-oldFx)>=EPSILON:
            # print "t = " + str(t)
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

                Haug = t*H + numpy.einsum('ji,ik->jk',G.T, G/s2)
                Dphi = Gs.sum(axis=0).reshape(p,1)
                g = t * gOrig + Dphi
                if j==0:
                    t = _findInitialBarrier(gOrig,Dphi,A)
            else:
                Haug = t*H
                g = t * gOrig
            # print "after setup t = " +str(t)
        #Dphi = matrix(0.0,(p,1))
        #blas.gemv(Gs, matrix(1.0,(m,1)),Dphi,'T')

        # print type(g)
        # print type(t)
        # print type(y)

        # print type(g)
        ## solving the QP to get the descent direction
            if A is not None:
                # readjust the bounds
                bTemp = b - A.dot(x)
            #print 
            #print bTemp
            # qpOut = solvers.coneqp(matrix(Haug), matrix(g), None, None, None, matrix(A), matrix(bTemp))
            # deltaX = numpy.array(qpOut['x'])
            # print "cone"
            # print qpOut['x']
                LHS = scipy.sparse.bmat([[Haug,A.T],[A,None]],'csc')
                RHS = numpy.append(g,-bTemp,axis=0)
            # print LHS.dot(numpy.append(numpy.array(qpOut['x']),numpy.array(qpOut['y']),axis=0)) + RHS
                if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                    deltaTemp = scipy.linalg.solve(LHS.todense(),-RHS).reshape(len(RHS),1)
                else:    
                    deltaTemp = scipy.sparse.linalg.spsolve(LHS,-RHS).reshape(len(RHS),1)
                deltaX = deltaTemp[:p]
            # print "scipy"
            # print deltaX
            # print LHS.dot(deltaTemp)+RHS
                y = deltaTemp[p::]
            else:
            #try:
                deltaX = scipy.linalg.solve(Haug,-g)
            # qpOut = solvers.coneqp(matrix(Haug), matrix(g))
            # deltaX = numpy.array(qpOut['x'])
            # except Exception as e:
            #     print "Actual H"
            #     print H
            #     print "H"
            #     print numpy.linalg.eig(H)[0]
            #     print "H aug"
            #     print numpy.linalg.eig(Haug)[0]
            #     raise e
        ## exact the descent direction and do a line search
        
            oldFx = fx
            oldGrad = gOrig

            lineFunc = lineSearch(step0, x, deltaX, barrierFunc)
            step, fx = exactLineSearch(step0, lineFunc)
            # print step
            # print barrierFunc(x + 0.01*deltaX)
            # print barrierFunc(x + step0*deltaX)
            if fx >= oldFx:
                step, fx = backTrackingLineSearch(step0,
                                                  lineFunc,
                                                  deltaX.ravel().dot(g.ravel()))

        # print step
        # print type(fx)
            x += step * deltaX
            j += 1
            dispObj.d(j, x.ravel() , fx, deltaX.ravel(), g.ravel())
            # end of inner iteration
        i += 1
        # obtain the missing Lagrangian multiplier
        if G is not None: 
            s = h - G.dot(x)
            z = 1.0 / (t * s)
        # print "augmented obj: "+ str(barrierFunc(x))
        # print "obj: "+str(func(x))
        # print "t = "+str(t)
        # print "gap = " +str(m/t)
        # print "step = " +str(step0) + "\n"
        #print s
        #print x

        #print qpOut
        # print "fx"
        #dispObj.d(i, x.ravel() , fx, deltaX.ravel(), g.ravel())
        
        if m/t < EPSILON:
            if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
                break
        else:
            t *= mu
        
        if i >= maxiter:
            break

        # end of outer iteration

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t
        output['iter'] = i

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

        return x, output
    else:
        return x



