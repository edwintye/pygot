
__all__ = [
    'ip'
    ]

from pygotools.optutils.optCondition import exactLineSearch, backTrackingLineSearch, lineSearch, sufficientNewtonDecrement
from pygotools.optutils.consMani import addLBUBToInequality
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian

from .convexUtil import _logBarrier, _findInitialBarrier, _dualityGap, _setup
from .approxH import *

import numpy
import scipy.sparse, scipy.linalg

#from cvxopt.solvers import coneqp
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

EPSILON = 1e-6
maxiter = 100

def ip(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        disp=0, full_output=False):

    x = checkArrayType(x0)
    x = x.reshape(len(x),1)
    p = len(x)

    # if lb is not None or ub is not None:
    #     G, h = addLBUBToInequality(lb,ub,G,h)

    # if G is not None:
    #     G = matrix(G)
    #     m,p = G.size
    # else:
    #     m = 1.0

    # if h is not None:
    #     h = matrix(h)

    # if A is not None:
    #     A = matrix(A)
    # if b is not None:
    #     b = matrix(b)
    
    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)

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
    mu = 5.0
    step0 = 1.0  # back tracking search step maximum value
    step = 0.0

    while abs(fx-oldFx)>=EPSILON:

        gOrig = grad(x)

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
            if i==0:
                t = _findInitialBarrier(gOrig,Dphi,A)
        else:
            Haug = t*H
            g = t * gOrig

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
            LHS = scipy.sparse.bmat([[Haug,A.T],[A,None]])
            RHS = numpy.append(g,-bTemp,axis=0)
            # print LHS.dot(numpy.append(numpy.array(qpOut['x']),numpy.array(qpOut['y']),axis=0)) + RHS

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
        ## exact the descent diretion and do a line search
        
        oldFx = fx
        oldGrad = gOrig

        barrierFunc = _logBarrier(x, func, t, G, h)

        #step, fx = exactLineSearch(step0, x, deltaX, barrierFunc)
        lineFunc = lineSearch(step0, x, deltaX, barrierFunc)
        
        step, fx = backTrackingLineSearch(step0,
                                          lineFunc,
                                          deltaX.ravel().dot(g.ravel()))

        # print step
        # print type(fx)
        x += step * deltaX
        i += 1
        
        # print "augmented obj: "+ str(barrierFunc(x))
        # print "obj: "+str(func(x))
        # print "t = "+str(t)
        # print "gap = " +str(m/t)
        # print "step = " +str(step0) + "\n"
        #print s
        #print x

        #print qpOut
        # print "fx"
        dispObj.d(i, x.ravel() , fx, deltaX.ravel(), g.ravel())
        
        if m/t < EPSILON:
            if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
                break
        else:
            t *= mu
        
        if i >= maxiter:
            break

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t


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

        return x, output
    else:
        return x



