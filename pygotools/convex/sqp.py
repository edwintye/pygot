
__all__ = [
    'sqp'
    ]

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch, sufficientNewtonDecrement, lineSearch
from pygotools.optutils.consMani import addLBUBToInequality
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian
from .approxH import *

from .convexUtil import _setup

import numpy

#from cvxopt.solvers import coneqp
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False

EPSILON = 1e-6

def sqp(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        disp=0, full_output=False):

    x = checkArrayType(x0)
    p = len(x)
    x = x.reshape(p,1)
    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)

    #G, h = addLBUBToInequality(lb,ub,G,h)

    # G = matrix(G)
    # h = matrix(h)
    
    # if A is not None:
    #     A = matrix(A)
    # if b is not None:
    #     b = matrix(b)

    g = numpy.zeros((p,1))
    H = numpy.zeros((p,p))

    oldFx = numpy.inf
    oldGrad = None
    deltaX = None
    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    step = 1

    if hessian is None:
        H = numpy.eye(len(x))

    while abs(fx-oldFx)>=EPSILON:

        g[:] = grad(x)

        if hessian is None:
            if oldGrad is not None:
                # print "Update"
                diffG = g - oldGrad
                #H = SR1(H, diffG, deltaX)
                #H = SR1Alpha(H, diffG, deltaX)
                H = BFGS(H, diffG.ravel(), step * deltaX.ravel())
                #H = DFP(H, diffG, deltaX)
            #print H
            #H = forwardGradCallHessian(grad, x)
        else:
            H = hessian(x)

        # readjust the bounds
        if G is not None:
            hTemp = h - G.dot(x)
            dims = {'l': G.shape[0], 'q': [], 's':  []}
        else:
            hTemp = None
            dims = []

        if A is not None:
            bTemp = b - A.dot(x)
        else:
            bTemp = None

        # solving the QP to get the descent direction
        if A is not None:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, matrix(A), matrix(bTemp))
        else:
            if G is not None:
                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims)
            else:
                qpOut = solvers.coneqp(matrix(H), matrix(g))

        # exact the descent diretion and do a line search
        deltaX = numpy.array(qpOut['x'])
        oldFx = fx
        oldGrad = g.copy()
        # print oldGrad

        #step, fx = exactLineSearch(1, x, deltaX, func)
        lineFunc = lineSearch(1, x, deltaX, func)
        #step, fx = exactLineSearch(1, lineFunc, deltaX.ravel().dot(g.ravel()))
        step, fx = backTrackingLineSearch(1, lineFunc, deltaX.ravel().dot(g.ravel()))
        #print qpOut

        x += step * deltaX
        i += 1
        dispObj.d(i, numpy.array(qpOut['z']).ravel(), fx, deltaX.ravel(), g.ravel())
        
        if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
            break
        
        if i >= maxiter:
            break

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        
        output['H'] = H
        output['g'] = g.ravel()
        output['fx'] = fx
        output['iter'] = i
        if G is not None:
            output['z'] = numpy.array(qpOut['z']).ravel()
            output['s'] = (h - G.dot(x)).ravel()
        if A is not None:
            output['y'] = numpy.array(qpOut['y']).ravel()

        return x, output
    else:
        return x
