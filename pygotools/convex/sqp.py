
__all__ = [
    'sqp'
    ]

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch, sufficientNewtonDecrement, lineSearch
from pygotools.optutils.consMani import addLBUBToInequality, feasiblePoint, feasibleStartingValue
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forward, forwardGradCallHessian
from .approxH import *

from .convexUtil import _setup, _checkInitialValue

import numpy
import scipy.linalg

from cvxopt import solvers, matrix

solvers.options['show_progress'] = False


EPSILON = 1e-6

def sqp(func, grad=None, hessian=None, x0=None,
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
    if grad is None:
        def finiteForward(x,func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(x,func,p)
        
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

        g[:] = grad(x).reshape(p,1)

        if hessian is None:
            if oldGrad is not None:
                diffG = (g - oldGrad).ravel()
                # print "diffG"
                # print diffG
                # print "deltaX"
                # print deltaX
                H = approxH(H, diffG, step * deltaX.ravel())
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
        try:
            if A is not None:
                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, matrix(A), matrix(bTemp))
            else:
                if G is not None:
                    qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims)
                else:
                    qpOut = solvers.coneqp(matrix(H), matrix(g))
        except Exception as e:
            #print "H"
            #print H
            #print "H eigenvalue"
            #print scipy.linalg.eig(H)[0]
            raise e
        # print "H"
        # print H
        # exact the descent diretion and do a line search
        deltaX = numpy.array(qpOut['x'])
        oldFx = fx
        oldGrad = g.copy()
        # print oldGrad

        #step, fx = exactLineSearch(1, x, deltaX, func)
        lineFunc = lineSearch(1, x, deltaX, func)
        step, fx = exactLineSearch(1, lineFunc)
        if fx >= oldFx:
            step, fx = backTrackingLineSearch(1, lineFunc, deltaX.ravel().dot(g.ravel()))
        #print qpOut

        x += step * deltaX
        i += 1
        #dispObj.d(i, numpy.array(qpOut['z']).ravel(), fx, deltaX.ravel(), g.ravel())
        dispObj.d(i, x.ravel(), fx, deltaX.ravel(), g.ravel())
        # print "s"
        # print h - G.dot(x)
        # print "z"
        # print numpy.array(qpOut['z']).ravel()
        
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
