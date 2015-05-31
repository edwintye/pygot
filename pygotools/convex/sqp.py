
__all__ = [
    'sqp'
    ]

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch, exactLineSearch2, sufficientNewtonDecrement, lineSearch
from pygotools.optutils.consMani import addLBUBToInequality, feasiblePoint, feasibleStartingValue
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forward, forwardGradCallHessian
from .approxH import *

from .convexUtil import _setup, _checkInitialValue

import numpy
import scipy.linalg
from scipy.optimize import line_search

#from cvxopt import solvers, matrix

from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False
solvers.options['reltol'] = 1e-6
solvers.options['reltol'] = 1e-6

EPSILON = 1e-6
maxRadius = 1000.0

def sqp(func, grad=None, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        method='trust',
        disp=0, full_output=False):

    if method.lower()=='trust' or method.lower()=='line':
        pass
    else:
        raise Exception("Input method not recognized")
    

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
    oldOldFx = numpy.inf
    oldGrad = None
    update = True
    deltaX = numpy.zeros((p,1))
    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    step = 1.0
    radius = 1.0

    if hessian is None:
        H = numpy.eye(len(x))

    while maxiter>=i:

        g[:] = grad(x).reshape(p,1)

        if hessian is None:
            if oldGrad is not None:
                # print update
                # print g
                # print oldGrad
                if update:
                    diffG = (g - oldGrad).ravel()
                    H = approxH(H, diffG, step * deltaX.ravel())
        else:
            H = hessian(x)

        if method=='trust':
            x, update, radius, deltaX, z, y, fx, oldFx, oldGrad = _updateTrustRegion(x, fx, oldFx, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b)
        else:
            x, deltaX, z, y, fx, oldFx, oldOldFx, oldGrad, step = _updateLineSearch(x, fx, oldFx, oldOldFx, deltaX, g, H, func, grad, z, G, h, y, A, b)

        # print qpOut
        # print "b Temp" 
        # print bTemp
        # print "b" 
        # print b - A.dot(x)

        i += 1
        dispObj.d(i, x , fx, deltaX.ravel(), g.ravel(), radius)

        # print "s"
        # print h - G.dot(x)
        # print "z"
        # print numpy.array(qpOut['z']).ravel()
        
        if sufficientNewtonDecrement(deltaX.ravel(),g.ravel()):
            break
        
        if abs(fx-oldFx)<=EPSILON:
            break

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        
        output['H'] = H
        output['g'] = g.flatten()

        output['fx'] = fx
        output['iter'] = i
        if G is not None:
            output['z'] = z.flatten()
            output['s'] = (h - G.dot(x)).flatten()
        if A is not None:
            output['y'] = y.flatten()

        return x, output
    else:
        return x


def _updateLineSearch(x, fx, oldFx, oldOldFx, oldDeltaX, g, H, func, grad, z, G, h, y, A, b):

    initVals = dict()
    initVals['x'] = matrix(oldDeltaX)
    # readjust the bounds and initial value if possible
    # as we try our best to use warm start
    if G is not None:
        hTemp = h - G.dot(x)
        dims = {'l': G.shape[0], 'q': [], 's':  []}
        initVals['z'] = matrix(z)
        s = hTemp - G.dot(oldDeltaX)

        while numpy.any(s<=0.0):
            oldDeltaX *= 0.5
            s = h - G.dot(oldDeltaX)
        initVals['s'] = matrix(s)
        initVals['x'] = matrix(oldDeltaX)

        #print initVals['s']
    else:
        hTemp = None
        dims = []

    if A is not None:
        initVals['y'] = matrix(y)
        bTemp = b - A.dot(x)
    else:
        bTemp = None

    # solving the QP to get the descent direction

    try:
        if A is not None:
            if G is not None:
                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, matrix(A), matrix(bTemp))
            else:
                qpOut = solvers.coneqp(matrix(H), matrix(g), None, None, None, matrix(A), matrix(bTemp))
        else:
            if G is not None:
                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, initvals=initVals)
            else:
                qpOut = solvers.coneqp(matrix(H), matrix(g))
    except Exception as e:
        if type(e) is ValueError:
            print scipy.linalg.eig(H)[0]
        #print "H"
        #print H
        #print "H eigenvalue"
        #print scipy.linalg.eig(H)[0]
        raise e

    # exact the descent diretion and do a line search
    deltaX = numpy.array(qpOut['x'])
    oldOldFx = oldFx
    oldFx = fx
    oldGrad = g.copy()
    # print oldGrad

    lineFunc = lineSearch(1, x, deltaX, func)
    #step, fx = exactLineSearch(1, x, deltaX, func)
    # step, fc, gc, fx, oldFx, new_slope = line_search(func,
    #                                                  grad,
    #                                                  x.ravel(),
    #                                                  deltaX.ravel(),
    #                                                  g.ravel(),
    #                                                  oldFx,
    #                                                  oldOldFx)

    # print step
    # if step is None:
    # step, fx = exactLineSearch2(1, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)
    # step, fx = exactLineSearch(1, lineFunc)
    # if fx >= oldFx:
    step, fx = backTrackingLineSearch(1, lineFunc, deltaX.ravel().dot(g.ravel()), alpha=0.0001)
    # print qpOut

    x += step * deltaX
    if G is not None:
        z[:] = numpy.array(qpOut['z'])
    if A is not None:
        y[:] = numpy.array(qpOut['y'])

    return x, deltaX, z, y, fx, oldFx, oldOldFx, oldGrad, step

def _updateTrustRegion(x, fx, oldFx, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b):

    # initVals = dict()
    # initVals['x'] = matrix(oldDeltaX)
    # readjust the bounds and initial value if possible
    # as we try our best to use warm start
    GTemp = numpy.append(numpy.zeros((1,p)), numpy.eye(p), axis=0)
    hTemp = numpy.zeros(p+1)
    hTemp[0] += radius

    if G is not None:
        GTemp = numpy.append(G, GTemp, axis=0)
        hTemp = numpy.append(h - G.dot(x), hTemp)
        dims = {'l': G.shape[0], 'q': [p+1], 's':  []}
        # initVals['z'] = matrix(z)
        # initVals['s'] = matrix(hTemp - G.dot(oldDeltaX))
    else:
        dims = {'l': 0, 'q': [p+1], 's':  []}

    if A is not None:
        # initVals['y'] = matrix(y)
        bTemp = b - A.dot(x)
    else:
        bTemp = None

    # solving the QP to get the descent direction

    try:
        if A is not None:
            if G is not None:
                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims, matrix(A), matrix(bTemp))
            else:
                # qpOut = solvers.coneqp(matrix(H), matrix(g), None, None, None, matrix(A), matrix(bTemp))

                qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims, matrix(A), matrix(bTemp))
        else:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims)
            # qpOut1 = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp))
            # print qpOut['x']
            # print qpOut1['x']
    except Exception as e:
        #print "H"
        #print H
        #print "H eigenvalue"
        #print scipy.linalg.eig(H)[0]
        raise e

    # exact the descent diretion and do a line search
    deltaX = numpy.array(qpOut['x'])

    M = diffM(deltaX.flatten(), g.flatten(), H)
    
    # print x
    # print deltaX
    newFx = func(x + deltaX)
    predRatio = (fx - newFx) / M(deltaX)
    # print "fx = " +str(fx)+ " newFx = " +str(newFx)
    # print deltaX
    # print predRatio
        
    if predRatio>=0.75:
        #if tau>0.0:
        radius = min(2.0*radius, maxRadius)
    elif predRatio<=0.25:
        radius *= 0.25

    if predRatio>=0.25:
        oldGrad = g.copy()
        x += deltaX
        oldFx = fx
        fx = newFx
        update = True
    else:
        update = False

    if G is not None:
        z[:] = numpy.array(qpOut['z'])[G.shape[0]]
    if A is not None:
        y[:] = numpy.array(qpOut['y'])

    return x, update, radius, deltaX, z, y, fx, oldFx, oldGrad

def diffM(deltaX, g, H):
    def M(deltaX):
        m = -g.dot(deltaX) - 0.5 * deltaX.T.dot(H).dot(deltaX)
        return m
    return M

