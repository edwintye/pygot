
__all__ = [
    'ip'
    ]

from pygotools.optutils.optCondition import exactLineSearch, backTrackingLineSearch, sufficientNewtonDecrement
from pygotools.optutils.consMani import addLBUBToInequality
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian

import numpy
import numpy.linalg

from cvxopt.solvers import coneqp
from cvxopt import matrix, mul, div
from cvxopt import blas

EPSILON = 1e-6
maxiter = 100

def ip(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        disp=0, full_output=False):

    x = checkArrayType(x0)
    if lb is not None or ub is not None:
        G, h = addLBUBToInequality(lb,ub,G,h)

    if G is not None:
        G = matrix(G)
        m,p = G.size
    else:
        m = 1.0

    if h is not None:
        h = matrix(h)

    if A is not None:
        A = matrix(A)
    if b is not None:
        b = matrix(b)

    oldFx = numpy.inf
    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    t = 0.01
    mu = 5.0
    step0 = 1.0  # back tracking search step maximum value

    def logBarrier(x,func,t,G,h):
        def F(x):
            s = h - G * matrix(x)
            s = numpy.array(s)
            if numpy.any(s<=0):
                return numpy.inf
            else:
                return t * func(x) - numpy.log(s).sum()
        return F


    while abs(fx-oldFx)>=EPSILON:

        if hessian is None:
            H = forwardGradCallHessian(grad, x)
        else:
            H = hessian(x)

        H = t * matrix(H)
        g = matrix(grad(x))

        # readjust the bounds
        if A is not None:
            bTemp = b - A * matrix(x)
        else:
            bTemp = None

        ## standard log barrier
        s = h - G * matrix(x)
        Gs = div(G,s[:,matrix(0,(1,p))])
        s2 = s**2

        H += matrix(numpy.einsum('ji,ik->jk',G.T, div(G,s2[:,matrix(0,(1,p))])))

        y = matrix(0.0,(p,1))
        blas.gemv(Gs, matrix(1.0,(m,1)),y,'T')
        if i==0:
            t = findInitialBarrier(g,y,A)
            # print "First iteration"
            # print float(numpy.linalg.lstsq(g, -y)[0])
            # t = float(numpy.linalg.lstsq(g, -y)[0][0][0])

        # print type(g)
        # print type(t)
        # print type(y)
        g = t * g + y
        # print type(g)
        ## solving the QP to get the descent direction
        if A is not None:
            qpOut = coneqp(H, g, [], [], [], A, bTemp)
        else:
            qpOut = coneqp(H, g)
        ## exact the descent diretion and do a line search
        deltaX = numpy.array(qpOut['x']).ravel()
        oldFx = fx

        barrierFunc = logBarrier(x, func, t, G, h)

        step, fx = exactLineSearch(step0, x, deltaX, barrierFunc)
        #step, fx = backTrackingLineSearch(step0, x, deltaX, barrierFunc, grad(x))
        x += step * deltaX
        i += 1
        
        s = h - G * matrix(x)
        # print "augmented obj: "+ str(barrierFunc(x))
        # print "obj: "+str(func(x))
        # print "t = "+str(t)
        # print "gap = " +str(m/t)
        # print "step = " +str(step0)
        # print s
        # print x

        #print qpOut
        dispObj.d(i, x , fx, deltaX, g)
        
        if m/t < EPSILON:
        #if sufficientNewtonDecrement(deltaX,g):
            break
        else:
            t *= mu
        
        if i >= maxiter:
            break

    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t

        y = numpy.array(qpOut['y']).ravel()/t
        s = numpy.array(h - G * matrix(x)).ravel()
        z = numpy.array(1.0 / (t * s))

        # gap = func(x)
        # if A is not None:
        #     gap += y.T.dot(numpy.array(A).dot(x) - b)
        # if G is not None:
        #     gap += z.T.dot(-s)

        gap = calculateDualityGap(func, x,
                                  z, G, h,
                                  y, A, b)

        output['s'] = s
        output['y'] = y
        output['z'] = z
        output['subopt'] = m/t
        output['dgap'] = gap
        output['fx'] = func(x)

        return x, output
    else:
        return x


def findInitialBarrier(g,y,A):
    if A is None:
        t = float(numpy.linalg.lstsq(g, -y)[0].ravel()[0])
    else:
        X = numpy.append(A,g,axis=1)
        t = float(numpy.linalg.lstsq(X, -y)[0].ravel()[-1])

    return t

def calculateDualityGap(func, x, z, G, h, y, A, b):
    gap = func(x)
    if A is not None:
        A = numpy.array(A)
        b = numpy.array(b).ravel()
        gap += y.dot(A.dot(x) - b)
        # print gap
    if G is not None:
        G = numpy.array(G)
        h = numpy.array(h).ravel()
        # print G
        # print h
        # print G.dot(x) - h
        gap += z.dot(G.dot(x) - h)
        # print gap

    return gap

