
__all__ = [
           'sqp'
           ]

from pygotools.optutils.optCondition import exactLineSearch
from pygotools.optutils.consMani import addLBUBToInequality
from pygotools.optutils.checkUtil import *
from pygotools.gradient.finiteDifference import forwardGradCallHessian

from cvxopt import coneqp

def sqp(func, grad, hessian=None, x0=None, lb=None, ub=None, G=None, h=None, A=None, b=None):

    theta = checkArrayType(x0)
    G, h = addLBUBToBox(lb,ub,G,h)

numpy.array(delta).dot(o['grad'])

    for i in range(10):
        theta = numpy.array(theta)
        if hessian is None:
            H = forwardGradCallHessian(grad, theta)
        else:
            H = hessian(x)

        g = grad(x)
        if G is not None:
            # readjust the bounds
            hTemp = h - G*matrix(theta)
        if A is not None:
            bTemp = b - A*matrix(theta)

        # solving the QP to get the descent direction
        qpOut = coneqp(matrix(H), matrix(g), G, hTemp, [], A, bTemp)
        # exact the descent diretion and do a line search
        deltaX = numpy.array(qpOut['x']).ravel()
        t, fx = exactListSearch(1,theta,deltaX,func)
    
        theta += t * delta

    return theta


