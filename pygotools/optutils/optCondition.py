__all__ = [
    'lineSearch',
    'exactLineSearch'
    ]

import scipy.optimize

def lineSearch(t, theta, delta, func):
    def F(t):
        return func(theta + t*delta)
    return F

def backTrackingLineSearch(t, theta, delta, func, grad, alpha=0.1, beta=0.5):
    
    f = lineSearch(1,theta,delta,func)
    fx = f(0)
    fdeltaX = f(t)
    g = grad(theta)
    newtonDecrement = delta.dot(g)

    while fdeltaX > fx + alpha * t * newtonDecrement:
        t *= beta
        fdeltaX = f(t)

    return t, fdeltaX

def exactLineSearch(t, theta, delta, func):
    f = lineSearch(1,theta,delta,func)
    t, fx, ierr, numfunc = scipy.optimize.fminbound(f, 0, 1, full_output=True) 
    return t, fx

def sufficientNewtonDecrement(deltaX, grad):
    if abs(deltaX.dot(grad)) <= 1e-6:
        return True
    else:
        return False


