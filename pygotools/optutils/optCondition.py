__all__ = [
    'lineSearch',
    'exactLineSearch'
    ]

import scipy.optimize

def lineSearch(t, theta, delta, func):
    def F(t):
        return func(theta + t*delta)
    return F

def backTrackingLineSearch(t, theta, delta, func, g, alpha=0.1, beta=0.5):
    
    f = lineSearch(1,theta,delta,func)
    fx = f(0)
    fdeltaX = f(t)
    #g = grad(theta)
    newtonDecrement = delta.dot(g)

    while fdeltaX > fx + alpha * t * newtonDecrement:
        t *= beta
        fdeltaX = f(t)

    return t, fdeltaX

def exactLineSearch(t0, theta, delta, func):
    f = lineSearch(t0,theta,delta,func)
    res = scipy.optimize.minimize_scalar(f,bracket=(1e-8,t0))
    #print res 
    return res['x'], res['fun']

def sufficientNewtonDecrement(deltaX, grad):
    if abs(deltaX.dot(grad)) <= 1e-6:
        return True
    else:
        return False


