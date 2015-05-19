__all__ = [
    'lineSearch',
    'exactLineSearch',
    'backTrackingLineSearch',
    'exactLineSearch2'
]

import scipy.optimize

def lineSearch(step, x, deltaX, func):
    '''
    Returns a callable which will take a single
    input argument and be used as a line search function.
    
    Parameters
    ----------
    t: numeric
        maximum descent step to take
    x: array like
        the parameters of interest
    deltaX: array like
        descent direction
    func:
        objective function :math:`f`
    '''
    def F(step):
        return func(x + step*deltaX)
    return F

def backTrackingLineSearch(step, func, s, alpha=0.1, beta=0.8):
    '''
    Back tracking line search with t as the maximum.  Continues
    until :math:`f(t) <= f(0) + \alpha t s` where :math:`s` is
    the scale which measures the expected decrease

    Parameters
    ----------
    t: numeric
        maximum descent step to take
    func: callable
        the function to be searched
    s: numeric
        such as :math:`d^{T} \nabla f(x)` where :math:`d`
        is the descent direction and :math:`f(x)` is the objective
        function.
    alpha: numeric
        also known as c1 in Armijo rule
    beta: numeric
        also known as c2 in Armijo rule.  Defaults to 0.8 which is
        a slow search

    Returns
    -------
    t: numeric
        step size
    fdeltaX: numeric
        f(t), which should be :math:`f(x+t\delta x)`
    '''
    fx = func(0)
    fdeltaX = func(step)

    while fdeltaX > fx + alpha * step * s:
        step *= beta
        fdeltaX = func(step)
        if step <= 1e-16:
            return step, fdeltaX
    return step, fdeltaX

def exactLineSearch(stepMax, func):
    '''
    Performes an exact line search that minimizes the input function
    suject to the maximum step size.

    Parameters
    ----------
    stepMax: numeric
        maximum step size
    func: callable
        f(x) - the function being searched

    Returns
    -------
    step: float
        size of step that is successful
    fx: float
        f(x) evaluated at the output step size
    '''
    res = scipy.optimize.minimize_scalar(func,
                                         method='bounded',
                                         bounds=(1e-12,stepMax),
                                         options={'maxiter':100})
    #print res 
    if res['x'] >= 1.0:
        return 1.0, float(res['fun'])
    else:
        return float(res['x']), float(res['fun'])

def exactLineSearch2(stepMax, func, searchScale, oldFx):
    '''
    Performes an exact line search that minimizes the input function
    suject to the maximum step size.  When it fails, i.e output step
    violates 0 < step <= stepMax, it tries a back tracking line search

    Parameters
    ----------
    stepMax: numeric
        maximum step size
    func: callable
        f(x) - the function being searched
    searchScale: numeric
        such as :math:`d^{T} \nabla f(x)` where :math:`d`
        is the descent direction and :math:`f(x)` is the objective
        function.

    Returns
    -------
    step: float
        size of step that is successful
    fx: float
        f(x) evaluated at the output step size
    '''    

    step, fx = exactLineSearch(stepMax, func)
    if step>=stepMax or step<=1e-15 or fx>oldFx:
        step, fx = backTrackingLineSearch(stepMax,
                                          func,
                                          searchScale)

    return step, fx

def sufficientNewtonDecrement(deltaX, grad):
    if abs(deltaX.dot(grad)) <= 1e-6:
        return True
    else:
        return False


