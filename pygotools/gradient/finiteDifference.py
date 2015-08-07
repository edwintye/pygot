
__all__ = [
    'forward',
    'backward',
    'central',
    'richardsonExtrapolation',
    'forwardHessian',
    'forwardGradCallHessian'
    ]

import numpy
import copy

def forward(f, x, h=None, *args):
    """
    Forward finite-difference approximation
    
    Parameters
    ----------
    f: callable
        The (scalar) function we wish to find the gradient of
    x: array like
        parameter value
    epsilon: array like
        epsilon used to compute the finite difference.  Does not 
        have a default value because it is only allowed in 
        python3 but not python2 :( WTF!  So we also allow the
        use of `None` here to force it to make its own mind
        up!! (aka default value)
    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    grad: :class:`numpy.ndarray`
        array of gradient

    """
    
    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    # f(x)
    #print(((x,) + args))
    f0 = f(*((x,) + args))
    #f0 = f(x,**kwargs)

    # epsilon checking and converting
    if h is None:
        h = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    else:
        h = numpy.array(h)
        if h.size==1:
            h = numpy.ones(numParam,float) * h

    # memory allocation
    grad = numpy.zeros(numParam, float)
    
    # going through the parameters
    for k in range(0,numParam):
        #xh = numpy.copy(x)
        xh = copy.deepcopy(x)
        xh[k] += h[k] 
        # print h
        # print x
        # print xh
        grad[k] = (f(*((xh,) + args)) - f0) / h[k]
        #grad[k] = (f(x,**kwargs) - f0) / h[k]
    
    return grad
    

def backward(f, x, epsilon=None, *args):
    """
    Backward finite-difference approximation
    
    Parameters
    ----------
    f: callable
        The (scalar) function we wish to find the gradient of

    x: array like

    epsilon: array like
        epsilon used to compute the finite difference.  Does not 
        have a default value because it is only allowed in 
        python3 but not python2 :( WTF!  So we also allow the
        use of `None` here to force it to make its own mind
        up!! (aka default value)

    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    grad: :class:`numpy.ndarray`
        array of gradient

    """

    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    f0 = f(*((x,) + args))

    # epsilon checking and converting
    if epsilon is None:
        epsilon = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    elif len(epsilon)==1:
        epsilon = numpy.ones(numParam,float) * epsilon
        
    grad = numpy.zeros(numParam, float)
    
    for k in range(0,numParam):
        xh = numpy.copy(x)
        xh[k] -= epsilon[k] 
        grad[k] = (f0 - f(*((xh,) + args))) / epsilon[k]
    
    return grad

def central(f, x, epsilon=None, *args):
    """
    Central finite-difference approximation
    
    Parameters
    ----------
    f: callable
        The (scalar) function we wish to find the gradient of

    x: array like

    epsilon: array like
        epsilon used to compute the finite difference.  Does not 
        have a default value because it is only allowed in 
        python3 but not python2 :( WTF!  So we also allow the
        use of `None` here to force it to make its own mind
        up!! (aka default value)

    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    grad: :class:`numpy.ndarray`
        array of gradient

    """

    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    #f0 = f(*((x,) + args))

    # epsilon checking and converting
    if epsilon is None:
        epsilon = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    elif len(epsilon)==1:
        epsilon = numpy.ones(numParam,float) * epsilon
        
    grad = numpy.zeros(numParam, float)
    
    # 
    for k in range(0,numParam):
        # f(x+h/2)
        xhPositive = numpy.copy(x)
        xhPositive[k] += + epsilon[k]/2
        # f(x-h/2)
        xhNegative = numpy.copy(x)
        xhNegative[k] -= epsilon[k]/2
        # now the difference for the gradient
        grad[k] = (f(*((xhPositive,) + args)) - f(*((xhNegative,) + args))) / epsilon[k]
    
    return grad


def richardsonExtrapolation(f, x, epsilon=None, *args):
    """
    Richardson extrapolation under forward finite-difference
    
    Parameters
    ----------
    f: callable
        The (scalar) function we wish to find the gradient of

    x: array like

    epsilon: array like
        epsilon used to compute the finite difference.

    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    grad: :class:`numpy.ndarray`
        array of gradient

    R: :class:`numpy.matrix`
        lower triangle of the extrapolation matrix.  First column is 
        the gradient. Difference in diagonal represent the error
        
    """
    TOLERANCE = numpy.sqrt(numpy.finfo(numpy.float).eps)

    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    f0 = f(*((x,) + args))
    #print(kwargs)
    #print(x)
    #print(dict(x=x))
    #print(dict(dict(x=x).items() + kwargs.items()))
    #f0 = f(**dict(dict(x=x).items() + kwargs.items()))
    #print(kwargs.items())
    #f0 = f(x,**kwargs)

    # epsilon checking and converting
    if epsilon is None:
        epsilon = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    elif len(epsilon)==1:
        epsilon = numpy.ones(numParam,float) * epsilon
        
    grad = numpy.zeros(numParam, float)
    Rlist = list()
    maxRow = 10

    for k in range(0,numParam):
        xh = numpy.copy(x)
        R = numpy.zeros((maxRow,maxRow),float)
        newEpsilon = epsilon
        xh[k] += newEpsilon[k]
        grad[k] = (f(*((xh,) + args)) - f0) / newEpsilon[k]
        #grad[k] = (f(**dict(dict(x=xh).items() + kwargs.items())) - f0) / epsilon[k]
        #grad[k] = ( f(xh,**kwargs) - f0) / epsilon[k]
        R[0,0] = grad[k]
        newEpsilon = epsilon
        for i in range(0,maxRow-1):
            xh = numpy.copy(x)
            newEpsilon[k] /= 2
            xh[k] +=  newEpsilon[k]
            grad[k] = (f(*((xh,) + args)) - f0) / newEpsilon[k]
            #grad[k] = (f(**dict(dict(x=xh).items() + kwargs.items())) - f0) / epsilon[k]
            #grad[k] = ( f(xh,**kwargs) - f0) / epsilon[k]
            R[i+1,0] = grad[k]
            #print("starting index " +str(i))
            #print("next index " +str(i+1))
            #print(R)
            for j in range(0,i+1):
                #print(R[i+1,j])
                #print(R[i,j])
                #print((4**(j+1)) * R[i+1,j] )
                #print(4**(j+1))
                #print(3*R[i+1,j])

                top = ((4**float(j+1))*R[i+1,j] - R[i,j])
                bottom = (4**float(j+1) - 1)
                
                #print(str(i+1) + " and " + str(j+1))
                #print(top)
                #print(bottom)
                if top==0:
                    R[i+1,j+1] = top
                else:
                    R[i+1,j+1] = top / bottom

            #print("Difference "+ str(R[i,i] - R[i+1,i+1]))
            if abs(R[i,i] - R[i+1,i+1]) < TOLERANCE:
                break     
            
        #print(R)
        Rlist.append(numpy.copy(R[0:i+2,0:i+2]))
        
    return grad,Rlist
    
def richardsonExtrapolationGeneric(f,g,h,**kwargs):
    ## TODO: fix or delete this... currently in a mess

    TOLERANCE = numpy.sqrt(numpy.finfo(numpy.float).eps)

    # epsilon checking and converting
    if h is None:
        h = numpy.sqrt(numpy.finfo(numpy.float).eps)
        
    Rlist = list()
    maxRow = 5

    R = numpy.zeros((maxRow,maxRow),float)
    hNew = numpy.copy(h)
    #out = f(*((hNew,) + args))
    #out = g(f=f,hNew,**kwargs)
    out = g(h=h,**dict(dict(f=f).items()+kwargs.items()))
    R[0,0] = out

    for i in range(0,maxRow-1):
        hNew /= 2
        #out = f(*((hNew,) + args))
        #out = g(f=f,hNew,**kwargs)
        out = g(hNew,**dict(dict(f=f).items()+kwargs.items()))
        R[i+1,0] = out
        for j in range(0,i):
            R[i+1,j+1] = ((4**j)*R[i,j] - R[i,j])/(4**j - 1)

        if abs(R[i,i] - R[i+1,i+1]) < TOLERANCE:
            Rlist.append(numpy.copy(R))
            break
            
    return (out,Rlist)

def forwardHessian(f, x, h=None, *args):
    """
    Forward finite-difference approximation of the Hessian
    
    Parameters
    ----------
    f: callable
        The (scalar) function 
    x: array like
        parameter value
    epsilon: array like
        epsilon used to compute the finite difference.  Does not 
        have a default value because it is only allowed in 
        python3 but not python2 :( WTF!  So we also allow the
        use of `None` here to force it to make its own mind
        up!! (aka default value)

    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    Hessian: :class:`numpy.ndarray`
        2d array of the Hessian

    """
    
    # TODO: check whether this is correct

    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    f0 = f(*((x,) + args))

    # epsilon checking and converting
    if h is None:
        h = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    else:
        h = numpy.array(h)
        if h.size==1:
            h = numpy.ones(numParam,float) * h

    # memory allocation
    perturb = numpy.zeros(numParam, float)
    
    # first, we wish to find the gradient without dividing through h
    for k in range(0,numParam):
        xh = numpy.copy(x)
        xh[k] += h[k]
        perturb[k] = f(*((xh,) + args))

    # print " with h "
    # print h
    # more memory allocation
    Hessian = numpy.zeros((numParam,numParam),float)
    for i in range(0,numParam):
        for j in range(i,numParam):
            xh = numpy.copy(x)
            xh[i] += h[i]
            xh[j] += h[j]
            Hessian[i,j] = (f(*((xh,) + args)) - perturb[i] - perturb[j] + f0) / (h[i] * h[j])
            # the other off diagonal
            Hessian[j,i] = Hessian[i,j]
    
    return Hessian


def forwardGradCallHessian(g, x, h=None, *args):
    """
    Forward finite-difference approximation of the Hessian using 
    perturbation on the gradient function
    
    Parameters
    ----------
    g: callable
        The gradient function 
    x: array like
        parameter value
    epsilon: array like
        epsilon used to compute the finite difference.  Does not 
        have a default value because it is only allowed in 
        python3 but not python2 :( WTF!  So we also allow the
        use of `None` here to force it to make its own mind
        up!! (aka default value)

    \*args: args, optional
        Any other arguments that are to be passed to `g`

    Returns
    -------
    Hessian: :class:`numpy.ndarray`
        2d array of the Hessian

    """
    
    # find out the number of parameters
    if type(x) is list:
        numParam = len(x)
        x = numpy.array(x)
    elif type(x) is numpy.ndarray:
        numParam = len(x)
    elif type(x) in (int,float):
        numParam = 1
        x = numpy.array([x])
    else:
        raise Exception("WTF is your input? Type = " + str(type(x)))

    g0 = g(*((x,) + args))

    # epsilon checking and converting
    if h is None:
        h = numpy.ones(numParam,float) * numpy.sqrt(numpy.finfo(numpy.float).eps)
    else:
        h = numpy.array(h)
        if h.size==1:
            h = numpy.ones(numParam,float) * h

    # memory allocation
    perturb = numpy.zeros((numParam,numParam), float)
    
    # first, we wish to find the gradient without dividing through h
    for k in range(0,numParam):
        xh = numpy.copy(x)
        xh[k] += h[k]
        perturb[k,:] = g(*((xh,) + args))

    # print g0
    # print perturb
    # print " with h "
    # print h
    # more memory allocation
    Hessian = numpy.zeros((numParam,numParam),float)
    for i in range(0,numParam):
        for j in range(i,numParam):
            Hessian[i,j] = (perturb[j,i] - g0[i]) / (2*h[j]) + (perturb[i,j] - g0[j]) / (2*h[i])
            # the other off diagonal
            Hessian[j,i] = Hessian[i,j]
    
    return Hessian



