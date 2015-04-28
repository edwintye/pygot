
__all__ = [
    'linear',
    'closestVector'
    ]

import numpy
import scipy.linalg
#import sklearn.preprocessing

def linear(f, x0, h=None, S=None, lb=None, ub=None, *args):
    """
    Linear simplex gradient.  
    
    Parameters
    ----------
    f: callable
        The (scalar) function we wish to find the gradient of
    x0: array like
        parameter value at the center of the simplex
    S: array like
        direct of the simplex
    \*args: args, optional
        Any other arguments that are to be passed to `f`

    Returns
    -------
    grad: :class:`numpy.ndarray`
        array of gradient
        
    Notes
    -----
    Only accepts box constraints at the moment

    """
    
    # find out the number of parameters and convert
    x0, numParam = _checkArrayType(x0)

    if S is None:
        # assume that we want to use the maximum spanning set
        S = numpy.append(numpy.eye(numParam),-numpy.eye(numParam),axis=0)
        m = numParam * 2
    else:
        # dimension of the simplex 
        m = S.shape[0]
        # dimension of the parameters
        p = S.shape[1]
        if p!= numParam:
            raise Exception("The directions must be the same size as the parameters")
        
    # bounds checking, first the lower
    if lb is None:
        lb = numpy.zeros(numParam)
    else:
        lb,blah = _checkDimension(lb,x0)
        
    # now the upper
    if ub is None:
        ub = numpy.ones(numParam)
    else:
        ub,blah = _checkDimension(ub,x0)
        
    adjustedStepSize = 0
    if h is None:
        # the shortest distance to the boundary
        h = min(numpy.append(x0-lb,ub-x0))/2
        # a redefine and not just an adjustment
        adjustedStepSize = -1
    else:
        minDist = min(numpy.append(x0-lb,ub-x0,axis=0))
        if h > minDist:
            # output info that we adjusted
            adjustedStepSize = 1
            while h > minDist:
                h /= 2         

    # compute f0
    f0 = f(*((x0,) + args))
    
    # creating all the location / corners of our simplex
    X = x0 - h * S
    
    # going through the simplex
    fx = _getFx(X,m,*args)
        
    # print scipy.linalg.lstsq(X - x0,fx-f0)
    #beta,resid,r,s = scipy.linalg.lstsq(X - x0,fx-f0)
    beta = _getLSGrad(f,X,x0,fx,f0)

    info = dict()
    info['f0'] = f0
    info['fx'] = fx
    info['X'] = X
    info['S'] = S
    
    return beta, h, adjustedStepSize, info

def _getLSGrad(X,x0,fx,f0):
    beta,resid,r,s = scipy.linalg.lstsq(X - x0,fx-f0)
    return beta

def _getFx(f,X,m,*args):
    fx = numpy.zeros(m,float)
    for i in range(0,m):
        fx[i] =  f(*((X[i,:],) + args))
    
    return fx

def closestVector(A,x):
    '''
    Find the closest vector between each row of matrix A and the vector x.  
    Closeness is defined as the smallest angle between two vector
        
    Parameters
    ----------
    A: array like
        matrix of dimension (n,p)
    y: array like
        vector of dimension (p,)
        
    Returns
    -------
    y: :class:`numpy.array`
        The vector closest to x
    r: :class:`numpy.array`
        vector of angels in radian
    index: int
        the index value

    '''

    # normalize our vector
    x /= numpy.linalg.norm(x)
    # then our matrix
    #A = numpy.linalg.norm(A,axis=1)
    # A1 = sklearn.preprocessing.normalize(A,axis=1,norm='l2')
    A = numpy.divide(A.T,numpy.linalg.norm(A,axis=1)).T
    # print "normalization"
    # print A
    # print A1
    # find out the angel, in radian
    r = numpy.arccos(A.dot(x))
    # the argmin of it
    index = numpy.argmin(numpy.degrees(r))
    return A[index,:], r, index

def _checkDimension(x,y):
    '''
    Compare the length of two array like objects.  Converting both to a numpy
    array in the process if they are not already one.
        
    Parameters
    ----------
    x: array like
        first array
    y: array like
        second array
        
    Returns
    -------
    x: :class:`numpy.array`
        checked and converted first array
    y: :class:`numpy.array`
        checked and converted second array
        
    '''
    
    y,m = _checkArrayType(y)
    x,n = _checkArrayType(x)
    
    if n != m:
        raise Exception("The number of observations and time points should have the same length")
    
    return (x,y)

def _checkArrayType(x):
    '''
    Check to see Compare the length of two array like objects.  Converting both to a numpy
    array in the process if it is not already one.
        
    Parameters
    ----------
    x: array like
        which can be either a :class:`numpy.ndarray` or list or tuple
    
    Returns
    -------
    x: :class:`numpy.ndarray`
        checked and converted array
        
    '''
    
    if type(x) is numpy.ndarray:
        pass
    elif type(x) in (list,tuple):
        if type(x[0]) in (int,float):
            x = numpy.array(x)
        else:
            raise Exception("Expecting elements of float or int ")
    elif type(x[0]) in (int,float):
        x = numpy.array(list(x))
    else:
        raise Exception("Expecting an array like object")
    
    return x, len(x)
