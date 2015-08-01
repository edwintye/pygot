__all__ = [
    'checkArrayType',
    'checkDimension'
    ]

import numpy

class ArrayError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class InputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def checkArrayType(x):
    '''
    Check to see if the type of input is suitable.  Only operate on one
    or two dimension arrays

    Parameters
    ----------
    x: array like
        which can be either a :class:`numpy.ndarray` or list or tuple

    Returns
    -------
    x: :class:`numpy.ndarray`
        checked and converted array
    '''

    if isinstance(x, numpy.ndarray):
        y = x.copy()
    elif isinstance(x, (list, tuple)):
        if isNumeric(x[0]):
            y = numpy.array(x)
        elif isinstance(x[0], (list, tuple, numpy.ndarray)):
            if isNumeric(x[0][0]):
                y = numpy.array(x)
            else:
                raise ArrayError("Expecting elements of float or int")
        else:
            raise ArrayError("Expecting elements of float or int")
    elif isNumeric(x):
        y = numpy.array([x])
    else:
        raise ArrayError("Expecting an array like object")

    return y

def checkDimension(x, y):
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

    y = checkArrayType(y)
    x = checkArrayType(x)

    if len(y) != len(x):
        raise InputError("The number of observations and time points should have the same length")

    return (x, y)

def isNumeric(x):
    '''
    Test whether the input is a numeric

    Parameters
    ----------
    x: 
        anything

    Returns
    -------
    bool:
        True if it belongs to one of the recognized data type from
        the list (int, numpy.int16, numpy.int32, numpy.int64,
        float, numpy.float16, numpy.float32, numpy.float64)
    '''
    return isinstance(x,
                      (int, numpy.int16, numpy.int32, numpy.int64,
                      float, numpy.float16, numpy.float32, numpy.float64))

    
def _checkFunction2DArray(func,x):
    try:
        fx = func(x)
    except Exception:
        fx = numpy.nan
    
    if numpy.any(numpy.isnan(fx)) or numpy.any(numpy.isinf(fx)):
        funcOrig = func
        def aFunc():
            def bFunc(y):
                return funcOrig(y.ravel())
            return bFunc
        func = aFunc()
    
    return func