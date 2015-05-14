__all__[
        'checkArrayType',
        'checkDimension'
        ]

import numpy

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
        pass
    elif isinstance(x, (list, tuple)):
        if isNumeric(x[0]):
            x = numpy.array(x)
        elif isinstance(x[0], (list, tuple, numpy.ndarray)):
            if isNumeric(x[0][0]):
                x = numpy.array(x)
            else:
                raise ArrayError("Expecting elements of float or int")
        else:
            raise ArrayError("Expecting elements of float or int")
    elif isNumeric(x):
        x = numpy.array([x])
    else:
        raise ArrayError("Expecting an array like object")

    return x

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