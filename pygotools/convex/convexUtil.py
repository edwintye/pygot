import numpy

from pygotools.optutils.consMani import addLBUBToInequality, feasiblePoint, feasibleStartingValue
from pygotools.optutils.checkUtil import checkArrayType, _checkFunction2DArray
from .approxH import *
from pygotools.gradient.finiteDifference import forward

class InitialValueError(Exception):
    '''
    Issues in getting an initial value
    '''
    pass

def _setup(lb=None, ub=None,
        G=None, h=None,
        A=None, b=None):
    
    if lb is not None and ub is not None:
        if lb is None:
            lb = numpy.nan_to_num(numpy.ones(ub.shape) * - numpy.inf)
        if ub is None:
            ub = numpy.nan_to_num(numpy.ones(lb.shape) * numpy.inf)
        G, h = addLBUBToInequality(lb,ub,G,h)

    if G is not None:
        m,p = G.shape
        z = numpy.ones((m,1))
        h = h.reshape(len(h),1)
    else:
        m = 1.0
        z = None

    if A is not None:
        y = numpy.ones((A.shape[0],1))
        b = b.reshape(len(b),1)
    else:
        y = None

    return z, G, h, y, A, b

def _checkInitialValue(x0, G, h, A, b):
    if x0 is None:
        if G is None:
            if A is None:
                raise Exception("Fail to obtain an initial value")
            else:
                x = scipy.linalg.lstsq(A,b)[0]
                if not feasiblePoint(x, G, h):
                    raise Exception("Fail to obtain an initial value")
        else:
            x = feasibleStartingValue(G, h)
    else:
        x = checkArrayType(x0)
        x = x.reshape(len(x),1)
        
        if G is not None:
            if not feasiblePoint(x, G, h):
                x = feasibleStartingValue(G, h)
        # else (we do not care as we can use infeasible Newton steps 

    return x

def _checkFuncGradHessian(x0, func, grad=None, hessian=None):
    func = _checkFunction2DArray(func, x0)
    p = len(x0)
    
    if grad is None:
        def finiteForward(func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(func,p)
    else:
        grad = _checkFunction2DArray(grad,x0)
        
    if hessian is None:
        approxH = BFGS
    elif type(hessian) is str:
        if hessian.lower()=='bfgs':
            approxH = BFGS
        elif hessian.lower()=='sr1':
            approxH = SR1
        elif hessian.lower()=='dfp':
            approxH = DFP
        else:
            raise Exception("Input name of hessian is not recognizable")
        hessian = None
    else:
        hessian = _checkFunction2DArray(hessian, x0)
        approxH = None
        
    return func, grad, hessian, approxH