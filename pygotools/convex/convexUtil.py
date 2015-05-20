import numpy

from pygotools.optutils.consMani import addLBUBToInequality, feasiblePoint, feasibleStartingValue
from pygotools.optutils.checkUtil import checkArrayType
import scipy.linalg

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
        z = numpy.zeros((m,1))
        h = h.reshape(len(h),1)
    else:
        m = 1.0
        z = None

    if A is not None:
        y = numpy.zeros((A.shape[0],1))
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

def _logBarrier(x, func, t, G, h):
    def F(x):
        p = len(x)
        x = x.reshape(p,1)
        if G is not None:
            s = h - G .dot(x)
            #print "s"
            #print s
            if numpy.any(s<=0):
                return numpy.nan_to_num(numpy.inf)
            else:
                return t * func(x) - numpy.log(s).sum()
        else:
            return t * func(x)
    return F

def _logBarrierGrad(x, func, gOrig, t, G, h):
    def F(x):
        p = len(x)
        x = x.reshape(p,1)
        
        if G is not None:
            s = h - G.dot(x)
            Gs = G/s
            Dphi = Gs.sum(axis=0).reshape(p,1)
            g = t * gOrig + Dphi
        else:
            g = t * gOrig
        return g.ravel()
    return F

def _findInitialBarrier(g,y,A):
    if A is None:
        t = float(numpy.linalg.lstsq(g, -y)[0].ravel()[0])
    else:
        # print A
        # print g
        X = numpy.append(A.T,g,axis=1)
        # print X
        t = float(numpy.linalg.lstsq(X, -y)[0].ravel()[-1])
        #print X
        # print numpy.linalg.lstsq(X, -y)
        # print t
        # print type(t)

    #TODO: check
    return abs(t)

def _dualityGap(func, x, z, G, h, y, A, b):
    gap = func(x)
    if A is not None:
        gap += y.T.dot(A.dot(x) - b)[0]
    if G is not None:
        gap += z.T.dot(G.dot(x) - h)[0]

    return gap

def _surrogateGap(x, z, G, h, y, A, b):
    s = h - G.dot(x)
    return numpy.inner(s.ravel(),z.ravel())

    
def _rDualFunc(x, gradFunc, z, G, y, A):
    g = gradFunc(x)
    g = g.reshape(len(g),1)
    if G is not None:
        g += G.T.dot(z)
    if A is not None:
        g += A.T.dot(y)
    return g

def _rCentFunc(z, s, t):
    return z*s - (1.0/t)

def _rPriFunc(x, A, b):
    return A.dot(x) - b         
        
        