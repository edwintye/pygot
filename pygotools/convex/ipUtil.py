import numpy

from pygotools.optutils.consMani import addLBUBToInequality

def _setup(lb=None, ub=None,
        G=None, h=None,
        A=None, b=None):
    
    if lb is not None or ub is not None:
        G, h = addLBUBToInequality(lb,ub,G,h)

    if G is not None:
        m,p = G.shape
        z = numpy.zeros(m)
        h = h.reshape(len(h),1)
    else:
        m = 1.0
        z = None

    if A is not None:
        y = numpy.zeros(A.shape[0])
        b = b.reshape(len(b),1)
    else:
        y = None
    

    return z, G, h, y, A, b

def _logBarrier(x,func,t,G,h):
    def F(x):
        s = h - G .dot(x)
        if numpy.any(s<=0):
            return numpy.inf
        else:
            return t * func(x) - numpy.log(s).sum()
    return F

def _findInitialBarrier(g,y,A):
    if A is None:
        t = float(numpy.linalg.lstsq(g, -y)[0].ravel()[0])
    else:
        X = numpy.append(A,g,axis=1)
        t = float(numpy.linalg.lstsq(X, -y)[0].ravel()[-1])

    #TODO: check
    return abs(t)

def _dualityGap(func, x, z, G, h, y, A, b):
    gap = func(x)
    if A is not None:
        gap += y.dot(A.dot(x) - b)
    if G is not None:
        gap += z.dot(G.dot(x) - h)

    return gap

def _surrogateGap(x, z, G, h, y, A, b):
    s = h - G.dot(x)
    return numpy.inner(s,z)


