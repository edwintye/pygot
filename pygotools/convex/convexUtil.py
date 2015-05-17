import numpy

from pygotools.optutils.consMani import addLBUBToInequality

def _setup(lb=None, ub=None,
        G=None, h=None,
        A=None, b=None):
    
    if lb is not None or ub is not None:
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

def _logBarrier(x,func,t,G,h):
    def F(x):
        if G is not None:
            s = h - G .dot(x)
            if numpy.any(s<=0):
                return numpy.inf
            else:
                return t * func(x) - numpy.log(s).sum()
        else:
            return t * func(x)
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


