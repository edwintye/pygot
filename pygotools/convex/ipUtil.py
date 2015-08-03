import numpy
import scipy.linalg

from pygotools.optutils.optCondition import lineSearch, exactLineSearch2, exactLineSearch, backTrackingLineSearch, sufficientNewtonDecrement

EPSILON = 1e-6

def _logBarrier(func, t, G, h):
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
                return t*func(x) - numpy.log(s).sum()
        else:
            return func(x)
    return F

def _logBarrierGrad(func, gOrig, t, G, h):
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

def _rCentFunc(z, s, t=None):
    if t==None:
        return z*s
    else:
        return z*s - (1.0/t)

def _rCentFunc2(x, z, G, h, t):
    s = h - G.dot(x)
    return _rCentFunc(z, s, t)

def _rCentFuncCorrect(z, s, deltaZ, deltaS, t=None):
    # print deltaZ * deltaS
    if t==None:
        return z*s + deltaZ * deltaS
    else:
        return z*s + deltaZ * deltaS - (1.0/t)

def _rPriFunc(x, A, b):
    return A.dot(x) - b         

def _checkDualFeasible(x, z, G, h, y, A, b, gradFunc, m, mu):

    if G is None:
        raise Exception("No Linear Inequality")
    
    feasible = True
    eta = _surrogateGap(x, z, G, h, y, A, b)
    if eta >= EPSILON:
        feasible = False
    
    r = _rDualFunc(x, gradFunc, z, G, y, A)
    if scipy.linalg.norm(r) >= EPSILON:
        feasible = False

    if A is not None:
        r = _rPriFunc(x, A, b)
        if scipy.linalg.norm(r) >= EPSILON:
            feasible = False

    t = mu * m / eta
    
    return feasible, t    

def _solveKKTAndUpdatePDC(x, func, grad, fx, g, gOrig, Haug, z, G, h, y, A, b, t):
    p = len(x)
    step = 1
    deltaX = None
    deltaZ = None
    deltaY = None

    rDual = _rDualFunc(x, grad, z, G, y, A)
    RHS = rDual
    if G is not None:
        s = h - G.dot(x)
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        g += A.T.dot(y)

        rPri = _rPriFunc(x, A, b)
        RHS = numpy.append(RHS, rPri, axis=0)

        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T, A.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0), None],
                    [A, None, None]
                    ],'csc')
            if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            else:
                deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS), 1)
                deltaX = deltaTemp[:p]
                deltaZ = deltaTemp[p:-len(A)]
                deltaY = deltaTemp[-len(A):]
        else: # G is None
            LHS = scipy.sparse.bmat([
                    [Haug, A.T],
                    [A, None]
                    ],'csc')
            if LHS.size>= (LHS.shape[0] * LHS.shape[1])/2:
                deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            else:
                deltaTemp = scipy.linalg.solve(LHS.todense(), -RHS).reshape(len(RHS),1)
                deltaX = deltaTemp[:p]
                deltaY = deltaTemp[p::]
    else: # A is None
        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0)],
                    ],'csc')
            deltaTemp = scipy.sparse.linalg.spsolve(LHS, -RHS).reshape(len(RHS), 1)
            deltaX = deltaTemp[:p]
            deltaZ = deltaTemp[p::]
        else:
            deltaX = scipy.linalg.solve(Haug, -RHS).reshape(len(RHS), 1)

    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()

    if G is None:
        # print "obj"
        maxStep = 1
        barrierFunc = _logBarrier(func, t, G, h)
        lineFunc = lineSearch(x, deltaX, barrierFunc)
        searchScale = deltaX.ravel().dot(g.ravel())
    else:
        maxStep = _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)
        lineFunc = _residualLineSearchPDC(x, deltaX,
                                       grad, t,
                                       z, deltaZ, G, h,
                                       y, deltaY, A, b)
        searchScale = -lineFunc(0.0)

    # perform a line search.  Because the minimization routine
    # in scipy can sometimes be a bit weird, we assume that the
    # exact line search can sometimes fail, so we do a
    # back tracking line search if that is the case
    step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)
    # step, fx =  exactLineSearch(maxStep, lineFunc)
    # if fx >= oldFx or step <= 0 or step>=maxStep:
    #     step, fx =  backTrackingLineSearch(maxStep, lineFunc, searchScale, oldFx)

    if z is not None:
        z += step * deltaZ
    if y is not None:
        y += step * deltaY
        
    x += step * deltaX

    return x, y, z, fx, step, oldFx, oldGrad, deltaX

def _maxStepSizePDC(z, deltaZ, x, deltaX, G, h):
    index = deltaZ<0
    step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    
    s = h - G.dot(x + step * deltaX)
    while numpy.any(s<=0):
        step *= 0.8
        s = h - G.dot(x + step * deltaX)

    return step

def _residualLineSearchPDC(x, deltaX,
                       gradFunc, t,
                       z, deltaZ, G, h,
                       y, deltaY, A, b):
    
    def F(step):
        newX = x + step * deltaX
        if G is not None:
            newZ = z + step * deltaZ
        else:
            newZ = z

        if A is not None:
            newY = y + step * deltaY
        else:
            newY = y

        r1 = _rDualFunc(newX, gradFunc, newZ, G, newY, A)
        if G is not None:
            s = h - G.dot(newX)
            r2 = _rCentFunc(newZ, s, t)
            r1 = numpy.append(r1,r2,axis=0)

        if A is not None:
            r2 = _rPriFunc(newX, A, b)
            r1 = numpy.append(r1,r2,axis=0)

        return scipy.linalg.norm(r1)
    return F

