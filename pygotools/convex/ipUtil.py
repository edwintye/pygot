import numpy
import scipy.linalg, scipy.sparse

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
    primal = func(x)
    
    dual = primal + 0
    if A is not None:
        dual -= y.T.dot(A.dot(x) - b)[0]
    if G is not None:
        dual -= z.T.dot(G.dot(x) - h)[0]

    return primal - dual

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

def _deltaZFunc(x, deltaX, t, z, G, h):
    s = h - G.dot(x)
    rC = _rCentFunc(z, s, t)
    return z/s * G.dot(deltaX) - rC/s


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
    deltaX = None
    deltaZ = None
    deltaY = None

    RHS = _rDualFunc(x, grad, z, G, y, A)
    
    if G is not None:
        s = h - G.dot(x)
        # now find the matrix/vector of our qp
        rCent = _rCentFunc(z, s, t)
        RHS = numpy.append(RHS, rCent, axis=0)

    ## solving the QP to get the descent direction
    if A is not None:
        rPri = _rPriFunc(x, A, b)
        RHS = numpy.append(RHS, rPri, axis=0)

        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T, A.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0), None],
                    [A, None, None]
                    ],'csc')
        else: # G is None
            LHS = scipy.sparse.bmat([
                    [Haug, A.T],
                    [A, None]
                    ],'csc')
    else: # A is None
        if G is not None:
            LHS = scipy.sparse.bmat([
                    [Haug, G.T],
                    [G*-z, scipy.sparse.diags(s.ravel(),0)],
                    ],'csc')
        else:
            LHS = Haug

    deltaTemp = _solveSparseAndRefine(LHS, -RHS)

    deltaX = deltaTemp[:p]
    if A is not None:
        if G is not None:
            deltaZ = deltaTemp[p:-len(A)]
            deltaY = deltaTemp[-len(A):]
        else:
            deltaY = deltaTemp[p::]
    else:
        if G is not None:
            deltaZ = deltaTemp[p::]

    # store the information for the next iteration
    oldFx = fx
    oldGrad = gOrig.copy()

    step, fx = _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx)

    x, y, z = _updateVar(x, deltaX, z, deltaZ, y, deltaY, step)
    # if z is not None:
    #     z += step * deltaZ
    # if y is not None:
    #     y += step * deltaY
        
    # x += step * deltaX

    return x, y, z, fx, step, oldFx, oldGrad, deltaX

def _findStepSize(x, deltaX, z, deltaZ, G, h, y, deltaY, A, b, func, grad, t, g, fx, oldFx):

    if G is None and A is None:
        step, fc, gc, fx, oldFx, new_slope = scipy.optimize.line_search(func,
                                                                        grad,
                                                                        x.ravel(),
                                                                        deltaX.ravel(),
                                                                        g.ravel()
                                                                        )
        if step is None:
            lineFunc = lineSearch(x, deltaX, func)
            step, fx =  backTrackingLineSearch(step0, lineFunc, None, alpha=0.0001, beta=0.8)
    else:
        if G is not None:
            maxStep = _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)
        else:
            maxStep = 1.0
        lineFunc = _residualLineSearchPDC(x, deltaX,
                                          grad, t,
                                          z, deltaZ, G, h,
                                          y, deltaY, A, b)
        # searchScale = -lineFunc(0.0)
        searchScale = None
        # perform a line search.  Because the minimization routine
        # in scipy can sometimes be a bit weird, we assume that the
        # exact line search can sometimes fail, so we do a
        # back tracking line search if that is the case
        step, fx = exactLineSearch2(maxStep, lineFunc, searchScale, oldFx)
    
    return step, fx

def _solveSparseAndRefine(A,b):
    sparseSolver = False
    if scipy.sparse.issparse(A):
        if A.size<=(A.shape[0] * A.shape[1])/2:
            sparseSolver = True
        else:
            A = A.todense()
            sparseSolver = False
    else:
        sparseSolver = False

#     print A
#     print type(A)
#     print "is sparse = "+str(sparseSolver)
    
    i = 0
    if sparseSolver:
        solve = scipy.sparse.linalg.factorized(A)
        x = solve(b).reshape(len(b), 1)
        r = b - A.dot(x)
        # while scipy.linalg.norm(r)>=EPSILON:
        while numpy.any(r/x>=EPSILON):
            d = solve(r).reshape(len(b), 1)
            x += d
            r = b - A.dot(x)
            i += 1
            if i>5:
                break
            elif scipy.linalg.norm(r)<=EPSILON:
                break

        # if scipy.linalg.norm(r)>=EPSILON:
        #     d = solve(r).reshape(len(b), 1)
        #     return x + d
        # else:
        #     return x + d
    else:
        # return scipy.linalg.solve(A,b).reshape(len(b), 1)
        lu, piv = scipy.linalg.lu_factor(A)
        x = scipy.linalg.lu_solve((lu, piv), b).reshape(len(b), 1)
        # for i in range(10):
        r = b - A.dot(x)
        # while scipy.linalg.norm(r)>=EPSILON:
        while numpy.any(r/x>=EPSILON):
            # print scipy.linalg.norm(r)
            # print numpy.linalg.cond(A)
            d = scipy.linalg.lu_solve((lu, piv), r).reshape(len(b), 1)
            x += d
            r = b - A.dot(x)
            i += 1
            if i>5:
                break
            elif scipy.linalg.norm(r)<=EPSILON:
                break

        # if scipy.linalg.norm(r)>=EPSILON:
        #     d = scipy.linalg.lu_solve((lu, piv), r).reshape(len(b), 1)
        #     return x + d
        # else:
        #     return x

            # print scipy.linalg.norm(r)
        # r = b - A.dot(x)
        # d = scipy.linalg.lu_solve((lu, piv), r).reshape(len(b), 1)
    return x

def _updateVar(x, deltaX, z, deltaZ, y, deltaY, step):
    # found one iteration, now update the information
    if z is not None:
        z += step * deltaZ
    if y is not None:
        y += step * deltaY

    x += step * deltaX

    return x, y, z

def _updateVarSimultaneous(x, delta, G, A):
    return _extractLagrangianElement(x+delta, G, A)

def _extractLagrangianElement(b, G, A):
    x = None
    z = None
    y = None

    if A is not None:
        m = len(A)
    if G is not None:
        l = len(G)

    if A is not None:
        if G is not None:
            x = b[:-(m+l)]
            z = b[-(m+l):-m]
            y = b[-m::]
        else:
            x = b[:-m]
            y = b[-m::]
    else:
        if G is not None:
            x = b[:-l]
            y = b[-l::]
        else:
            x = b

    return x, z, y
    

def _maxStepSizePD(z, x, deltaX, t, G, h):
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    return _maxStepSizePDC(z, deltaZ, x, deltaX, G, h)

def _residualLineSearchPD(x, deltaX,
                       gradFunc, t,
                       z, deltaZFunc, G, h,
                       y, deltaY, A, b):
    
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    return _residualLineSearchPDC(x, deltaX,
                                  gradFunc, t,
                                  z, deltaZ, G, h,
                                  y, deltaY, A, b)


def _maxStepSizePDC(z, deltaZ, x, deltaX, G, h):
    index = deltaZ<0
    # step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    
    if any(index==True):
        step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    else:
        step = 1.0

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
