import numpy
from cvxopt import matrix
from cvxopt import solvers

from pygotools.optutils.optCondition import lineSearch, backTrackingLineSearch
import scipy.sparse

maxRadius = 1000.0

def _updateLineSearch(x, fx, oldFx, oldOldFx, oldDeltaX, g, H, func, grad, z, G, h, y, A, b):

    _runOptions()

    initVals = dict()
    initVals['x'] = matrix(oldDeltaX)
    # readjust the bounds and initial value if possible
    # as we try our best to use warm start
    if G is not None:
        hTemp = h - G.dot(x)
        dims = {'l': G.shape[0], 'q': [], 's':  []}
        initVals['z'] = matrix(z)
        s = hTemp - G.dot(oldDeltaX)

        while numpy.any(s<=0.0):
            oldDeltaX *= 0.5
            s = h - G.dot(oldDeltaX)
        initVals['s'] = matrix(s)
        initVals['x'] = matrix(oldDeltaX)

        #print initVals['s']
    else:
        hTemp = None
        dims = []

    if A is not None:
        initVals['y'] = matrix(y)
        bTemp = b - A.dot(x)
    else:
        bTemp = None

    # solving the QP to get the descent direction    
    if A is not None:
        if G is not None:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, matrix(A), matrix(bTemp))
        else:
            qpOut = solvers.coneqp(matrix(H), matrix(g), None, None, None, matrix(A), matrix(bTemp))
    else:
        if G is not None:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(G), matrix(hTemp), dims, initvals=initVals)
        else:
            qpOut = solvers.coneqp(matrix(H), matrix(g))

    # exact the descent diretion and do a line search
    deltaX = numpy.array(qpOut['x'])
    oldOldFx = oldFx
    oldFx = fx
    oldGrad = g.copy()

    lineFunc = lineSearch(x, deltaX, func)
    #step, fx = exactLineSearch(1, x, deltaX, func)
    # step, fc, gc, fx, oldFx, new_slope = line_search(func,
    #                                                  grad,
    #                                                  x.ravel(),
    #                                                  deltaX.ravel(),
    #                                                  g.ravel(),
    #                                                  oldFx,
    #                                                  oldOldFx)

    # print step
    # if step is None:
    # step, fx = exactLineSearch2(1, lineFunc, deltaX.ravel().dot(g.ravel()), oldFx)
    # step, fx = exactLineSearch(1, lineFunc)
    # if fx >= oldFx:
    step, fx = backTrackingLineSearch(1, lineFunc, deltaX.ravel().dot(g.ravel()), alpha=0.0001, beta=0.8)

    x += step * deltaX
    if G is not None:
        z[:] = numpy.array(qpOut['z'])
    if A is not None:
        y[:] = numpy.array(qpOut['y'])

    return x, deltaX, z, y, fx, oldFx, oldOldFx, oldGrad, step, qpOut['iterations']


def _updateTrustRegionSOCP(x, fx, oldFx, oldDeltaX, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b):

    _runOptions()

    if A is not None:
        bTemp = b - A.dot(x)
    else:
        bTemp = None

    GTemp = numpy.append(numpy.zeros((1,p)), numpy.eye(p), axis=0)
    hTemp = numpy.zeros(p+1)
    hTemp[0] += radius

    if G is not None:
        GTemp = numpy.append(G, GTemp, axis=0)
        hTemp = numpy.append(h - G.dot(x), hTemp)
        dims1 = {'l': G.shape[0], 'q': [p+1,p+1], 's': []}
    else:
        dims1 = {'l': 0, 'q': [p+1,p+1], 's': []}

    # now we have finished the setup process, we reformulate
    # the problem as a SOCP

    m,n = GTemp.shape
    c = matrix([1.0] + [0.0]*n)

    hTemp1 = matrix([0.0]+(-g.flatten()).tolist())
    GTemp1 = matrix(numpy.array(scipy.sparse.bmat([
        [[-1.0],None],
        [None,H]
    ]).todense()))

    GTemp1 = matrix(numpy.append(numpy.append(numpy.array([0]*m).reshape(m,1),numpy.array(GTemp),axis=1),
                                 numpy.array(GTemp1),
                                 axis=0))

    hTemp1 = matrix(numpy.append(hTemp,hTemp1))
    
    if A is not None:
        out = solvers.conelp(c, GTemp1, hTemp1, dims1, matrix(A), matrix(bTemp))
    else:
        out = solvers.conelp(c, GTemp1, hTemp1, dims1)

    # exact the descent direction and do a line search
    deltaX = numpy.array(out['x'][1::])

    M = _diffM(g.flatten(), H)
    
    newFx = func((x + deltaX).ravel())
    
    predRatio = (fx - newFx) / M(deltaX)
    # print "predRatio = " +str(predRatio)
        
    if predRatio>=0.75:
        radius = min(2.0*radius, maxRadius)
    elif predRatio<=0.25:
        radius *= 0.25
    elif numpy.isnan(predRatio):
        radius *= 0.25

    if predRatio>=0.25:
        oldGrad = g.copy()
        x += deltaX
        oldFx = fx
        fx = newFx
        update = True
    else:
        update = False

    if G is not None:
        # only want the information for the inequalities
        # and not the two cones - trust region, objective functionn
        z[:] = numpy.array(out['z'])[G.shape[0]]
    if A is not None:
        y[:] = numpy.array(out['y'])

    # print numpy.append(numpy.array(out['s']),numpy.array(s),axis=1)

    return x, update, radius, deltaX, z, y, fx, oldFx, oldGrad, out['iterations']


def _updateTrustRegion(x, fx, oldFx, oldDeltaX, p, radius, g, oldGrad, H, func, grad, z, G, h, y, A, b):

    _runOptions()

    # readjust the bounds and initial value if possible
    # as we try our best to use warm start
    GTemp = numpy.append(numpy.zeros((1,p)), numpy.eye(p), axis=0)
    hTemp = numpy.zeros(p+1)
    hTemp[0] += radius

    if G is not None:
        GTemp = numpy.append(G, GTemp, axis=0)
        hTemp = numpy.append(h - G.dot(x), hTemp)
        dims = {'l': G.shape[0], 'q': [p+1], 's':  []}
    else:
        dims = {'l': 0, 'q': [p+1], 's':  []}

    if A is not None:
        bTemp = b - A.dot(x)
    else:
        bTemp = None

    # solving the QP to get the descent direction
    try:
        if A is not None:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims, matrix(A), matrix(bTemp))
            # print qpOut
        else:
            qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims)
    except Exception as e:
        raise e

    # exact the descent diretion and do a line search
    deltaX = numpy.array(qpOut['x'])
    # diffM is the difference between the real objective
    # function and M, the quadratic approximation 
    # M = diffM(deltaX.flatten(), g.flatten(), H)
    M = _diffM(g.flatten(), H)
    
    newFx = func((x + deltaX).ravel())
    predRatio = (fx - newFx) / M(deltaX)
        
    if predRatio>=0.75:
        radius = min(2.0*radius, maxRadius)
    elif predRatio<=0.25:
        radius *= 0.25
    elif numpy.isnan(predRatio):
        radius *= 0.25

    if predRatio>=0.25:
        oldGrad = g.copy()
        x += deltaX
        oldFx = fx
        fx = newFx
        update = True
    else:
        update = False

    if G is not None:
        z[:] = numpy.array(qpOut['z'])[G.shape[0]]
    if A is not None:
        y[:] = numpy.array(qpOut['y'])

    return x, update, radius, deltaX, z, y, fx, oldFx, oldGrad, qpOut['iterations']


# def diffM(deltaX, g, H):
def _diffM(g, H):
    def M(deltaX):
        m = -g.dot(deltaX) - 0.5 * deltaX.T.dot(H).dot(deltaX)
        return m
    return M

def _runOptions(progress=False,atol=1e-6,reltol=1e-6):
    solvers.options['show_progress'] = progress
    solvers.options['atol'] = atol
    solvers.options['reltol'] = reltol
