
__all__ = [
    'ipD'
    ]

from pygotools.optutils.optCondition import lineSearch, exactLineSearch, backTrackingLineSearch, sufficientNewtonDecrement

from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian
from .approxH import *
from .convexUtil import _setup, _logBarrier, _findInitialBarrier, _surrogateGap

import numpy
#import numpy.linalg

import scipy.linalg, scipy.sparse

#from cvxopt.solvers import coneqp
from cvxopt import solvers
from cvxopt import matrix, mul, div, spdiag
from cvxopt import blas

solvers.options['show_progress'] = True

EPSILON = 1e-6
maxiter = 100

def ipD(func, grad, hessian=None, x0=None,
        lb=None, ub=None,
        G=None, h=None,
        A=None, b=None,
        maxiter=100,
        disp=0, full_output=False):

    x = checkArrayType(x0)
    p = len(x)
    x = x.reshape(p,1)
    g = numpy.zeros((p,1))
    gOrig = g.copy()

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)

    oldFx = numpy.inf
    oldGrad = None
    deltaX = None
    deltaY = 0
    deltaZ = 0
    H = numpy.zeros((p,p))
    Haug = H.copy()

    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    mu = 5.0
    step = 1.0
    t = 1.0
    # because we determine the size of the back tracking
    # step on the fly, we don't give it a maximum.  At the
    # same time, because we are only evaluating the residuals
    # of the KKT system, there are times where we may want to
    # give the descent a nudge
    #step0 = 1.0  # back tracking search step maximum value
    noImprovement = 0

    if G is not None:
        s = h - G.dot(x)
        z = 1.0/s
        #z = numpy.ones(s.shape)
        # print G.dot(x)
        # print s
        # print z
        m = G.shape[0]

    if hessian is None:
        approxH = BFGS

    if G is not None:
        eta = _surrogateGap(x, z, G, h, y, A, b)
        # print eta
        t = mu * m / eta
        # print t

    while 1>=EPSILON:

        gOrig[:] = grad(x).reshape(p,1)
        g[:] = gOrig.copy()
        
        if hessian is None:
            if oldGrad is None:
                H = numpy.eye(len(x))
            else:
                diffG = (gOrig - oldGrad).ravel()
                # print "update"
                # print "gOrig"
                # print gOrig
                # print "old grad"
                # print oldGrad
                # print "diff g"
                # print diffG
                # print "delta x"
                # print deltaX
                H = approxH(H, diffG, step * deltaX.ravel())
            #H = forwardGradCallHessian(grad,x.ravel())
        else:
            H = hessian(x)

        Haug[:] = H.copy()

        ## standard log barrier, \nabla f(x) / -f(x)
        # print h
        # print G.dot(x)
        if G is not None:
            s = h - G.dot(x)
            Gs = G/s
            zs = z/s
            # print "s"
            # print s
            # print "z"
            # print z
            # print "zs"
            # print zs
            # print "aug part"
            # print numpy.einsum('ji,ik->jk',G.T, G*zs)

            # now find the matrix/vector of our qp
            Haug += numpy.einsum('ji,ik->jk',G.T, G*zs)

        # print "H"
        # print H
        # print numpy.linalg.eig(H)[0]
        # print "Haug"
        # print Haug
        # print numpy.linalg.eig(Haug)[0]
            Dphi = Gs.sum(axis=0).reshape(p,1)
            g += Dphi / t

        # readjust the bounds
        # if A is not None:

        #     # print g
        #     # print A.T.dot(y)

        # else:
        #     bTemp = None 

        #print "got here"

        ## solving the QP to get the descent direction
        if A is not None:
            bTemp = b - A.dot(x)
            g += A.T.dot(y)
            #print "here"
            LHS = scipy.sparse.bmat([[Haug,A.T],[A,None]])
            RHS = numpy.append(g,-bTemp,axis=0)
            # print LHS
            # print RHS
            deltaTemp = scipy.sparse.linalg.spsolve(LHS,-RHS).reshape(len(RHS),1)
            deltaX = deltaTemp[:p]
            deltaY = deltaTemp[p::]
            # print LHS.todense()
            # print scipy.linalg.solve(LHS.todense(),-RHS)
            # print "y"
            # print y
            # print "sparse"
            # print deltaX
            # print deltaY
            # print "diff"
            # print LHS.dot(deltaTemp) + RHS
            # qpOut = solvers.coneqp(matrix(Haug), matrix(g),
            #                        None, None, None,
            #                        matrix(A), matrix(bTemp))
            # print "scipy"
            # print deltaXS
            # print "cone"
            # print qpOut['x']
            # print qpOut['y']
            # deltaTemp[:p] = qpOut['x']
            # deltaTemp[p::] = qpOut['y']
            # print LHS.dot(deltaTemp) + RHS

            # deltaX = numpy.array(qpOut['x'])
            # deltaY = numpy.array(qpOut['y'])
        else:
            deltaX = scipy.linalg.solve(Haug,-g).reshape(p,1)
            # qpOut = solvers.coneqp(matrix(Haug), matrix(g))
            # print "scipy"
            # print deltaX
            # print "cone"
            # print qpOut['x']

            # try:
            #     qpOut = solvers.coneqp(matrix(Haug), matrix(g))
            # except Exception as e:
            #     # print "H"
            #     # print H
            #     # print "eigen"
            #     # print numpy.linalg.eig(numpy.array(H))[0]
            #     raise e

        #print "got out"
        ## exact the descent diretion and do a line search
        #print qpOut
        #deltaX = numpy.array(qpOut['x'])

        # store the information for the next iteration
        oldFx = fx
        oldGrad = gOrig.copy()

        if G is None:
            # print "obj"
            maxStep = 1
            barrierFunc = _logBarrier(x, func, t, G, h)
            lineFunc = lineSearch(maxStep, x, deltaX, barrierFunc)
        
            step, fx = backTrackingLineSearch(maxStep,
                                              lineFunc,
                                              deltaX.ravel().dot(g.ravel()))

        else:
            #print "resid"
            maxStep = _maxStepSize(z, x, deltaX, t, G, h)
            residualFunc = residualLineSearch(maxStep,
                                              x, deltaX,
                                              grad, t,
                                              z, _deltaZFunc, G, h,
                                              y, deltaY, A, b,
                                              _rDualFunc, _rPriFunc)

            #step, fx =  backTrackingLineSearch(maxStep, residualFunc, -residualFunc(0.0))
            step, fx =  backTrackingLineSearch(maxStep, residualFunc, -residualFunc(0.0))

        if z is not None:
            z += step * _deltaZFunc(x, deltaX, t, z, G, h)
        if y is not None:
            # deltaY = numpy.array(qpOut['y'])
            y += step * deltaY

        x += step * deltaX
        i += 1
        # now we can update s
        #s = h - G.dot(x)
        # print "finish iteration"
        # print "step = " +str(step)
        # print "max step = " +str(maxStep)
        # print "gOrig"
        # print gOrig
        # print "deltaX"
        # print deltaX

        # print "s"
        # print s
        # print "z"
        # print z

        #print qpOut
        dispObj.d(i, x , fx, deltaX.ravel(), g.ravel(), step)

        feasible = False
        if G is not None:
            feasible = True
            eta = _surrogateGap(x, z, G, h, y, A, b)
            if eta >= EPSILON:
                feasible = False
            if G is not None:
                r = _rDualFunc(x, grad, z, G, y, A)
                if scipy.linalg.norm(r) >= EPSILON:
                    feasible = False
            if A is not None:
                r = _rPriFunc(x, A, b)
                if scipy.linalg.norm(r) >= EPSILON:
                    feasible = False

            t = mu * m / eta

        if feasible:
            print "feasible"
            break
        else:
            if G is None:
                if abs(fx-oldFx)<=EPSILON:
                    break
        #print t

        if i >= maxiter:
            break

        # print "\naugmented obj: "+ str(barrierFunc(x))
        # print "iteration = " +str(i)
        # print "obj: "+str(func(x))
        # print "t = "+str(t)
        # print "gap = " +str(eta)
        # print "step = " +str(step)
        # print "max step =" +str(_maxStepSize(z, matrix(x), matrix(deltaX), t, G, h))


        # print "full step"
        # print residualFunc(1.0)
        # print "no step"
        # print residualFunc(0.0)
        # print "max step"
        # print residualFunc(maxStep)
        # print "dual residual"
        # print blas.nrm2(_rDual(x, grad, z, G, y, A))
        # print "obj with full step"
        # print func(x+maxStep * deltaX)
                
        # print "z : "
        # print z

        # residualFunc = residualLineSearch(step, matrix(x), matrix(deltaX), g, t,
        #                                   z, _deltaZ, G, h,
        #                                   y, deltaY, A, b,
        #                                   _rDual,_rPri)
        # print "residual"
        # print residualFunc(0)
        # print "residual +"
        # print numpy.linalg.norm(residualFunc(step))
        # print "residual backtrack"
        # print _backTrackingLineSearch(1, residualFunc, -residualFunc(0))

        # print s
        # print x


    # TODO: full_output- dual variables
    if full_output:
        output = dict()
        output['t'] = t

        if G is not None:
            gap = _surrogateGap(x, z, G, h, y, A, b)
        else:
            gap = 0

        output['dgap'] = gap
        output['fx'] = func(x)
        output['H'] = H
        output['g'] = gOrig.ravel()

        if G is not None:
            output['s'] = s.ravel()
            output['z'] = z.ravel()
            output['rDual'] = _rDualFunc(x, grad, z, G, y, A).ravel()
        if A is not None:
            output['rPri'] = _rPriFunc(x, A, b).ravel()
            output['y'] = y.ravel()

        return x, output
    else:
        return x


def _deltaZFunc(x, deltaX, t, z, G, h):
    s = h - G.dot(x)
    rC = _rCentFunc(z, s, t)
    return z/s * G.dot(deltaX) + rC/-s
    
def _rDualFunc(x, gradFunc, z, G, y, A):
    g = gradFunc(x)
    if G is not None:
        g += G.T.dot(z)
    if A is not None:
        g += A.T.dot(y)
    return g

def _rCentFunc(z, s, t):
    return z*s - (1.0/t)

def _rPriFunc(x, A, b):
    return A.dot(x) - b

def _maxStepSize(z, x, deltaX, t, G, h):
    deltaZ = _deltaZFunc(x, deltaX, t, z, G, h)
    index = deltaZ<0
    
    # print "z"
    # print z
    # print "deltaz"
    # print deltaZ
    # print "index"
    # print index
    
    step = 0.99 * min(1,min(-z[index]/deltaZ[index]))
    #zTemp = z + step * deltaZ
    # print "max step size"
    # print s
    # print "zTemp"
    
    s = h - G.dot(x + step * deltaX)
    while numpy.any(s<=0):
        step *= 0.5
        s = h - G.dot(x + step * deltaX)

    # print "max z"
    # print z + step * deltaZ
    # print "max s"
    # print h - G.dot(x + step * deltaX)
    # print "deltaZ"
    # print deltaZ

    return step

def residualLineSearch(step, x, deltaX,
                       gradFunc, t,
                       z, deltaZFunc, G, h,
                       y, deltaY, A, b,
                       dualFunc,priFunc):
    
    def F(step):
        newX = x + step * deltaX
        if z is not None:
            newZ = z + step * deltaZFunc(x, deltaX, t, z, G, h)
            r1 = dualFunc(newX, gradFunc, newZ, G, y, A).ravel()
        else:
            r1 = numpy.zeros(1)

        if y is not None:
            #newY = y + step * deltaY
            r2 = numpy.array(priFunc(x, A, b)).ravel()
        else:
            r2 = numpy.zeros(1)

        r = numpy.append(r1,r2,axis=0)
        # print "full vector"
        # print r
        return scipy.linalg.norm(r)
    return F


# def _backTrackingLineSearch(step, f, scale, alpha=0.1, beta=0.5):
    
#     maxStep = step

#     fx = f(0.0)
#     fdeltaX = f(step)

#     i = 0

#     # print "residual line search"
#     #print (fdeltaX,fx + alpha * step * scale, step)
#     while fdeltaX > fx + alpha * step * scale:
#         # print "step="+str(step)
#         # print (fdeltaX,fx + alpha * step * scale, step)
#         step *= beta
#         fdeltaX = f(step)
#         # if step <= 1e-16:
#         #     if f(maxStep) < fx + alpha * step * scale:
#         #         return maxStep, fdeltaX
#         #     else:
#         #         return 1e-16, fdeltaX
#         # else:
#         #     i += 1

#     # print "finish searching"
#     return step, fdeltaX
