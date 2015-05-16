
__all__ = [
    'ipD'
    ]

from pygotools.optutils.optCondition import exactLineSearch, backTrackingLineSearch, sufficientNewtonDecrement

from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forwardGradCallHessian
from .approxH import *
from .ipUtil import _setup, _logBarrier, _findInitialBarrier, _surrogateGap

import numpy
import numpy.linalg

from cvxopt.solvers import coneqp
from cvxopt import matrix, mul, div, spdiag
from cvxopt import blas

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
    g = matrix(0.0,(p,1))

    z, G, h, y, A, b = _setup(lb, ub, G, h, A, b)

    oldFx = numpy.inf
    oldGrad = None
    deltaX = None
    deltaY = 0
    deltaZ = 0

    fx = func(x)

    dispObj = Disp(disp)
    i = 0
    t = 0.01
    mu = 5.0
    # because we determine the size of the back tracking
    # step on the fly, we don't give it a maximum.  At the
    # same time, because we are only evaluating the residuals
    # of the KKT system, there are times where we may want to
    # give the descent a nudge
    #step0 = 1.0  # back tracking search step maximum value
    noImprovement = 0

    if G is not None:
        s = h - G * matrix(x)
        z = div(1.0, s)
        m = G.size[0]

    if hessian is None:
        approxH = BFGS

    while 1>=EPSILON:

        gOrig = matrix(grad(x))
        blas.copy(gOrig,g)
        
        if hessian is None:
            if oldGrad is None:
                H = numpy.eye(len(x))
            else:
                # print "update hessian"
                # print "g"
                # print g
                # print "old g"
                # print oldGrad
                diffG = numpy.array(g - oldGrad).ravel()
                H = approxH(numpy.array(H), diffG, deltaX)
        else:
            H = hessian(x)

        # readjust the bounds
        if A is not None:
            bTemp = b - A * matrix(x)
        else:
            bTemp = None

        ## standard log barrier, \nabla f(x) / -f(x)
        Gs = div(G,s[:,matrix(0,(1,p))])
        zs = div(z,s)

        H = matrix(H) 
        H += matrix(numpy.einsum('ji,ik->jk',G.T, div(G,zs[:,matrix(0,(1,p))])))

        # first we have an empty vector, then we inplace replace
        # via blas routine
        Dphi = matrix(0.0,(p,1))
        blas.gemv(Gs, matrix(1.0,(m,1)),Dphi,'T')
        # print "g"
        # print g
        # print "Dphi"
        # print Dphi

        if i==0:
            t = _findInitialBarrier(g,Dphi,A)
            # print "First iteration"
            # print float(numpy.linalg.lstsq(g, -y)[0])
            # t = float(numpy.linalg.lstsq(g, -y)[0][0][0])

        # print z
        # print y
        # print Dphi

        g += Dphi / t
        if y is not None:
            g += A.T * y
        # print type(g)

        ## solving the QP to get the descent direction
        if A is not None:
            qpOut = coneqp(H, g, [], [], [], A, bTemp)
        else:
            try:
                qpOut = coneqp(H, g)
            except Exception as e:
                # print "H"
                # print H
                # print "eigen"
                # print numpy.linalg.eig(numpy.array(H))[0]
                raise e

        ## exact the descent diretion and do a line search
        # print qpOut
        deltaX = numpy.array(qpOut['x']).ravel()
        barrierFunc = _logBarrier(x, func, t, G, h)

        # store the information for the next iteration
        oldFx = fx
        oldGrad = gOrig

        maxStep = _maxStepSize(z, matrix(x), matrix(deltaX), t, G, h)
        if noImprovement>=5:
            # print "using obj line search"
        #step, fx = exactLineSearch(step0, x, deltaX, barrierFunc)
            #step, fx = backTrackingLineSearch(maxStep, x, deltaX, barrierFunc, grad(x))
            step, fx = backTrackingLineSearch(maxStep, x, deltaX, func, grad(x))

        # print "step"
        # print step
        # print "fx"
        # print fx
        # print "deltaX"
        # print deltaX

        else:
            residualFunc = residualLineSearch(maxStep,
                                              matrix(x), matrix(deltaX),
                                              grad, t,
                                              z, _deltaZ, G, h,
                                              y, deltaY, A, b,
                                              _rDual,_rPri)

        # print " after max step"
        # print z +maxStep * _deltaZ(matrix(x), matrix(deltaX), t, z, G, h)


        # print "full step"
        # print residualFunc(1.0)
        # print "no step"
        # print residualFunc(0.0)
        # print "max step"
        #print maxStep
        #print type(maxStep)
        # print residualFunc(maxStep)

            step, fx =  _backTrackingLineSearch(maxStep, residualFunc, -residualFunc(0.0))

        # print "z"
        # print z
        # print "step"
        # print step
        # print "deltaZ"
        # print  _deltaZ(matrix(x), matrix(deltaX), t, z, G, h)

        z += step * _deltaZ(matrix(x), matrix(deltaX), t, z, G, h)
        # print "new z "
        # print z

        x += step * deltaX
        # print "x"
        # print x
        # print "x size"
        # print x.size
        i += 1
        
        # now we can update s
        s = h - G * matrix(x)

        if y is not None:
            deltaY = numpy.array(qpOut['y']).ravel()
            y += step * qpOut['y']
        
        #print rDual(g, z, G, y, A)
        #print z
        #print s
        #print rCent(z, s, t)

        # rC = _rCent(z, s, t)
        # print "t"
        # print t
        # print "rCent"
        # print rC
        # print "z"
        # print z

        # print "2nd"
        # print div(rC,-s)
        # print "1st"
        # print spdiag(div(z,-s)) * (G * matrix(deltaX))
        # print "1st alternative"
        # print mul(div(z,-s),G * matrix(deltaX))

        # print numpy.append(mul(div(z,-s),G * matrix(deltaX)),div(rC,-s),axis=1)

        # print "\ndelta z"
        # print div(rC,-s) + spdiag(div(z,-s)) * (G * matrix(deltaX))
        # print "new z"
        # print z + div(rC,-s) + spdiag(div(z,-s)) * (G * matrix(deltaX))
        # print "delta z function"
        # print deltaZ(matrix(x), deltaX, t, z, G, h)
        # print type(deltaZ(matrix(x), matrix(deltaX), t, z, G, h))
        # print type(z)
        # print type(step)

        # print h - G * matrix(x)
        # print z.T * (h - G * matrix(x))

        #return div(rC,-s) - div(spdiag(z) * G * deltaX,s)
        
        # print "delta z"
        # print deltaZ(matrix(x), matrix(deltaX), t, z, G, h)

        eta = _surrogateGap(matrix(x), z, G, h, y, A, b)

        # print "eta"
        # print eta
        # print "t"
        # print t

        #print qpOut
        dispObj.d(i, x , func(x), deltaX, numpy.array(g))
        feasible = True

        if eta >= EPSILON:
            feasible = False
        if G is not None:
            r = _rDual(x, grad, z, G, y, A)
            if blas.nrm2(r) >= EPSILON:
                feasible = False
        if A is not None:
            r = _rPri(x, A, b)
            if blas.nrm2(r) >= EPSILON:
                feasible = False

        if feasible:
            print "feasible"
            break
        else:
            if abs(fx-oldFx)<=EPSILON:
                noImprovement += 1
                if noImprovement >= 10:
                    H = matrix(numpy.eye(p))
            else:
                noImprovement = 0
            t = mu * m / eta
        
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

        # gap = dualityGap(func, x,
        #                  z, G, h,
        #                  y, A, b)

        gap = _surrogateGap(matrix(x), z, G, h, y, A, b)
        # print z.size
        # print type(G)
        # print type(h)
        # print type(G*matrix(x) - h)
        # print type(z)
        # print (G*matrix(x) - h).size
        # print (G*matrix(x) - h).T * z

        # y = numpy.array(qpOut['y']).ravel()/t
        # s = numpy.array(h - G * matrix(x)).ravel()
        # z = numpy.array(1.0 / (t * s))

        output['subopt'] = m/t
        output['dgap'] = gap
        output['fx'] = func(x)
        output['H'] = numpy.array(H)

        if G is not None:
            output['s'] = numpy.array(s).ravel()
            output['z'] = numpy.array(z).ravel()
            output['rDual'] = numpy.array(_rDual(x, grad, z, G, y, A)).ravel()
        if A is not None:
            output['rPri'] = numpy.array(_rPri(x, A, b)).ravel()
            output['y'] = numpy.array(y).ravel()

        return x, output
    else:
        return x


def _deltaZ(x, deltaX, t, z, G, h):
    #print G
    #print h
    s = h - G * x
    rC = _rCent(z, s, t)
    #print rC
    #return div(rC,-s) - spdiag(div(z,-s)) * (G * matrix(deltaX))
    return mul(div(z,s),G * deltaX) + div(rC,-s)
    #return spdiag(div(z,-s)) * (G * matrix(deltaX)) + div(rC,-s)
    #return div(rC,-s) - div(spdiag(z) * G * deltaX,s)
    
def _rDual(x, gradFunc, z, G, y, A):
    g = matrix(gradFunc(numpy.array(x)))
    if G is not None:
        g += G.T * z
    if A is not None:
        g += A.T * y
    return g

def _rCent(z, s, t):
    # print z[0]
    # print s[0]
    # print mul(z,s)
    # print 1.0/t
    return mul(z, s) - (1.0/t)

def _rPri(x, A, b):
    return A * x - b

def _maxStepSize(z, x, deltaX, t, G, h):
    npDeltaZ = numpy.array(_deltaZ(x, deltaX, t, z, G, h))
    index = npDeltaZ<0
    npZ = numpy.array(z)
    # print "delta z"
    # print npDeltaZ
    # print "z"
    # print npZ
    # print "rate"
    # print -npZ/npDeltaZ

    s = min(1,min(-npZ[index]/npDeltaZ[index]))
    # print "newZ"
    # print npZ * s * npDeltaZ
    npZTemp = npZ + s * npDeltaZ

    while numpy.any(npZTemp<=0):
        s *= 0.5
        npZTemp = npZ + s * npDeltaZ
        # print "s"
        # print s
        # print "new z"
        # print npZTemp

    # print "diff"
    # print -npZ[index]/npDeltaZ[index]
    return float(s)

def residualLineSearch(step, x, deltaX,
                       gradFunc, t,
                       z, deltaZFunc, G, h,
                       y, deltaY, A, b,
                       dualFunc,priFunc):
    
    def F(step):
        newX = x + step * deltaX
        # print "step size"
        # print step
        # print "old x"
        # print x
        # print "new x"
        # print newX
        if z is not None:
            newZ = z + step * deltaZFunc(x, deltaX, t, z, G, h)

            # print "function of residual line search"
            # print "deleta Z func"
            # print deltaZFunc(x, deltaX, t, z, G, h)
            # print "z"
            # print z
            # print "step size"
            # print step
            # print "change of z"
            # print step * deltaZFunc(x, deltaX, t, z, G, h)
            # print "new z"
            # print newZ

            # if numpy.any(numpy.array(newZ)>0):
            #     r1 = numpy.ones(1) * numpy.Inf
            # else:
            r1 = numpy.array(dualFunc(newX, gradFunc, newZ, G, y, A)).ravel()
        else:
            r1 = numpy.zeros(1)

        if y is not None:
            newY = y + step * deltaY
            r2 = numpy.array(priFunc(x, A, b)).ravel()
        else:
            r2 = numpy.zeros(1)

        r = numpy.append(r1,r2)
        # print "full vector"
        # print r
        return float(numpy.linalg.norm(r))
    return F


def _backTrackingLineSearch(step, f, scale, alpha=0.1, beta=0.5):
    
    maxStep = step

    fx = f(0.0)
    fdeltaX = f(step)
    #g = grad(theta)

    # print "internal step"
    # print step
    # print "scale, i.e at zero"
    # print scale

    # print "at zero"
    # print fx
    # print "at one"
    # print fdeltaX

    #print "residual line search"
    i = 0
    while fdeltaX > fx + alpha * step * scale:
        # print "step="+str(step)
        # print (fdeltaX,fx + alpha * step * scale)
        step *= beta
        fdeltaX = f(step)
        if step <= 1e-16:
            if f(maxStep) < fx + alpha * step * scale:
                return maxStep, float(fdeltaX)
            else:
                return 1e-16, float(fdeltaX)
        else:
            i += 1

    # print "finish searching"
    return float(step), float(fdeltaX)
