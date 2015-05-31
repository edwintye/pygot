
__all__ = [
    'trustRegion',
    'trustExact'
    ]

from .convexUtil import _checkInitialValue
from .approxH import *
from pygotools.optutils.checkUtil import checkArrayType
from pygotools.optutils.disp import Disp
from pygotools.gradient.finiteDifference import forward

import numpy
import scipy
import scipy.linalg

EPSILON = 1e-8
atol = 1e-8
reltol = 1e-8

def trustRegion(func, grad, hessian=None, x0=None,
                maxiter=100,
                method='exact',
                disp=0, full_output=False):

    x = checkArrayType(x0)
    p = len(x)

    if grad is None:
        def finiteForward(x,func,p):
            def finiteForward1(x):
                return forward(func,x.ravel())
            return finiteForward1
        grad = finiteForward(x,func,p)

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

    if method is None:
        trustMethod = trustExact
    elif type(method) is str:
        if method.lower()=='exact':
            trustMethod = trustExact
        else:
            raise Exception("Input name of hessian is not recognizable")

    fx = None
    oldGrad = None
    deltaX = None
    oldFx = numpy.inf
    i = 0
    oldi = -1
    j = 0
    tau = 1.0
    radius = 1.0
    maxRadius = 1.0
    
    dispObj = Disp(disp)

    while maxiter>i:

        # if we have successfully moved on, then
        # we would need to recompute some of the quantities
        if i!=oldi:
            g = grad(x)
            fx = func(x)
        
            if hessian is None:
                if oldGrad is None:
                    H = numpy.eye(len(x))
                else:
                    diffG = numpy.array(g - oldGrad)
                    H = approxH(H, diffG, deltaX)
            else:
                H = hessian(x)

        deltaX, tau = trustMethod(x, g, H, radius)
        deltaX = deltaX.real
        M = diffM(deltaX, g, H)
    
        # print x
        # print deltaX
        newFx = func(x + deltaX)
        predRatio = (fx - newFx) / M(deltaX)
        
        if predRatio>=0.75:
            if tau>0.0:
                radius = min(2.0*radius, maxRadius)
        elif predRatio<=0.25:
            radius *= 0.25
    
        if predRatio>=0.25:
            oldGrad = g
            x += deltaX
            oldFx = fx
            fx = newFx
            i +=1
            oldi = i - 1
            # we only allow termination if we make a move
            if (abs(fx-oldFx)/fx)<=reltol:
                break
            if abs(deltaX.dot(g))<=atol:
                break
        else:
            oldi = i

        
        dispObj.d(j, x , fx, deltaX, g, i)
        j += 1
        

    if full_output:
        output = dict()
        output['totalIter'] = i
        output['outerIter'] = j

        output['fx'] = func(x)
        output['H'] = H
        output['g'] = g

        return x, output
    else:
        return x

def trustExact(x, g, H, radius=1.0, maxiter=10):
    # we use tau as the size of our regularization because
    # lambda is a reserved keyword
    tau = 0
    p = len(x)

    e = scipy.linalg.eig(H)[0]
    e1 = min(e)
    if e1<0:
        tau = -e1+EPSILON

    for i in range(maxiter):
        try:
            R = scipy.linalg.cholesky(H + tau*scipy.eye(p))
        except:
            print "tau = "+str(tau)
            print scipy.linalg.eig(H + tau*scipy.eye(p))[0]
            raise Exception('shit')
        # scipy.linalg.solve(H + tau*numpy.eye(p),-g)
        pk = scipy.linalg.solve_triangular(R, -g, trans='T')
        pk = scipy.linalg.solve_triangular(R, pk, trans='N')
        
        a = scipy.linalg.norm(pk)
        if i==0:
            fx = 1.0/radius - 1.0/a
            if a<=radius:
                return pk, tau
        else:
            oldFx = fx
            fx = 1.0/radius - 1.0/a
            if fx<=0.0 or abs(oldFx-fx)<=0.1:
                # we are goint to force another iteration anyway just to refine
                # it because we have
                qk = scipy.linalg.solve_triangular(R, pk, trans='T')
                tau += (a/scipy.linalg.norm(qk))**2  * (a - radius) / radius
                return pk, tau

        qk = scipy.linalg.solve_triangular(R, pk, trans='T')
        tau += (a/scipy.linalg.norm(qk))**2  * (a - radius) / radius
        # print "tau = "+str(tau) + " phi2 = " +str(fx)+ " with e1 = "+str(e1)
        # print pk
    
    return  pk, tau

def trustSubspace(x, g, H, radius=1.0, maxiter=10):
    # we use tau as the size of our regularization because
    # lambda is a reserved keyword
    tau = 0
    p = len(x)

    e = scipy.linalg.eig(H)[0]
    e1 = min(e)
    if e1<0:
        tau = -e1+EPSILON

    R = scipy.linalg.cholesky(H + tau*scipy.eye(p))
    pS1 = scipy.linalg.solve_triangular(R, -g, trans='T')
    pS1 = scipy.linalg.solve_triangular(R, pS1, trans='N')
    
    pS2 = -g.copy()
    rhs = numpy.append(pS1,pS2,axis=0)
        

    for i in range(maxiter):
        try:
            R = scipy.linalg.cholesky(H + tau*scipy.eye(p))
        except:
            print "tau = "+str(tau)
            print scipy.linalg.eig(H + tau*scipy.eye(p))[0]
            raise Exception('shit')
        # scipy.linalg.solve(H + tau*numpy.eye(p),-g)
        pk = scipy.linalg.solve_triangular(R, -g, trans='T')
        pk = scipy.linalg.solve_triangular(R, pk, trans='N')
        
        a = scipy.linalg.norm(pk)
        if i==0:
            fx = 1.0/radius - 1.0/a
            if a<=radius:
                return pk, tau
        else:
            oldFx = fx
            fx = 1.0/radius - 1.0/a
            if fx<=0.0 or abs(oldFx-fx)<=0.1:
                # we are goint to force another iteration anyway just to refine
                # it because we have
                qk = scipy.linalg.solve_triangular(R, pk, trans='T')
                tau += (a/scipy.linalg.norm(qk))**2  * (a - radius) / radius
                return pk, tau

        qk = scipy.linalg.solve_triangular(R, pk, trans='T')
        tau += (a/scipy.linalg.norm(qk))**2  * (a - radius) / radius
        # print "tau = "+str(tau) + " phi2 = " +str(fx)+ " with e1 = "+str(e1)
        # print pk
    
    return  pk, tau


def diffM(deltaX, g, H):
    def M(deltaX):
        m = -g.dot(deltaX) - 0.5 * deltaX.dot(H).dot(deltaX)
        return m
    return M

