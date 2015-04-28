#%load_ext autoreload
#%autoreload 2
from pyGenericOdeModel import operateOdeModel,odeLossFunc,modelDef
from pyOptimUtil.direct import optimTestFun, directAlg, directUtil
import pyOptimUtil.gradient
import numpy
import scipy.integrate
import math,time,copy,os
import matplotlib.pyplot 


lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)

print("Now we start DIRECT, with scaled output")

rectListOptim,output = directAlg.directOptim(optimTestFun.rosen,lb,ub,
                                                 iteration=50,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

directUtil.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

from pyOptimUtil.direct import optimTestFun, directAlg, directUtil
import pyOptimUtil.direct 

optimalListLipschitz = pyOptimUtil.direct.identifyPotentialOptimalRectangleLipschitz(rectListOptim)
optimalList = pyOptimUtil.direct.identifyPotentialOptimalRectanglePareto(rectListOptim)

directUtil.plotParetoFront(rectListOptim,optimalList)

# can also use the class instead

directClass = directAlg.direct(optimTestFun.rosen,lb,ub)
rectListOptim = directClass.divide()

optimalListLipschitz = pyOptimUtil.direct.identifyPotentialOptimalRectangleLipschitz(rectListOptim)
optimalListPareto = pyOptimUtil.direct.identifyPotentialOptimalRectanglePareto(rectListOptim)

directUtil.plotDirectBox(rectListOptim,lb,ub,False,optimalListPareto)

directUtil.plotParetoFront(rectListOptim,optimalListPareto)

#####
#
# SIR model with two parameters
#
#####


# Again, standard SIR model with 2 parameter.  See the first script!
# initial time
t0 = 0
# the initial state, normalized to zero one
x0 = [1,1.27e-6,0]
# params
paramEval = [('b',0.5), ('k',1.0/3.0)]
# initialize the model
ode = modelDef.SIR().setParameters(paramEval).setInitialValue(x0,t0)

# set the time sequence that we would like to observe
t = numpy.linspace(1, 150, 100)
numStep = len(t) 
# Standard.  Find the solution.
solution = ode.integrate(t)

y = solution[1::,1:2]

theta = [0.4,1.0/2.0]
objSIR = odeLossFunc.squareLoss(theta,ode,x0,t0,t,y,['I','R'])

lb = numpy.array([0.,0.],float)
ub = numpy.array([2.,2.],float)

pyOptimUtil.gradient.simplexGradient

beta,h,adjustedStepSize,info = pyOptimUtil.gradient.simplexGradient.linear(objSIR.cost, theta, h = 0.1, S=None, lb=lb, ub=ub)

info['S'].dot(beta)

numpy.arccos(info['S'].dot(betaNorm))

numpy.arccos(info['S'].dot([0.5,0.5]/numpy.linalg.norm([0.5,0.5])))

numpy.degrees(numpy.arccos(info['S'].dot([0.5,0.5]/numpy.linalg.norm([0.5,0.5]))))

info['S'].dot([0.5,0])

numpy.argmin(abs(info['S'].dot(beta)))

betaNorm = beta / numpy.linalg.norm(beta)

info['S'].dot(betaNorm)

pyOptimUtil.gradient.linear(objSIR.cost, theta, h = 0.1, S=None, lb=lb, ub=ub)

pyOptimUtil.gradient.closestVector(info['S'],beta)

pyOptimUtil.gradient.simplexGradient.closestVector(info['S'],beta)

A = info['S'] + beta

for i in A:
    print i/numpy.linalg.norm(i)

for i in info['S']:
    a = beta - (i - beta)
    print a / numpy.linalg.norm(a)


# more approximations
objSIR.forward()
objSIR.central()

# the real gradients
objSIR.sensitivity()
objSIR.adjointInterpolateTest()

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

Xpoly2 = PolynomialFeatures(degree=2).fit_transform(info['X'])

numpy.linalg.matrix_rank(Xpoly2)


rectListOptim,output = rectOperation.directOptim(objSIR.cost,lb,ub,
                                                 iteration=20,
                                                 numBox=500, 
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

rectOperation.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)
rectOperation.plotParetoFront(rectListOptim)


############################################################
# cvx part

ode = modelDef.SIR().setParameters(paramEval).setInitialValue(x0,t0)
objSIR = odeLossFunc.squareLoss(theta,ode,x0,t0,t,y[1::],'R')

from cvxopt import solvers, matrix

def optimOde(objOde,ode,theta):
    p = len(theta)
    ff0 = numpy.zeros(ode._numState*p*p)
    s0 = numpy.zeros(p*ode._numState)
    def F(x=None, z=None):
        if x is None: return 0, matrix(theta)
        if min(x) <= 0.0: return None
        objJac,output = objOde.jac(theta=x,full_output=True)
        f = matrix((output['resid']**2).sum())
        Df = matrix(numpy.reshape(objJac.transpose().dot(-2*output['resid']),(1,p)))
        if z is None: return f, Df
        ode.setParameters(theta)
        ffParam = numpy.append(numpy.append(objSIR._x0,s0),ff0)
        solutionHessian,outputHessian = scipy.integrate.odeint(ode.odeAndForwardforward,
                                                               ffParam,t,
                                                               full_output=True)

        H = numpy.zeros((2,2))
        for i in range(0,len(output['resid'])):
            H += -2*output['resid'][i] * numpy.reshape(solutionHessian[i,-4:],(2,2))

        H += 2*objJac.transpose().dot(objJac)
        print H
        print scipy.linalg.cholesky(H)
        print numpy.linalg.matrix_rank(H)
        H = matrix(z[0] * H)
        return f, Df, H
    return solvers.cp(F)

cvxSol = optimOde(objSIR,ode,theta)

objSIR.forward(theta)
objSIR.sensitivity(theta)
objSIR.adjointInterpolate2(theta)
objSIR.forwardHessian(theta)
objSIR.forwardGradCallHessian(theta)
scipy.linalg.cholesky(objSIR.forwardHessian(theta))
scipy.linalg.cholesky(objSIR.forwardGradCallHessian(theta))

objSIR.jac()

    solutionHessian,outputHessian = scipy.integrate.odeint(ode.odeAndForwardforward,
                                                           ffParam,t,
                                                           full_output=True)

    H = numpy.zeros((2,2))
    for i in range(0,len(output['resid'])):
        H = H + -2*output['resid'][i] * numpy.reshape(solutionHessian[i,-4:],(2,2))
