from pyGenericOdeModel import operateOdeModel,modelDef,odeLossFunc
import numpy
import scipy.integrate
import math,time,copy
import matplotlib
from openopt import NLP,GLP


# initial guess
theta = numpy.array([5.0,5.0,5.0])
ode = modelDef.SEIR(theta)

yDeath = [29.0, 59.0, 60.0, 62.0, 66.0, 70.0, 70.0, 80.0, 83.0, 86.0, 95.0, 101.0, 106.0, 108.0, 122.0, 129.0, 136.0, 141.0, 143.0, 149.0, 155.0, 157.0, 158.0, 157.0, 171.0, 174.0, 186.0, 193.0, 208.0, 215.0, 226.0, 264.0, 267.0, 270.0, 303.0, 305.0, 307.0, 309.0, 304.0, 310.0, 310.0, 314.0, 319.0, 339.0, 346.0, 358.0, 363.0, 367.0, 373.0, 377.0, 380.0, 394.0, 396.0, 406.0, 430.0, 494.0, 517.0, 557.0, 568.0, 595.0, 601.0, 632.0, 635.0, 648.0, 710.0, 739.0, 768.0, 778.0, 843.0, 862.0, 904.0, 926.0, 997.0]

yCase = [49.0, 86.0, 86.0, 86.0, 103.0, 112.0, 112.0,  122.0,  127.0,  143.0,  151.0,  158.0,  159.0,  168.0,  197.0, 203.0,  208.0,  218.0,  224.0,  226.0,  231.0,  235.0,  236.0,  233.0,  248.0,  258.0,  281.0,  291.0,  328.0,  344.0, 351.0,  398.0,  390.0,  390.0,  413.0,  412.0,  408.0,  409.0,  406.0,  411.0,  410.0,  415.0,  427.0,  460.0,  472.0, 485.0, 495.0, 495.0, 506.0, 510.0, 519.0,  543.0,  579.0, 607.0,  648.0,  771.0,  812.0,  861.0,  899.0,  936.0, 942.0, 1008.0, 1022.0, 1074.0, 1157.0, 1199.0, 1298.0, 1350.0, 1472.0, 1519.0, 1540.0, 1553.0, 1906.0]

# the corresponding time
t = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 13.0, 16.0, 18.0, 20.0, 23.0, 25.0, 26.0, 29.0, 32.0, 35.0, 40.0, 42.0, 44.0, 46.0, 49.0, 51.0, 62.0, 66.0, 67.0, 71.0, 73.0, 80.0, 86.0, 88.0, 90.0, 100.0, 102.0, 106.0, 108.0, 112.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 137.0, 140.0, 142.0, 144.0, 147.0, 149.0, 151.0, 157.0, 162.0, 167.0, 169.0, 172.0, 175.0, 176.0, 181.0, 183.0, 185.0, 190.0, 193.0, 197.0, 199.0, 204.0, 206.0, 211.0, 213.0, 218.0]

# starting value
y = numpy.reshape(numpy.append(numpy.array(yCase),numpy.array(yDeath)),(len(yCase),2),'F') / 1175e4 
x0 = [1.0, 0., 49.0/1175e4, 29.0/1175e4]
t0 = 0

objLegrand = odeLossFunc.squareLoss(theta,ode,x0,t0,t[1::],y[1::,:],['I','R'],[1175e4,1175e4])

objLegrand.cost()
objLegrand.sensitivity()

objLegrand.adjointInterpolate()
objLegrand.adjointInterpolate1()
objLegrand.adjointInterpolate2()

[3.26106524e+00,   2.24798702e-04,   1.23660721e-02]
[3.25271562e+00,   2.16905309e-04,   1.20873233e-02]

[ 0.02597769,  9.00452147,  0.00992397]
[0.02598875,  9.00004177,  0.00993199]

HJTJ,outputHJTJ = objLegrand.HessianForward([ 0.02598392,  9.00000002,  0.00992684],full_output=True)

HJTJ,outputHJTJ = objLegrand.HessianForward([3.25271562e+00,   2.16905309e-04,   1.20873233e-02],full_output=True)

HJTJ,outputHJTJ = objLegrand.HessianForward(full_output=True)
numpy.linalg.inv(HJTJ)
numpy.linalg.eig(HJTJ)[0]

solution = ode.integrate(t[1::])

f,axarr = matplotlib.pyplot.subplots(2,2)
axarr[0,0].plot(t,solution[:,0])
axarr[0,0].set_title('Susceptible')
axarr[0,1].plot(t,solution[:,1])
axarr[0,1].set_title('Exposed')
axarr[1,0].plot(t,solution[:,2])
axarr[1,0].plot(t,y[:,0],'r')
axarr[1,0].set_title('Infectious')
axarr[1,1].plot(t,solution[:,3])
axarr[1,1].plot(t,y[:,1],'r')
axarr[1,1].set_title('Removed')
matplotlib.pyplot.xlabel('Days from outbreak')
matplotlib.pyplot.ylabel('Population')
matplotlib.pyplot.show()

boxBounds = [
          (0.0,10.0),
          (0.0,10.0),
          (0.0,10.0)
           ]

resQP = scipy.optimize.minimize(fun = objLegrand.cost,
                              jac = objLegrand.sensitivity,
                              x0 = theta,
                              bounds = boxBounds,
                              options = {'disp':True},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'SLSQP')

# find a suitable starting point
resDE = scipy.optimize.differential_evolution(objLegrand.cost,bounds=boxBounds,disp=True,polish=False,seed=20921391)

resQPRefine = scipy.optimize.minimize(fun = objLegrand.cost,
                                      jac = objLegrand.sensitivity,
                                      x0 = resDE['x'],
                                      bounds = boxBounds,
                                      options = {'disp':True},
                                      callback=odeLossFunc.thetaCallBack,
                                      method = 'SLSQP')

resQPRefine2 = scipy.optimize.minimize(fun = objLegrand.cost,
                                      jac = objLegrand.adjointInterpolate,
                                      x0 = resDE['x'],
                                      bounds = boxBounds,
                                      options = {'disp':True},
                                      callback=odeLossFunc.thetaCallBack,
                                      method = 'SLSQP')


resQPPrior = scipy.optimize.minimize(fun = objLegrand.cost,
                              jac = objLegrand.adjointInterpolate,
                              x0 = [0.027,9.0,0.01],
                              bounds = boxBounds,
                              options = {'disp':True,'maxiter':1000},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'L-BFGS-B')

resQPPrior2 = scipy.optimize.minimize(fun = objLegrand.cost,
                              jac = objLegrand.adjointInterpolate,
                              x0 = [3.26106524e+00,   2.24798702e-04,   1.23660721e-02],
                              bounds = boxBounds,
                              options = {'disp':True,'maxiter':1000},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'L-BFGS-B')

## then, we want to test the scipy solvers

solvers = [
    'ralg',
    'lincher',
    'gsubg',
    'scipy_slsqp',
    'scipy_cobyla',
    'interalg',
    'auglag',
    'ptn',
    'mma'
    ]


npBox = numpy.array(boxBounds)

lb = npBox[:,0] + 1e-8
ub = npBox[:,1] - 1e-8


pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro = NLP(f=objLegrand.cost,x0=resDE['x'],df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rRalg = pro.solve('ralg',xtol=1e-20)

pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.adjointInterpolate,lb=lb,ub=ub)
pro = NLP(f=objLegrand.cost,x0=resDE['x'],df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rLincher = pro.solve('lincher')

pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro = NLP(f=objLegrand.cost,x0=resDE['x'],df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rSLSQP = pro.solve('scipy_slsqp')

# GLP
pro = GLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro = GLP(f=objLegrand.cost,x0=resDE['x'],df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rPswarm = pro.solve('pswarm',size=50,maxFunEvals=100000,maxIter=1000)


pro2 = GLP(f=objLegrand.cost,x0=rPswarm.xf,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro2.plot = True
rPswarm2 = pro2.solve('pswarm')

pro3 = NLP(f=objLegrand.cost,x0=rPswarm2.xf,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro3.plot = True
rPswarm3 = pro3.solve('ralg')


pro = GLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rGalileo = pro.solve('galileo')

pro = GLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro.plot = True
rDe = pro.solve('de')

#
# DIRECT!
# 
from pyOptimUtil.direct import polyOperation, directAlg

directObj = directAlg.direct(objLegrand.cost,lb,ub)
rectListOptim,output = directObj.divide(1000,numBox=10000,full_output=True)

A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

directObj = directAlg.direct(objLegrand.cost,lb,ub,A,b)
polyListOptim,output = directObj.divide(50,numBox=5000,full_output=True)




#
# initial values
# 

thetaIV = resQPRefine['x'].tolist() + x0
thetaIV = theta.tolist() + x0
thetaIV[3] -= 1e-8
boxBoundsIV = boxBounds + [(0.,1.),(0.,1.),(0.,1.),(0.,1.)]
    
npBoxIV = numpy.array(boxBoundsIV)

lbIV = npBoxIV[:,0] + 1e-8
ubIV = npBoxIV[:,1] - 1e-8

objLegrand = odeLossFunc.squareLoss(theta,ode,x0,-1,t,y,['I','R'],[1175e4,1175e4])
objLegrand = odeLossFunc.squareLoss(theta,ode,x0,-1,t,y,['I','R'])

objLegrand.costIV(xIV1)
objLegrand.costIV(xIV)
residualIV = objLegrand.residualIV(xIV)

from pyGenericOdeModel import operateOdeUtil
solution,output = operateOdeUtil.integrateFuncJac(ode._odeT,ode._JacobianT,xIV1[-4:],-1,t,includeOrigin=True,full_output=True,intName='dopri5')
solution1,output1 = scipy.integrate.odeint(ode.ode,xIV1[-4:],numpy.append(-1,t),Dfun=ode.Jacobian,full_output=True)

solutionDiff = solution - solution1

f,axarr = matplotlib.pyplot.subplots(2,2)
axarr[0,0].plot(t,solutionDiff[1::,0])
axarr[0,0].set_title('Susceptible')
axarr[0,1].plot(t,solutionDiff[1::,1])
axarr[0,1].set_title('Exposed')
axarr[1,0].plot(t,solutionDiff[1::,2])
axarr[1,0].set_title('Infectious')
axarr[1,1].plot(t,solutionDiff[1::,3])
axarr[1,1].set_title('Removed')
matplotlib.pyplot.show()

solution = ode1.setInitialValue(xIV[-4:],-1).integrate(t)
(((y - solution[1::,2:4])*1175e4)**2).sum()
(((y - solution1[1::,2:4])*1175e4)**2).sum()

objLegrand.costIV(xIV1)
objLegrand.cost(xIV1[:3])
residualIV = objLegrand.residualIV(xIV1)
residual = objLegrand.residual(xIV1[:3])
residualDiff = residual - residualIV
residDiff = residualIV - solution[1::,2:4]

f,axarr = matplotlib.pyplot.subplots(1,2)
axarr[0].plot(t,residDiff[:,0])
axarr[1].plot(t,residDiff[:,1])
matplotlib.pyplot.show()

(residual**2).sum()

objLegrand.costIV(thetaIV)
objLegrand.sensitivityIV(thetaIV)

resDEIV = scipy.optimize.differential_evolution(objLegrand.costIV,bounds=boxBoundsIV,disp=True,polish=False,seed=20921391)

[  6.26103284e-01,   1.54562516e-03,   7.53153929e-03,
         3.45156318e-01,   3.14829487e-05,   1.33026722e-05,
         7.85790181e-06]

resQPIV = scipy.optimize.minimize(fun = objLegrand.costIV,
                              jac = objLegrand.sensitivityIV,
                              x0 = thetaIV,
                              bounds = boxBoundsIV,
                              options = {'disp':True,maxIter=1000},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'SLSQP')

resQPIVRefine = scipy.optimize.minimize(fun = objLegrand.costIV,
                              jac = objLegrand.sensitivityIV,
                              x0 = xIV1,
                              bounds = boxBoundsIV,
                              options = {'disp':True,'maxiter':1000},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'L-BFGS-B')

resQPIVRefine2 = scipy.optimize.minimize(fun = objLegrand.costIV,
                              jac = objLegrand.sensitivityIV,
                              x0 = rDE.xf,
                              bounds = boxBoundsIV,
                              options = {'disp':True},
                              callback=odeLossFunc.thetaCallBack,
                              method = 'SLSQP')


xIVBest = xIV
xIV = numpy.append(objLegrand._theta.copy(),objLegrand._x0.copy())

xIV1 = resDEIV['x']
xIV = resQPIVRefine['x']

xIV = [  1.64133359e-01,   2.19655111e-03,   7.57427832e-03,
         9.55224172e-01,   2.50934442e-05,   1.29125571e-05,
         7.62115796e-06]

array([  1.64133359e-01,   2.19655111e-03,   7.57427832e-03,
         9.55224172e-01,   2.50934442e-05,   1.29125571e-05,
         7.62115796e-06])

objLegrand.costIV(xIV)
objLegrand.sensitivityIV(xIV)
HJTJIV = objLegrand.HessianForward()

objLegrand.costIV(xIV1)
objLegrand.sensitivityIV(xIV1)
HJTJIV1 = objLegrand.HessianForward()

import copy
ode1 = copy.deepcopy(ode)
solution = ode1.setInitialValue(xIV[-4:],-1).setParameters(xIV[:3]).integrate(t)
((y - solution[1::,2:4])**2).sum()

f,axarr = matplotlib.pyplot.subplots(2,2)
axarr[0,0].plot(t,solution[1::,0])
axarr[0,0].set_title('Susceptible')
axarr[0,1].plot(t,solution[1::,1])
axarr[0,1].set_title('Exposed')
axarr[1,0].plot(t,solution[1::,2])
axarr[1,0].plot(t,y[:,0],'r')
axarr[1,0].set_title('Infectious')
axarr[1,1].plot(t,solution[1::,3])
axarr[1,1].plot(t,y[:,1],'r')
axarr[1,1].set_title('Removed')
matplotlib.pyplot.xlabel('Days from outbreak')
matplotlib.pyplot.ylabel('Population')
matplotlib.pyplot.show()

solution1 = ode1.setInitialValue(xIV1[-4:],-1).integrate(t)
((y - solution1[1::,2:4])**2).sum()

f,axarr = matplotlib.pyplot.subplots(2,2)
axarr[0,0].plot(t,solution1[1::,0])
axarr[0,0].set_title('Susceptible')
axarr[0,1].plot(t,solution1[1::,1])
axarr[0,1].set_title('Exposed')
axarr[1,0].plot(t,solution1[1::,2])
axarr[1,0].plot(t,y[:,0],'r')
axarr[1,0].set_title('Infectious')
axarr[1,1].plot(t,solution1[1::,3])
axarr[1,1].plot(t,y[:,1],'r')
axarr[1,1].set_title('Removed')
matplotlib.pyplot.xlabel('Days from outbreak')
matplotlib.pyplot.ylabel('Population')
matplotlib.pyplot.show()




pro = NLP(f=objLegrand.costIV,df=objLegrand.sensitivityIV,x0=thetaIV,lb=lbIV,ub=ubIV)
pro = NLP(f=objLegrand.costIV,df=objLegrand.sensitivityIV,x0=xIV1,lb=lbIV,ub=ubIV)


rLincher = pro.solve('lincher',maxFunEvals=100000,maxIter=1000)

rRalg = pro.solve('ralg',maxFunEvals=100000,maxIter=1000)

rSLSQP = pro.solve('scipy_slsqp',maxFunEvals=100000,maxIter=1000)

rCOBYLA = pro.solve('scipy_cobyla',maxFunEvals=100000,maxIter=1000)

rBFGS = pro.solve('scipy_lbfgsb',maxFunEvals=100000,maxIter=1000)

rTNC = pro.solve('scipy_tnc',maxFunEvals=100000,maxIter=1000)

# GLP
pro = GLP(f=objLegrand.costIV,x0=thetaIV,lb=lbIV,ub=ubIV)
pro = GLP(f=objLegrand.costIV,x0=xIV1,lb=lbIV,ub=ubIV)

rPswarm = pro.solve('pswarm',size=50,maxFunEvals=100000,maxIter=1000,maxNonSuccess=50)

rGalileo = pro.solve('galileo',maxFunEvals=100000,maxIter=1000,maxNonSuccess=100)

rDE = pro.solve('de',maxFunEvals=100000,maxIter=1000,maxNonSuccess=100)

rDE2 = pro.solve('de',maxFunEvals=100000,maxIter=1000,maxNonSuccess=100)


# direct, which fails badly
directObjIV = directAlg.direct(objLegrand.costIV,lbIV,ubIV)
rectListOptimIV,outputIV = directObjIV.divide(1000,numBox=10000,full_output=True)


#
# IV, S + E + H + F
# 

thetaIVSEHF = theta.tolist() + (numpy.array(x0)[[0,1,3,4]]).tolist()

boxBoundsIVSEHF = boxBounds + [(0,1)] + [(0,0.1)]*3
    
npBoxIVSEHF = numpy.array(boxBoundsIVSEHF)

lbIVSEHF = npBoxIVSEHF[:,0] + 1e-8
ubIVSEHF = npBoxIVSEHF[:,1] - 1e-8

# define our inequality
AIVSEHF = numpy.zeros((2,len(thetaIVSEHF)))
AIVSEHF[0,3] = -1
AIVSEHF[0,5] = 1
AIVSEHF[1,4] = -1
AIVSEHF[1,5] = 1
bIVSEHF = numpy.zeros(2)

objLegrand = odeLossFunc.squareLoss(thetaIVSEHF,ode,x0,t0,t[1::],y[1::,:],['I','R'],[1175e4,1175e4],targetState=['S','E','H','F'])


pro = GLP(f=objLegrand.costIV,x0=thetaIVSEHF,lb=lbIVSEHF,ub=ubIVSEHF,A=AIVSEHF,b=bIVSEHF)
pro.plot = True
rPswarm = pro.solve('pswarm',size=50,maxFunEvals=100000,maxIter=1000)

pro1 = NLP(f=objLegrand.costIV,df=objLegrand.sensitivityIV,x0=thetaIVSEHF,lb=lbIVSEHF,ub=ubIVSEHF,A=AIVSEHF,b=bIVSEHF)
pro1.plot = True
rLincher = pro1.solve('lincher',maxFunEvals=100000,maxIter=1000)


# refinement
pro = GLP(f=objLegrand.cost,x0=rPswarm.xf,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rPswarm = pro.solve('pswarm',size=100,maxIter=1000,maxFunEvals=100000)


objLegrand.costIV(rPswarm.xf)
objLegrand.cost()
objLegrand.sensitivity()
objLegrand.adjointInterpolate()
objLegrand.adjointInterpolate1()

gSens,output=objLegrand.sensitivity(full_output=True)
solution = output['sens'][:,:6]


pro = NLP(f=objLegrand.costIV,x0=thetaIVS,df=objLegrand.sensitivityIV,lb=lbIVS,ub=ubIVS,A=AIVS,b=bIVS)
pro.plot = True
rRalg = pro.solve('ralg')
