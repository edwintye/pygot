from pyGenericOdeModel import operateOdeModel,modelDef,odeLossFunc
import numpy
import scipy.integrate
import math,time,copy
import matplotlib.pyplot 
from openopt import NLP,GLP

from numpy import cos, arange, ones, asarray, abs, zeros, sqrt, asscalar
from pylab import legend, show, plot, subplot, xlabel, subplots_adjust
from string import rjust, ljust, expandtabs


ode = modelDef.LegrandEbola()

ode.setParameters(theta).setInitialValue(x0,t0)

solution = ode.integrate(t[1::])
ode.plot()

ode.ode(x0,t0)

numpy.array(ode.evalOde(state=x0,time=t0).tolist())

numpy.array(ode.evalOde(state=x0,time=t0))

ode.Jacobian(x0,t0)

import mpmath

import sympy.mpmath

sympy.mpmath.array(ode.evalJacobian(state=x0,time=t0))

numpy.array(ode.evalJacobian(state=x0,time=t0))

type(ode.evalJacobian(state=x0,time=t0))

numpy.array(ode.evalJacobian(state=x0,time=t0).tolist(),float)

mpmath.matrix(ode.evalJacobian(state=x0,time=t0))

numpy.array(mpmath.matrix(ode.evalJacobian(state=x0,time=t0)))

numpy.array(ode.evalJacobian(state=x0,time=t0))

ode.evalGrad(state=x0,time=t0)

ode.Grad(x0,t0)

yDeath = [29.0, 59.0, 60.0, 62.0, 66.0, 70.0, 70.0, 80.0, 83.0, 86.0, 95.0, 101.0, 106.0, 108.0, 122.0, 129.0, 136.0, 141.0, 143.0, 149.0, 155.0, 157.0, 158.0, 157.0, 171.0, 174.0, 186.0, 193.0, 208.0, 215.0, 226.0, 264.0, 267.0, 270.0, 303.0, 305.0, 307.0, 309.0, 304.0, 310.0, 310.0, 314.0, 319.0, 339.0, 346.0, 358.0, 363.0, 367.0, 373.0, 377.0, 380.0, 394.0, 396.0, 406.0, 430.0, 494.0, 517.0, 557.0, 568.0, 595.0, 601.0, 632.0, 635.0, 648.0, 710.0, 739.0, 768.0, 778.0, 843.0, 862.0, 904.0, 926.0, 997.0]

yCase = [49.0, 86.0, 86.0, 86.0, 103.0, 112.0, 112.0,  122.0,  127.0,  143.0,  151.0,  158.0,  159.0,  168.0,  197.0, 203.0,  208.0,  218.0,  224.0,  226.0,  231.0,  235.0,  236.0,  233.0,  248.0,  258.0,  281.0,  291.0,  328.0,  344.0, 351.0,  398.0,  390.0,  390.0,  413.0,  412.0,  408.0,  409.0,  406.0,  411.0,  410.0,  415.0,  427.0,  460.0,  472.0, 485.0, 495.0, 495.0, 506.0, 510.0, 519.0,  543.0,  579.0, 607.0,  648.0,  771.0,  812.0,  861.0,  899.0,  936.0, 942.0, 1008.0, 1022.0, 1074.0, 1157.0, 1199.0, 1298.0, 1350.0, 1472.0, 1519.0, 1540.0, 1553.0, 1906.0]


# the corresponding
t = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0, 13.0, 16.0, 18.0, 20.0, 23.0, 25.0, 26.0, 29.0, 32.0, 35.0, 40.0, 42.0, 44.0, 46.0, 49.0, 51.0, 62.0, 66.0, 67.0, 71.0, 73.0, 80.0, 86.0, 88.0, 90.0, 100.0, 102.0, 106.0, 108.0, 112.0, 114.0, 117.0, 120.0, 123.0, 126.0, 129.0, 132.0, 135.0, 137.0, 140.0, 142.0, 144.0, 147.0, 149.0, 151.0, 157.0, 162.0, 167.0, 169.0, 172.0, 175.0, 176.0, 181.0, 183.0, 185.0, 190.0, 193.0, 197.0, 199.0, 204.0, 206.0, 211.0, 213.0, 218.0]

# initial guess
theta1 = numpy.array([0.5,0.5,0.5,### the beta
                     2.0,1.5,1.0,3.0,      ### the omega
                     0.5,0.5,0.5, ### alpha, delta, theta
                     10,10])  ### kappa,intervention time

theta2 = numpy.array([7.0*0.588,7.0*0.794,7.0*1.653, ### the beta
                     10.0,9.6,5.0,2.0, ### the omega
                     7.0,0.81,0.80, ### alpha, delta, theta
                     300,12.0])  ### kappa,intervention time


# starting value
y = numpy.reshape(numpy.append(numpy.array(yCase),numpy.array(yDeath)),(len(yCase),2),'F') / 1175e4 
x0 = [1, 49.0/1175e4, 0.0, 0.0, 0.0, 29.0/1175e4,0.]
t0 = 0

theta = numpy.array([0.1,0.1,0.1, ### the beta
                     10.0,9.6,5.0,2.0, ### the omega
                     7.0,0.81,0.80, ### alpha, delta, theta
                     0.1,1.0])  ### kappa,intervention time


theta3 = [  2.47063919e-02,   8.34896582e-01,   2.00000000e+00,
         6.52250983e+01,   8.88857685e+01,   7.62593478e+00,
         3.51943235e+01,   2.77555756e-17,   8.14831909e-01,
         3.46944695e-18,   3.73088898e+02,   1.03377245e+01]

theta4 = [  2.31507228e-02,   1.99997489e+00,   1.45839939e+00,
         1.00000000e+02,   1.00000000e+02,   9.87598235e+01,
         1.53450124e+01,   1.00000000e-08,   7.10575971e-01,
         9.99999990e-01,   3.29899411e+02,   1.28215876e+01]

objLegrand = odeLossFunc.squareLoss(theta,ode,x0,t0,t[1::],y[1::,:],['I','R'],[1175e4,1175e4])

objLegrand.cost(theta)
objLegrand.cost(theta1)

objLegrand.sensitivity(theta,intName='ivode')
objLegrand.sensitivity(theta,intName='lsoda')
ode.Jacobian(x0,t0)
scipy.linalg.eig(ode.Jacobian(x0,t0))[0]

S = solution[:,0]
E = solution[:,1]
I = solution[:,2]
H = solution[:,3]
F = solution[:,4]
R = solution[:,5]
import matplotlib
f,axarr = matplotlib.pyplot.subplots(2,3)
axarr[0,0].plot(t,S)
axarr[0,0].set_title('Susceptible')
axarr[0,1].plot(t,E)
axarr[0,1].set_title('Exposed')
axarr[0,2].plot(t,I)
axarr[0,2].plot(t,y[:,0],'r')
axarr[0,2].set_title('Infectious')
axarr[1,0].plot(t,H)
axarr[1,0].set_title('Hospitalised')
axarr[1,1].plot(t,F)
axarr[1,1].set_title('Awaiting Burial')
axarr[1,2].plot(t,R)
axarr[1,2].plot(t,y[:,1],'r')
axarr[1,2].set_title('Removed')
matplotlib.pyplot.xlabel('Days from outbreak')
matplotlib.pyplot.ylabel('Population')
matplotlib.pyplot.show()


boxBounds = [
          (0.0,2.0),
          (0.0,2.0),
          (0.0,2.0),
          (0.0,100.0),
          (0.0,100.0),
          (0.0,100.0),
          (0.0,100.0),
          (0.0,100.0),
          (0.0,1.0),
          (0.0,1.0),
          (0.0,1000.0),
          (0.0,218.0)
           ]

cons = ({'type': 'ineq',
          'fun' : lambda x: numpy.array([x[3]-x[5],x[4]-x[5]])
          })


npBox = numpy.array(boxBounds)

lb = npBox[:,0] + 1e-8
ub = npBox[:,1] - 1e-8

# define our inequality
A = numpy.zeros((2,len(theta)))
A[0,3] = -1
A[0,5] = 1
A[1,4] = -1
A[1,5] = 1
b = numpy.zeros(2)

## First, we want to find some good starting conditions
totalSample = 100000
initSample = list()
target = list()    
for i in range(0,totalSample):
    sample = numpy.zeros(12)
    for j in range(0,12):
        sample[j] = numpy.random.uniform(boxBounds[j][0],boxBounds[j][1],1)
    if (sample[4] > sample[5]) and (sample[3] > sample[5]):
        initSample.append(sample)
        target.append(objLegrand.cost(sample))


def F(n):
    initSample = list()
    target = list()    
    i = 0
    while i < n:
        sample = numpy.zeros(12)
        for j in range(0,12):
            sample[j] = numpy.random.uniform(boxBounds[j][0],boxBounds[j][1],1)
        #print (sample[4] > sample[5]) and (sample[3] > sample[5])
        if (sample[4] > sample[5]) and (sample[3] > sample[5]):
            initSample.append(sample)
            target.append(objLegrand.cost(sample))
            i += 1
    return (target,initSample)

F(3)

## then, we want to test the scipy solvers

res = scipy.optimize.differential_evolution(objLegrand.cost,bounds=boxBounds)

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


pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rRalg = pro.solve('ralg')

pro = NLP(f=objLegrand.cost,x0=theta3,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rRalg = pro.solve('ralg')



pro = NLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rSLSQP = pro.solve('scipy_slsqp')

# we try to refine the solution

pro = NLP(f=objLegrand.cost,x0=rPswarm.xf,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rSLSQP = pro.solve('ralg')

pro = NLP(f=objLegrand.cost,x0=rPswarm.xf,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rSLSQP = pro.solve('scipy_slsqp')



objLegrand.cost(rPswarm.xf)
objLegrand.cost()
gSens,output=objLegrand.sensitivity(full_output=True)
solution = output['sens'][:,:6]

# here we pretend the bounds don't exist
pro2 = GLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub)
pro2.plot = True
r2 = pro2.solve('galileo')

pro = GLP(f=objLegrand.cost,x0=theta,df=objLegrand.sensitivity,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rDe = pro.solve('de')

pro = GLP(f=objLegrand.cost,x0=theta,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rPswarm = pro.solve('pswarm',size=100)

pro = GLP(f=objLegrand.cost,x0=rPswarm.xf,lb=lb,ub=ub,A=A,b=b)
pro.plot = True
rPswarm = pro.solve('pswarm',size=100,maxIter=1000,maxFunEvals=100000)



colors = ['b', 'k', 'y', 'r', 'g']
#############
solvers = ['ralg','lincher','scipy_slsqp','ptn','mma']
#############
colors = colors[:len(solvers)]

lines, results = [], {}
for j in range(len(solvers)):
    solver = solvers[j]
    color = colors[j]
    p = NLP(objSIR.cost, theta, df=objSIR.sensitivity,lb = lb, ub = ub, ftol = 1e-6, maxFunEvals = 1e7, maxIter = 1220, plot = 1, color = color, iprint = 0, legend = [solvers[j]], show= False, xlabel='time', goal='minimum', name='nlp3')
    if solver == 'algencan':
        p.gtol = 1e-1
    elif solver == 'ralg':
        p.debug = 1

    r = p.solve(solver, debug=1)
    print 'c1 evals:', cc1, 'c2 evals:', cc2, 'c3 evals:', cc3
    results[solver] = (r.ff, p.getMaxResidual(r.xf), r.elapsed['solver_time'], r.elapsed['solver_cputime'], r.evals['f'], r.evals['c'], r.evals['h'])
    subplot(2,1,1)
    F0 = asscalar(p.f(p.x0))
    lines.append(plot([0, 1e-15], [F0, F0], color= colors[j]))

# for i in range(2):
#     subplot(2,1,i+1)
#     legend(solvers)

subplots_adjust(bottom=0.2, hspace=0.3)

xl = ['Solver                              f_opt     MaxConstr   Time   CPUTime  fEvals  cEvals  hEvals']
for i in range(len(results)):
    xl.append((expandtabs(ljust(solvers[i], 16)+' \t', 15)+'%0.2f'% (results[solvers[i]][0]) + '        %0.1e' % (results[solvers[i]][1]) + ('      %0.2f'% (results[solvers[i]][2])) + '    %0.2f      '% (results[solvers[i]][3]) + str(results[solvers[i]][4]) + '   ' + rjust(str(results[solvers[i]][5]), 5) + expandtabs('\t' +str(results[solvers[i]][6]),8)))

xl = '\n'.join(xl)
subplot(2,1,1)
xlabel('Time elapsed (without graphic output), sec')

from pylab import *
subplot(2,1,2)
xlabel(xl)

show()
