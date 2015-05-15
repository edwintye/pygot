from pygom import OperateOdeModel,common_models,odeLossFunc
import numpy
import scipy.integrate
import math,time,copy
import matplotlib.pyplot 
from openopt import NLP,GLP

from numpy import cos, arange, ones, asarray, abs, zeros, sqrt, asscalar
from pylab import legend, show, plot, subplot, xlabel, subplots_adjust
from string import rjust, ljust, expandtabs

t0 = 0
# the initial state, normalized to zero one
x0 = [1,1.27e-6,0]
# set the time sequence that we would like to observe
t = numpy.linspace(0, 150, 100)
numStep = len(t)
# Standard.  Find the solution.
ode = modelDef.SIR()
ode.setParameters([0.5,1.0/3.0])
ode.setInitialValue(x0,t0)
solution = ode.integrate(t[1::],full_output=False)

import scipy.stats
d = dict()
d['b'] = scipy.stats.gamma(1,0,0.5)
d['k'] = scipy.stats.gamma(1,0,1/3)


c = d['b'].rvs(1000)
numpy.mean(c)
numpy.var(c)

ode.setParameters(d).setInitialValue(x0,t0)
solution = ode.integrate(t[1::],full_output=False)
ode.plot()

theta = [0.2,0.2]

# y = copy.copy(solution[:,1:2])
# y[1::] = y[1::] + numpy.random.normal(loc=0.0,scale=0.1,size=numStep-1)

# odeSIR = odeLossFunc.squareLoss([0.5,1.0/3.0] ,ode,x0,t0,t[1:len(t)],y[1:len(t)],'R')

objSIR = odeLossFunc.squareLoss(theta,ode,x0,t0,t[1::],solution[1::,1:3],['I','R'])

box = [(0.,2.),(0.,2.)]
npBox = numpy.array(box)
lb = npBox[:,0]
ub = npBox[:,1]

pro = GLP(f=objSIR.cost,x0=theta,lb=lb,ub=ub)
pro.plot = True
rGalileo = pro.solve('galileo')

pro = GLP(f=objSIR.cost,x0=theta,lb=lb,ub=ub)
pro.plot = True
rDe = pro.solve('de')

pro = GLP(f=objSIR.cost,x0=theta,lb=lb,ub=ub)
pro.plot = True
rPSwarm = pro.solve('pswarm')

pro = NLP(f=objSIR.cost,df=objSIR.sensitivity,x0=theta,lb=lb,ub=ub)
pro.plot = True
rLincher = pro.solve('lincher')

pro = NLP(f=objSIR.cost,x0=theta,df=objSIR.sensitivity,lb=lb,ub=ub)
pro.plot = True
rSqlcp = pro.solve('sqlcp')

pro = NLP(f=objSIR.cost,x0=theta,df=objSIR.sensitivity,lb=lb,ub=ub)
pro.plot = True
rSLSQP = pro.solve('scipy_slsqp')

pro = NLP(f=objSIR.cost,x0=theta,df=objSIR.sensitivity,lb=lb,ub=ub)
pro.plot = True
rRalg = pro.solve('ralg')



colors = ['b', 'k', 'y', 'r', 'g']
#############
solvers = ['ralg','lincher','scipy_slsqp','sqlcp']
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
