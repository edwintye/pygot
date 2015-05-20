%load_ext autoreload
%autoreload 2

from pygom import OperateOdeModel, common_models, odeutils, SquareLoss
import numpy
import matplotlib.pyplot 


x0 = [-1.0,1.0]
t0 = 0
# params
paramEval = [('a',0.2), ('b',0.2),('c',3.0)]

ode = common_models.FitzHugh().setParameters(paramEval).setInitialValue(x0,t0)
# the time points for our observations
t = numpy.linspace(1, 20, 100).astype('float64')
# Standard.  Find the solution.
solution,output = ode.integrate(t,full_output=True)

theta = [0.5,0.5,0.5]

objFH = SquareLoss(theta,ode,x0,t0,t,solution[1::,:],['V','R'])

boxBounds = [
    (1e-4,5.0),
    (1e-4,5.0),
    (1e-4,5.0)
    ]
boxBoundsArray = numpy.array(boxBounds)
lb = boxBoundsArray[:,0]
ub = boxBoundsArray[:,1]

from pygotools.convex import sqp,ip,ipPDC, ipPD, ipBar

## sqp

xhat, output = sqp(objFH.cost,
                   objFH.gradient,
                   x0=theta,
                   disp=4, full_output=True)

xhat, output = sqp(objFH.cost,
                   objFH.gradient,
                   x0=theta,
                   lb=lb, ub=ub,
                   disp=4, full_output=True)

## interior point interface

xhat, output = ip(objFH.cost,
                  objFH.gradient,
                  x0=theta,
                  lb=None, ub=None,
                  G=None, h=None,
                  A=None, b=None,
                  method='pdc',
                  disp=3, full_output=True)


xhat, output = ip(objFH.cost,
                  objFH.gradient,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=None, b=None,
                  maxiter=200,
                  method='bar',
                  disp=3, full_output=True)


x = numpy.array(theta)
oldFx = objFH.cost(x)
g = objFH.gradient(x)
deltaX = 0.5 * numpy.linalg.solve(objFH.jtj(x),-g)
objFH.cost(x + deltaX)

step, fc, gc, fx, old_fval, new_slope = scipy.optimize.line_search(objFH.cost,
                                                                   objFH.gradient,
                                                                   numpy.array(theta),
                                                                   deltaX,
                                                                   g,
                                                                   oldFx
                                                                   )

numpy.array(x) + step * deltaX
from pygotools.optutils import lineSearch
lineFunc = lineSearch(1, x, deltaX, objFH.cost)

res = scipy.optimize.minimize_scalar(lineFunc,
                                     method='bounded',
                                     bounds=(1e-12,1),
                                     options={'maxiter':100,'disp':True})


res = scipy.optimize.minimize_scalar(lineFunc,
                                     method='brent',
                                     bracket=(1e-12,1),
                                     options={'maxiter':100})

res = scipy.optimize.minimize_scalar(lineFunc,
                                     method='golden',
                                     bracket=(1e-12,1),
                                     options={'maxiter':100})


### with Hessian


xhat, output = ip(objFH.cost,
                  objFH.gradient,
                  objFH.jtj,
                  x0=theta,
                  lb=None, ub=None,
                  G=None, h=None,
                  A=None, b=None,
                  method='bar',
                  disp=5, full_output=True)




x = numpy.array(theta)
oldFx = objFH.cost(x)
g = objFH.gradient(x)
deltaX = numpy.linalg.solve(objFH.jtj(x),-g)
objFH.cost(x + deltaX)

step, fc, gc, fx, old_fval, new_slope = scipy.optimize.line_search(objFH.cost,
                                                                   objFH.gradient,
                                                                   numpy.array(theta),
                                                                   deltaX,
                                                                   g,
                                                                   oldFx
                                                                   )

numpy.array(x) + step * deltaX


scipy.linalg.norm(deltaX)


radius = 1.0


