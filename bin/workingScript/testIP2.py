%load_ext autoreload
%autoreload 2

from pygom import OperateOdeModel, common_models, odeutils, SquareLoss
import numpy
import matplotlib.pyplot 
import scipy.optimize

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

res = scipy.optimize.minimize(fun=objFH.cost,
                              jac=objFH.sensitivity,
                              hess=objFH.jtj,
                              x0 = theta,
                              method = 'dogleg',
                              options = {'disp':True},
                              callback=objFH.thetaCallBack)

boxBounds = [
    (1e-4,5.0),
    (1e-4,5.0),
    (1e-4,5.0)
    ]
boxBoundsArray = numpy.array(boxBounds)
lb = boxBoundsArray[:,0]
ub = boxBoundsArray[:,1]

from pygotools.convex import sqp,ip,ipPDC, ipPD, ipBar, trustRegion

## trust
xhat, output = trustRegion(objFH.cost, objFH.gradient, hessian=objFH.jtj,
                x0=theta,
                maxiter=100,
                method='exact',
                disp=3, full_output=True)

xhat, output = trustRegion(objFH.cost, objFH.gradient, hessian=objFH.hessian,
                x0=theta,
                maxiter=100,
                method='exact',
                disp=3, full_output=True)


xhatA, outputA = trustRegion(objFH.cost, objFH.gradient, hessian='SR1',
                x0=theta,
                maxiter=100,
                method='exact',
                disp=3, full_output=True)



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
                  method='bar',
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
p = len(theta)
oldFx = objFH.cost(x)
g = objFH.gradient(x)
H = objFH.jtj(x)
deltaX = numpy.linalg.solve(H,-g)
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

if scipy.linalg.norm(deltaX)>=radius:
    tau = 1.0
    deltaX = numpy.linalg.solve(objFH.jtj(x)+tau*numpy.eye(p),-g)
    scipy.linalg.norm(deltaX)


def phi(tau,H,g,p,radius):
    def F(tau):
        deltaX = numpy.linalg.solve(H + tau*numpy.eye(p), -g)
        return 1.0/radius - 1.0/scipy.linalg.norm(deltaX)
    return F



f = phi(0.0,objFH.jtj(x),g,p,radius)
f(1.0)

sol = scipy.optimize.root(f,0.0)

# simplistic version

tau = 0.0
diffTau = 1.0

for i in range(5):
    newTau = tau + diffTau
    fx = f(tau)

    oldFx = fx
    fx = f(newTau)
    tau = newTau

    diffTau = -(fx * diffTau)/(fx - oldFx)
    print "tau = " +str(newTau) + " with f(x) = " +str(fx)


# more complicated version

tau = 0.0
for i in range(8):
    R = scipy.linalg.cholesky(H + tau*numpy.eye(p))
    #scipy.linalg.solve(H + tau*numpy.eye(p),-g)
    pk = scipy.linalg.solve_triangular(R,-g,trans='T')
    pk = scipy.linalg.solve_triangular(R,pk,trans='N')
    qk = scipy.linalg.solve_triangular(R,pk,trans='T')
    a = scipy.linalg.norm(pk)
    tau += (a/scipy.linalg.norm(qk))**2  * (a - radius) / radius
    print "tau = " +str(tau) + " with f(x) = " +str(f(tau))


scipy.linalg.norm(scipy.linalg.solve(H+tau*numpy.eye(p),-g))

scipy.linalg.cholesky(H + -min(scipy.linalg.eig(H)[0])*numpy.eye(p))


from pygotools.convex import trustExact


x = numpy.array(theta)
p = len(theta)
oldFx = objFH.cost(x)
g = objFH.gradient(x)
H = objFH.jtj(x)
deltaX = numpy.linalg.solve(H,-g)
objFH.cost(x + deltaX)



x = numpy.array(theta)
p = len(theta)
radius = 1.0

func = objFH.cost
grad = objFH.gradient
hessian = objFH.jtj

def diffM(deltaX, g, H):
    def M(deltaX):
        m = -g.dot(deltaX) - 0.5 * deltaX.dot(H).dot(deltaX)
        return m
    return M

for i in range(100):
    
    g = grad(x)
    H = hessian(x)
    fx = func(x)

    deltaX, tau = trustExact(x, g, H, radius)
    M = diffM(deltaX, g, H)
    
    newFx = func(x + deltaX)
    predRatio = (fx - newFx) / M(deltaX)
    
    if predRatio>=0.75:
        if tau>0.0:
            radius = min(2.0*radius, 1)
    elif predRatio<=0.25:
        radius *= 0.25
    
    if predRatio>=0.25:
        x += deltaX
        print "x = " +str(x)
        fx = newFx

    print "tau = " +str(tau) + " with f(x) = " +str(func(x))+ " and m(x) "+str(M(deltaX))
    print "radius = "+str(radius)+ " ratio = " +str(predRatio)
    print ""

