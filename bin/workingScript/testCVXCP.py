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

xhat = objFH.fit(theta,lb=boxBoundsArray[:,0],ub=boxBoundsArray[:,1])

from cvxopt import solvers, matrix
solvers.options['abstol'] = 1e-4
solvers.options['reltol'] = 1e-4

def odeObj(obj, theta, boxBounds):
    numParam = obj._numParam
    G = matrix(numpy.append(numpy.eye(numParam), -numpy.eye(numParam), axis=0))
    h = matrix(numpy.append(boxBounds[:, 1], -boxBounds[:, 0], axis=0))
    dims = {'l': G.size[0], 'q': [], 's':  []}
    def F(x=None, z=None):
        if x is None: return 0, matrix(theta)
        if min(x) <= 0.0: return None
        H, o = obj.jtj(numpy.array(x).ravel(),full_output=True)
        r = o['resid']
        f = matrix((r ** 2).sum())
        Df = matrix(o['grad']).T
        if z is None: return f, Df
        H = z[0] * matrix(H)
        return f, Df, H
    return solvers.cp(F, G, h)
       
cvxOut = odeObj(objFH, theta, numpy.array(boxBounds))

p = objFH._numParam
G = matrix(numpy.append(numpy.eye(p), -numpy.eye(p), axis=0))
h = matrix(numpy.append(boxBoundsArray[:, 1], -boxBoundsArray[:, 0], axis=0))

solvers.options['show_progress'] = False
solvers.options['show_progress'] = True

theta = [0.5,0.5,0.5]

for i in range(10):
    theta = numpy.array(theta)

    JTJ,o = objFH.jtj(theta,full_output=True)
    hTemp = h - G*matrix(theta)
    qpOut = coneqp(matrix(JTJ), matrix(o['grad']), G, hTemp)
    delta = numpy.array(qpOut['x']).ravel()
    # backTrack = lineSearch(1,theta,delta,objFH.cost)
    # t = scipy.optimize.fminbound(backTrack,0,1,full_output=True,disp=0)
    t, fx = backTrackingLineSearch(1, theta, delta, o['grad'], objFH.cost)
    # t, fx = exactLineSearch(1, theta, delta, objFH.cost)

    #print qpOut['x']
    theta += t * delta
    # print theta.T
    # print numpy.array(delta).dot(o['grad'])
    print theta
    #print objFH.cost(theta)

thetaHat = theta
qpOutHat = qpOut

from pygotools.optutils.optCondition import backTrackingLineSearch, exactLineSearch
exactLineSearch(1, theta, delta, objFH.cost)
t, fx = backTrackingLineSearch(1, theta, delta, objFH.grad, objFH.cost)

import scipy.optimize

res = scipy.optimize.minimize(fun=objFH.cost,
                              x0=theta,
                              jac=objFH.gradient,
                              hess=objFH.jtj,
                              bounds = numpy.array(boxBounds),
                              method = 'trust-ncg',
                              options = {'maxiter':1000},
                              callback=objFH.thetaCallBack)



from pygotools.convex import sqp, ip

boxBoundsArray = numpy.array(boxBounds)
lb = numpy.array([0.0,0.0,0.0])
ub = numpy.array([5.0,5.0,5.0])

xhat,qpOut = sqp(objFH.cost, objFH.gradient, hessian=objFH.jtj, x0=[0.5,0.5,0.5], lb=lb, ub=ub,disp=3,full_output=True)

xhat,qpOut = sqp(objFH.cost, objFH.gradient, hessian=None, x0=[0.5,0.5,0.5], lb=lb, ub=ub,disp=3,full_output=True)

xhat,output = ip(objFH.cost,
                 objFH.gradient,
                 hessian=objFH.jtj,
                 x0=[0.5,0.5,0.5],
                 lb=lb, ub=ub,
                 method='bar',
                 disp=3, full_output=True)

xhat,output = ip(objFH.cost, objFH.gradient, hessian=None, x0=[0.5,0.5,0.5], lb=lb, ub=ub,disp=3,full_output=True)

xhat,output = ipD(objFH.cost, objFH.gradient, hessian=objFH.jtj, x0=[0.5,0.5,0.5], lb=lb, ub=ub,disp=3,full_output=True)

xhat,output = ipD(objFH.cost, objFH.gradient, hessian=None, x0=[0.5,0.5,0.5], lb=lb, ub=ub,disp=3,full_output=True)

from pygotools.optutils.consMani import addLBUBToInequality
G,h = addLBUBToInequality(lb,ub)
G = matrix(G)
h = matrix(h)
z = qpOut['z']

numpy.einsum('ji,ik->jk',G.T, G*z )
print mul(matrix(G),matrix(z)[:,matrix(0,(1,len(G[0])))])

numpy.einsum('ji,ik->jk',G.T, mul(matrix(G),matrix(z)[:,matrix(0,(1,G.size[1]))]))
from cvxopt import blas 
y = matrix(0.0,(G.size[1],1))
blas.gemv(Gs, matrix(1.0,(G.size[0],1)),y,'T')

C = numpy.zeros((4,4))

for i in range(3):
    C += numpy.outer(GT[:,i],G[i])
    
