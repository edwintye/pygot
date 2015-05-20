%load_ext autoreload
%autoreload 2
import numpy

# testing the standard Rosenbrock function

def rosen(x):
    """The Rosenbrock function"""
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def rosen_hess(x):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

import numpy as np
theta = np.array([1.3, 0.7, 0.8, 1.9, 1.2])

boxBounds = list()
for i in range(len(theta)):
    boxBounds.append([-2.0,2.0])

box = np.array(boxBounds)
lb = box[:,0]
ub = box[:,1]

A = numpy.ones((1,len(theta)))
b = numpy.ones(1) * len(theta)

from scipy.optimize import minimize
res = minimize(rosen, theta, method='Newton-CG',
               jac=rosen_der, hess=rosen_hess,
               options={'xtol': 1e-8, 'disp': True})

res = minimize(rosen, theta, method='BFGS',
               jac=rosen_der,
               options={'xtol': 1e-8, 'disp': True})

res = minimize(rosen, theta,
               method='SLSQP',
               bounds=numpy.reshape(numpy.append(lb,ub),(len(lb),2),'F'),
               jac=rosen_der,
               options={'xtol': 1e-8, 'disp': True})


from pygotools.convex import sqp, ip, ipPDC, ipPD, ipBar

## sqp

xhat, output = sqp(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=None, b=None,
                   disp=4, full_output=True)

xhat, output = sqp(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=A, b=b,
                   disp=4, full_output=True)

## interior point interface

xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=None, ub=None,
                  G=None, h=None,
                  A=None, b=None,
                  method='pdc',
                  disp=5, full_output=True)


xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=None, b=None,
                  maxiter=300,
                  method='pd',
                  disp=5, full_output=True)


xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=A, b=b,
                  method='pdc',
                  disp=3, full_output=True)


## interior point barrier


xhat, output = ipBar(rosen,
                     rosen_der,
                     x0=theta,
                     lb=None, ub=None,
                     G=None, h=None,
                     A=None, b=None,
                     disp=3, full_output=True)


xhat, output = ipBar(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=None, b=None,
                  maxiter=100,
                  disp=3, full_output=True)


xhat, output = ipBar(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=A, b=b,
                  disp=3, full_output=True)



## interior point primal dual with central path

xhat, output = ipPDC(rosen,
                     rosen_der,
                     x0=theta,
                     lb=None, ub=None,
                     G=None, h=None,
                     A=None, b=None,
                     maxiter=100,
                     disp=5, full_output=True)

xhat, output = ipPDC(rosen,
                     rosen_der,
                     x0=theta,
                     lb=lb, ub=ub,
                     G=None, h=None,
                     A=None, b=None,
                     maxiter=1000,
                     disp=5, full_output=True)

xhat, output = ipPDC(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=A, b=b,
                   maxiter=50,
                   disp=5, full_output=True)

# all search path

xhat, output = ipPD(rosen,
                   rosen_der,
                   x0=theta,
                   lb=None, ub=None,
                   G=None, h=None,
                   A=None, b=None,
                   maxiter=100,
                   disp=5, full_output=True)

xhat, output = ipPD(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=None, b=None,
                   maxiter=1000,
                   disp=5, full_output=True)

xhatA, outputA = ipPD(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=A, b=b,
                   maxiter=30,
                   disp=5, full_output=True)

 
