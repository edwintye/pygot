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

res = minimize(rosen, theta, method='trust-ncg',
               jac=rosen_der, hess=rosen_hess,
               options={'xtol': 1e-8})

res = minimize(rosen, theta, method='dogleg',
               jac=rosen_der, hess=rosen_hess)

res = minimize(rosen, theta, method='BFGS',
               jac=rosen_der,
               options={'xtol': 1e-8, 'disp': True})

res = minimize(rosen, theta,
               method='SLSQP',
               bounds=numpy.reshape(numpy.append(lb,ub),(len(lb),2),'F'),
               jac=rosen_der,
               options={'xtol': 1e-8, 'disp': True})


from pygotools.convex import sqp, ip, ipPDC, ipPD, ipBar

## abc

from cvxopt import solvers, matrix, blas
from pygotools.convex.convexUtil import _setup
z, G, h, y, A, b = _setup(lb, ub, None, None, None, None)
p = len(theta)
x = numpy.array(theta).reshape(p,1)
radius = 1.0

GTemp = numpy.append(numpy.zeros((1,p)), numpy.eye(p), axis=0)
hTemp = numpy.zeros(p+1)
hTemp[0] += radius

GTemp = numpy.append(G, GTemp, axis=0)
hTemp = numpy.append(h - G.dot(x), hTemp)
dims = {'l': G.shape[0], 'q': [p+1], 's':  []}

H = rosen_hess(theta)
H = numpy.eye(p)
g = rosen_der(theta)

qpOut = solvers.coneqp(matrix(H), matrix(g), matrix(GTemp), matrix(hTemp), dims)

# converting to a cp

def F(x=None, z=None):
    if x is None: return 0, matrix(0.0,(p,1))
    # H = matrix(rosen_hess(numpy.array(x).ravel()))
    # H = matrix(rosen_hess(theta))
    H = matrix(numpy.eye(p))
    g = matrix(rosen_der(numpy.array(x).ravel()))
    g = matrix(rosen_der(theta))
    f = 0.5 * x.T * H * x + g.T * x
    df = (H * x + g).T
    #df = -(g).T
    if z is None: return f, df
    H = z[0] * H
    return f, df, H
        
sol1 = solvers.cp(F, matrix(GTemp), matrix(hTemp), dims)

def F2(x=None, z=None):
    if x is None: return 0, matrix(0.0,(p,1))
    # H = matrix(rosen_hess(numpy.array(x).ravel()))
    # H = matrix(rosen_hess(theta))
    H = matrix(numpy.eye(p))
    # g = matrix(rosen_der(numpy.array(x).ravel()))
    g = matrix(rosen_der(theta))
    f = blas.nrm2(H*x+g)**2
    df = 2.0*(H.T * (H*x+g)).T
    if z is None: return f, df
    H = z[0] * 2.0 * H.T * H
    return f, df, H
        
sol2 = solvers.cp(F2, matrix(GTemp), matrix(hTemp), dims)

print sol1['x']
print sol2['x']
print qpOut['x']

import scipy.sparse

H = rosen_hess(theta)
m,n = GTemp.shape
c = matrix([1.0] + [0.0]*n)

hTemp1 = matrix([0.0]+(-g).tolist())
GTemp1 = matrix(numpy.array(scipy.sparse.bmat([
    [[-1.0],None],
    [None,H]
    ]).todense()))

GTemp1 = matrix(numpy.append(numpy.append(numpy.array([0]*m).reshape(m,1),numpy.array(GTemp),axis=1),
                             numpy.array(GTemp1),
                             axis=0))

hTemp1 = matrix(numpy.append(hTemp,hTemp1))
dims1 = {'l': G.shape[0], 'q': [n+1,n+1], 's': []}

solSOCP = solvers.conelp(c, GTemp1, hTemp1, dims1)

print solSOCP['x'][1::]

## testing the change in objective function

print F(matrix(theta))[0]
print F(matrix(theta) + solSOCP['x'][1::])[0]
print F(matrix(theta) + sol1['x'])[0]

blas.nrm2(solSOCP['x'][1::])
blas.nrm2(sol1['x'])
blas.nrm2(qpOut['x'])

blas.nrm2(hTemp1[-n::] - matrix(GTemp)[-n::,:] * solSOCP['x'][1::] )

blas.nrm2(hTemp[1::] - GTemp[1::,1::] * sol1['x'] )

# now an socp

print sol1['x']
print sol2['x']
print qpOut['x']
print solSOCP['x'][1::]
## sqp

xhat, output = sqp(rosen,
                   rosen_der,
                   x0=theta,
                   maxiter=100,
                   method='trust',
                   disp=5, full_output=True)

xhat, output = sqp(rosen,
                   rosen_der,
                   rosen_hess,
                   x0=theta,
                   maxiter=100,
                   method='trust',
                   disp=5, full_output=True)

xhat, output = sqp(rosen,
                   rosen_der,
                   rosen_hess,
                   x0=theta,
                   lb=lb, ub=ub,
                   maxiter=100,
                   method='trust',
                   disp=5, full_output=True)

xhat, output = sqp(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=None, b=None,
                   maxiter=100,
                   method='line',
                   disp=5, full_output=True)

xhat, output = sqp(rosen,
                   rosen_der,
                   x0=theta,
                   lb=lb, ub=ub,
                   G=None, h=None,
                   A=A, b=b,
                   method='trust',
                   disp=3, full_output=True)


## interior point interface

xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=None, ub=None,
                  G=None, h=None,
                  A=None, b=None,
                  maxiter=50,
                  method='pdc',
                  disp=5, full_output=True)


xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=None, b=None,
                  maxiter=500,
                  method='pdc',
                  disp=5, full_output=True)


xhat, output = ip(rosen,
                  rosen_der,
                  x0=theta,
                  lb=lb, ub=ub,
                  G=None, h=None,
                  A=A, b=b,
                  method='pd',
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

 
#
from pygotools.convex import trustRegion, trustExact

## trust
xhat, output = trustRegion(rosen,
                           rosen_der,
                           rosen_hess,
                           x0=theta,
                           maxiter=100,
                           method='exact',
                           disp=3, full_output=True)

xhat, output = trustRegion(rosen,rosen_der,
                           hessian='BFGS',
                           x0=theta,
                           maxiter=100,
                           method='exact',
                           disp=3, full_output=True)


xhatA, outputA = trustRegion(rosen,rosen_der,
                             hessian='SR1',
                             x0=theta,
                             maxiter=100,
                             method='exact',
                             disp=3, full_output=True)


## test subspace
theta = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
radius = 0.5

for i in range(10):
    g = rosen_der(theta)
    H = rosen_hess(theta)
    # trustExact(theta, g, H, radius=1.0, maxiter=10)

    # g.T.dot(g)/(g.T.dot(H).dot(g.T)) * g

    e = scipy.linalg.eig(H)[0]
    tau = 1.5 * abs(min(e)) 
    Haug = H + tau*scipy.eye(len(theta))

    R = scipy.linalg.cholesky(Haug)
    pFS = scipy.linalg.solve_triangular(R, -g, trans='T')
    pFS = scipy.linalg.solve_triangular(R, pFS, trans='N')
    
    if scipy.linalg.norm(pFS)<=radius:
        theta += pFS
        print 1
    else:
    
        # pU = -g
        pU = g.T.dot(g)/(g.T.dot(H).dot(g)) * -g
        pU = -g

        lhs = scipy.sparse.bmat([
            [pU.T.dot(Haug).dot(pU), pU.T.dot(Haug).dot(pFS)],
            [pFS.T.dot(Haug).dot(pU), pFS.T.dot(Haug).dot(pFS)]
        ])

        rhs = numpy.append(pU.T.dot(g),pFS.T.dot(g))
        eta = scipy.linalg.solve(lhs.todense(),-rhs)

        pBar = eta[0] * pU + eta[1] * pFS
        pBar = 1.0 * pU + 0.0 * pFS
        scipy.linalg.norm(pBar)
    
        if scipy.linalg.norm(pBar)<=radius:
            theta += pBar
            print 2
        else:
            theta += radius * pU/scipy.linalg.norm(pU)
            print 3

    print theta


scipy.linalg.norm(pBar)

eta[0] = 1.0 / scipy.linalg.norm(pU)
eta[1] = 1.0 / scipy.linalg.norm(pFS)

eta[0]**2 * scipy.linalg.norm(pU)**2 
eta[1]**2 * scipy.linalg.norm(pFS)**2 


e,V = scipy.linalg.eig(H)

V.dot(numpy.diag(e)).dot(V.T)

V[:,3].dot(H).dot(V[:,3])/numpy.inner(V[:,3],V[:,3])

V[:,3].T.dot(pFS)



v = -pU/scipy.linalg.norm(pU)
v.T.dot(H).dot(v) / (v.T.dot(v))


