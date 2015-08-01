%load_ext autoreload
%autoreload 2
import scipy.io
import numpy

A = scipy.io.loadmat('spData') 
sp = A['sp']

y = sp['y'][0][0]
x = sp['x'][0][0]
s = numpy.append(x,y,axis=1)

y = sp['Y'][0][0].ravel()
x1 = sp['X'][0][0].ravel()
n = len(y)

n = 10
x = numpy.array([0.056, 6.257, 1.204, 4.346, 4.902, 9.8, 7.624, 4.258, 2.835, 5.497]).reshape(n,1)
y = numpy.array([ 5.972,  3.391,  4.891,  5.352,  4.423,  3.057,  4.553,  4.365, 7.374, 6.554]).reshape(n,1)
s = numpy.append(x,y,axis=1)

import scipy.spatial 

D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(s))

import scipy.stats, scipy.optimize
from pygotools.responseSurface import GP, exp

rbfFun = exp()
rbfFun.f(1e-8,D)

gp = GP(y.ravel(),None,s,nugget=False)
theta = gp.getInitialGuess()
theta = numpy.array([5,1,1,1],float)

gp.negLogLike(theta)
gp.gradient(theta)
gp.hessian(theta)

from pygotools.gradient import finiteDifference
finiteDifference.forward(gp.negLogLike,theta)
finiteDifference.central(gp.negLogLike,theta)
finiteDifference.forwardHessian(gp.negLogLike,theta)
finiteDifference.forwardGradCallHessian(gp.gradient,theta)-gp.hessian(theta)


box = list()
for i in range(len(theta)):
    box.append((1e-8,10))

boxArray = numpy.array(box)

out = scipy.optimize.minimize(fun=gp.negLogLike,
                              jac=gp.gradient,
                              hess=gp.hessian,
                              x0=theta,
                              method='dogleg')
                              bounds=box)

scipy.optimize.root(gp.gradient,theta)

from pygotools.convex import sqp,ip,ipPDC, ipPD, ipBar, trustRegion
xhat, output = sqp(gp.negLogLike,
                   gp.gradient,
                   gp.hessian,
                   method='line',
                   x0=theta,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=4, full_output=True)

xhat, output = sqp(gp.negLogLike,
                   gp.gradient,
                   gp.hessian,
                   x0=theta,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=4, full_output=True)

xhat, output = ip(gp.negLogLike,
                   gp.gradient,
                  method='pdc',
                   x0=theta,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=5, full_output=True)


gp.predict(s,x1)
gp.predictGradient(s,x1)
gp.predictGradient(s[0],x1[0])
from pygotools.gradient.finiteDifference import forward
forward(gp.predict,s[0])
gp.predictGradient(s[0])

res = scipy.optimize.minimize(fun=gp.negLogLike,
                              x0=theta,
                              jac=gp.gradient,
                              method='SLSQP',
                              bounds=box)

from pygotools.optutils import optimTestFun

f = optimTestFun.rosen

niter = 100
A = numpy.random.rand(niter,2)*4 - 2

y = numpy.array([f(a) for a in A])

gp = GP(y,A,A)

theta1 = theta.tolist() + [1.,1.]
box = list()
for i in range(3):
    box.append((None,None))
for i in range(3):
    box.append((1e-8,numpy.inf))

res = scipy.optimize.minimize(fun=gp.negLogLike,
                              x0=theta1,
                              jac=gp.gradient,
                              method='tnc',
                              options={'maxiter':1000},
                              bounds=box,
                              callback=callback)

def callback(x):
    a = x.copy()
    b = gp.negLogLike(x)
    print numpy.append(a,b)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig) 
ax.plot_trisurf(A[:,0], A[:,1], numpy.log(y), cmap=cm.jet, linewidth=0.2)
ax.view_init(elev=40,azim=105)
plt.show()

niterPred = 10000
APred = numpy.random.rand(niterPred,2)*4 - 2

yPred = gp.predict(APred)


fig = plt.figure()
ax = Axes3D(fig) 
ax.plot_trisurf(APred[:,0], APred[:,1], numpy.log(yPred), cmap=cm.jet, linewidth=0.2)
ax.view_init(elev=40,azim=105)
plt.show()

res = scipy.optimize.minimize(fun=gp.predict,
                              x0=APred[numpy.random.randint(len(APred))-1],
                              method='nelder-mead')


res1 = scipy.optimize.minimize(fun=gp.predict,
                              jac=gp.predictGradient,
                              x0=APred[numpy.random.randint(len(APred))-1],
                              method='newton-cg')

res2 = scipy.optimize.minimize(fun=gp.predict,
                               jac=gp.predictGradient,
                               hess=gp.predictHessian,
                               x0=APred[numpy.random.randint(len(APred))-1],
                               method='trust-ncg')


j = numpy.random.randint(len(A))-1

from pygotools.gradient import finiteDifference
finiteDifference.forward(gp.predict,A[j])
finiteDifference.central(gp.predict,A[j])
gp.predictGradient(A[j])

gp.predictHessian(A[j])

numpy.linalg.eig(rbfFun(1e-8,D))[0]
numpy.linalg.cholesky(rbfFun(1e-8,D))

phi = 1.0
sigma2 = 2.0
tau2 = 0.5
beta = 1.0

G = sigma2*rbfFun(phi,D);
H = tau2*numpy.eye(n) + G;
W = (y-x*beta);

scipy.stats.multivariate_normal.logpdf(y,x*beta,H)

def gpLL(theta):
    beta = theta[0]
    phi = theta[1]
    sigma2 = theta[2]
    tau2 = theta[3]
    
    G = sigma2*rbfFun(phi,D);
    H = tau2*numpy.eye(n) + G;

    try:
        return -scipy.stats.multivariate_normal.logpdf(y,x1*beta,H,True)
    except Exception as e:
        print phi
        print e
        raise e

theta = [beta,phi,sigma2,tau2]
gpLL(theta)

import scipy.optimize

box = list()
for i in range(len(theta)):
    box.append((1e-8,numpy.inf))

scipy.optimize.minimize(fun=gpLL,
                        x0=theta,
                        bounds=box)

def gpGrad(theta):
    beta = theta[0]
    phi = theta[1]
    sigma2 = theta[2]
    tau2 = theta[3]
    
    G = sigma2*rbfFun(phi,D);
    H = tau2*numpy.eye(n) + G;
    W = (y-x1*beta);

    invH = numpy.linalg.solve(H,numpy.eye(len(y)))
    quadForm = invH.dot(W);

    DG = -D*G
    dPhi = invH.dot(DG);
    dSigma = invH.dot(G/sigma2); 
    dTau = invH;

    g = numpy.zeros(len(theta))

    g[0] = x1.T.dot(quadForm);
    g[1] = (quadForm.T.dot(DG.dot(quadForm)) - numpy.trace(dPhi))/2; 
    g[2] = (quadForm.T.dot((G/sigma2).dot(quadForm)) - numpy.trace(dSigma))/2;
    g[3] = (quadForm.T.dot(quadForm) - numpy.trace(dTau))/2;
    return -g

gpGrad(theta)


res = scipy.optimize.minimize(fun=gpLL,
                              x0=theta,
                              jac=gpGrad,
                              method='tnc',
                              bounds=box)


import pygot.gradient.finiteDifference
pygot.gradient.finiteDifference.forward(gpLL,theta)
pygot.gradient.finiteDifference.central(gpLL,theta)
gpGrad(theta)


def gpHessian(theta):
    beta = theta[0]
    phi = theta[1]
    sigma2 = theta[2]
    tau2 = theta[3]
    
    G = sigma2*rbfFun(phi,D);
    H = tau2*numpy.eye(n) + G;
    W = (y-x1*beta);

    invH = numpy.linalg.solve(H,numpy.eye(len(y)))
    quadForm = invH.dot(W);

    DG = -D*G
    
    dPhi = invH.dot(DG);
    dSigma = invH.dot(G/sigma2); 
    dTau = invH;

    A = numpy.zeros((3,3));
    #A[0,0] = quadForm.T.dot(2*DG.dot(dPhi) - (D**2)*G).dot(quadForm) + numpy.trace(invH.dot((D**2)*G) - dPhi.dot(dPhi));
    # print quadForm.T.dot(2*DG.dot(dPhi) - (D**2)*G).dot(quadForm)
    # print numpy.linalg.eig(2*DG.dot(dPhi) - (D**2)*G)[0]
    # print numpy.trace(invH.dot(D**2)*G - dPhi.dot(dPhi));
    # print numpy.trace(invH.dot((D**2)*G) - dPhi.dot(dPhi));
    A[0,0] = quadForm.T.dot(2*DG.dot(dPhi)).dot(quadForm) + numpy.trace(-dPhi.dot(dPhi));

    #A[0,1] = quadForm.T.dot(DG.dot(dSigma) + (G/sigma2).dot(dPhi) - DG/sigma2).dot(quadForm) + numpy.trace(invH.dot(DG/sigma2) - dPhi.dot(dSigma));

    A[0,1] = quadForm.T.dot(DG.dot(dSigma) + dPhi).dot(quadForm) + numpy.trace(-dPhi.dot(dSigma));

    A[0,2] = quadForm.T.dot(DG.dot(dTau) + dPhi).dot(quadForm) + numpy.trace(-dPhi.dot(dTau));

    # sigma2
    A[1,1] = quadForm.T.dot(2*(G/sigma2).dot(dSigma)).dot(quadForm) + numpy.trace(-dSigma.dot(dSigma));
    #A[1,1] = quadForm.T.dot((2*G/sigma2).dot(invH).dot(G/sigma2)).dot(quadForm) + numpy.trace( -dSigma.dot(dSigma));
        
    A[1,2] = quadForm.T.dot((G/sigma2).dot(dTau) + dSigma).dot(quadForm) + numpy.trace(-dSigma.dot(dTau));
        
    # tau2
    A[2,2] = quadForm.T.dot(2*dTau).dot(quadForm) + numpy.trace(-dTau.dot(dTau));

    A *= 0.5
    A = A.T + A - numpy.diag(numpy.diag(A));
    
    B = numpy.zeros((1,3));
    B[:,0] = x1.T.dot(dPhi.dot(quadForm));
    B[:,1] = x1.T.dot(dSigma.dot(quadForm));
    B[:,2] = x1.T.dot(dTau.dot(quadForm));
    
    #print B
    #print (x1.T.dot(invH)).dot(x1)

    XTX = numpy.zeros((1,1))
    XTX[:] = (x1.T.dot(invH)).dot(x1)

    return numpy.bmat([[ XTX, B],[B.T, A]]);

gpHessian(theta)

scipy.optimize.minimize(fun=gpLL,
                        x0=theta,
                        jac=gpGrad,
                        hess=gpHessian,
                        method='newton-cg',
                        bounds=box)


pygot.gradient.finiteDifference.forwardHessian(gpLL,theta)
pygot.gradient.finiteDifference.forwardGradCallHessian(gpGrad,theta)

