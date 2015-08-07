import scipy.spatial, scipy.optimize
import numpy
from pygotools.gradient.finiteDifference import forward, central, richardsonExtrapolation

n = 10
x0 = numpy.array([0.056, 6.257, 1.204, 4.346, 4.902, 9.8, 7.624, 4.258, 2.835, 5.497]).reshape(n,1)
x1 = numpy.array([6.6820, 3.8590, 6.3270, 0.5460, 6.7870, 9.1670, 2.2650, 7.6700, 5.4310, 0.3630]).reshape(n,1)
y = numpy.array([ 5.972,  3.391,  4.891,  5.352,  4.423,  3.057,  4.553,  4.365, 7.374, 6.554]).reshape(n,1)

s = numpy.append(x0,x1,axis=1)
D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(s))



from pygotools.responseSurface import GP, exp

rbfFun = exp()
rbfFun.f(1e-8,D)

# print D

gp = GP(y.ravel(),None,s,nugget=True)
theta = gp.getInitialGuess()

box = [(1e-8,10)] * len(theta)
boxArray = numpy.array(box)

out = scipy.optimize.minimize(fun=gp.negLogLike,
                              jac=gp.gradient,
                              x0=theta,
                              bounds=box)

print out

# print 'gradient at initial'
# print gp.gradient(theta)
# print forward(gp.negLogLike,theta)
# print central(gp.negLogLike,theta)


# print 'gradient at optimal'
# print gp.gradient(out['x'])
# print forward(gp.negLogLike,out['x'])
# print central(gp.negLogLike,out['x'])

# print gp.hessian(out['x'])

from pygotools.convex import ip, ipPDC, sqp

xhat, output = sqp(gp.negLogLike,
                   gp.gradient,# gp.hessian,
                   x0=theta,
                   method='trust',
                   maxiter=100,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=3, full_output=True)

print output

xhat, output = ip(gp.negLogLike,
                   gp.gradient,#gp.hessian,
                   method='pd',
                   x0=theta,
                   maxiter=100,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=3, full_output=True)

print output
print output['g'].dot(output['g'])
print "Norm of gradient = "+str(numpy.linalg.norm(output['g']))
print xhat
 
# xhat, output = ipPDC(gp.negLogLike,
#                      gp.gradient,
#                      x0=theta,
#                      lb=boxArray[:,0], ub=boxArray[:,1],
#                      G=None, h=None,
#                      A=None, b=None,
#                      maxiter=1,
#                      disp=5, full_output=True)

# print output
# print xhat

# xhat, output = ip(gp.negLogLike,
#                    gp.gradient,
#                    gp.hessian,
#                    method='pdc',
#                    x0=theta,
#                    maxiter=20,
#                    lb=boxArray[:,0], ub=boxArray[:,1],
#                    disp=5, full_output=True)
# 
# print output
# print xhat


# xhat, output = ip(gp.negLogLike,
#                    gp.gradient,
#                    method='pd',
#                    x0=theta,
#                    maxiter=3,
#                    disp=5, full_output=True)
