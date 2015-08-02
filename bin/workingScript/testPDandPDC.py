import scipy.spatial, scipy.optimize
import numpy

n = 10
x = numpy.array([0.056, 6.257, 1.204, 4.346, 4.902, 9.8, 7.624, 4.258, 2.835, 5.497]).reshape(n,1)
y = numpy.array([ 5.972,  3.391,  4.891,  5.352,  4.423,  3.057,  4.553,  4.365, 7.374, 6.554]).reshape(n,1)
s = numpy.append(x,y,axis=1)
D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(s))



from pygotools.responseSurface import GP, exp

rbfFun = exp()
rbfFun.f(1e-8,D)

gp = GP(y.ravel(),None,s,nugget=False)
theta = gp.getInitialGuess()

box = [(1e-8,10)] * len(theta)
boxArray = numpy.array(box)

out = scipy.optimize.minimize(fun=gp.negLogLike,
                              jac=gp.gradient,
                              x0=theta,
                              bounds=box)

print out


from pygotools.convex import ip, ipPDC

xhat, output = ip(gp.negLogLike,
                   gp.gradient,#gp.hessian,
                   method='pd',
                   x0=theta,
                   maxiter=40,
                   lb=boxArray[:,0], ub=boxArray[:,1],
                   disp=3, full_output=True)

print output
print xhat
 
xhat, output = ipPDC(gp.negLogLike,
                     gp.gradient,
                     x0=theta,
                     lb=boxArray[:,0], ub=boxArray[:,1],
                     G=None, h=None,
                     A=None, b=None,
                     maxiter=100,
                     disp=5, full_output=True)

print output
print xhat

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
