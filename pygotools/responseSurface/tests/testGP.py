from unittest import TestCase

import numpy
import scipy.spatial, scipy.optimize
from pygotools.responseSurface import GP, exp

n = 10
x = numpy.array([0.056, 6.257, 1.204, 4.346, 4.902, 9.8, 7.624, 4.258, 2.835, 5.497]).reshape(n,1)
y = numpy.array([ 5.972,  3.391,  4.891,  5.352,  4.423,  3.057,  4.553,  4.365, 7.374, 6.554]).reshape(n,1)
s = numpy.append(x,y,axis=1)
D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(s))
        
class TestResponseSurface(TestCase):
    
    def gpNugget(self):
        rbfFun = exp()
        rbfFun.f(1e-8,D)

        gp = GP(y.ravel(),None,s,nugget=True)
        theta = gp.getInitialGuess()
    
        box = [(1e-8,10)] * len(theta)
  
        out = scipy.optimize.minimize(fun=gp.negLogLike,
                                          x0=theta,
                                          bounds=box)
        
    def gpNoNugget(self):
        rbfFun = exp()
        rbfFun.f(1e-8,D)

        gp = GP(y.ravel(),None,s,nugget=False)
        theta = gp.getInitialGuess()
    
        box = [(1e-8,10)] * len(theta)
  
        out = scipy.optimize.minimize(fun=gp.negLogLike,
                                          x0=theta,
                                          bounds=box)
    
    
    