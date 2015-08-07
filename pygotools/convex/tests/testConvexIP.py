from unittest import TestCase

from scipy.optimize import rosen, rosen_der, rosen_hess
from pygotools.convex import ip
from pygotools.optutils.consMani import addLBUBToInequality

import numpy

theta = numpy.array([1.3, 0.7, 0.8, 1.9, 1.2])
target = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0])

boxBounds = [(-2.0,2.0) for i in range(len(theta))]

box = numpy.array(boxBounds)
lb = box[:,0]
ub = box[:,1]

A = numpy.ones((1,len(theta)))
b = numpy.ones(1) * len(theta)

G, h = addLBUBToInequality(lb=lb, ub=ub)

ipMethods = ['bar','pdc','pd']
EPSILON = 1e-2

class TestConvexIP(TestCase):

    def test_IP_no_constraints(self):
        
        for m in ipMethods:
            xhat, output = ip(rosen,
                              rosen_der,
                              x0=theta,
                              maxiter=100,
                              method=m,
                              disp=None, full_output=True)

            if numpy.any(abs(target-xhat)>=EPSILON):
                raise Exception("Failed testing for method "+m)
            
    def test_IP_lb_ub(self):
        
        for m in ipMethods:
            xhat, output = ip(rosen,
                              rosen_der,
                              lb=lb,ub=ub,
                              x0=theta,
                              maxiter=100,
                              method=m,
                              disp=None, full_output=True)

            if numpy.any(abs(target-xhat)>=EPSILON):
                raise Exception("Failed testing for method "+m)
    
    def test_IP_Linear_Inequality(self):
        
        for m in ipMethods:
            xhat, output = ip(rosen,
                              rosen_der,
                              G=G,h=h,
                              x0=theta,
                              maxiter=100,
                              method=m,
                              disp=None, full_output=True)

            if numpy.any(abs(target-xhat)>=EPSILON):
                raise Exception("Failed testing for method "+m)
            
    def test_IP_Linear_Equality_and_Inequality(self):
        
        for m in ipMethods:
            xhat, output = ip(rosen,
                              rosen_der,
                              G=G,h=h,
                              A=A,b=b,
                              x0=theta,
                              maxiter=100,
                              method=m,
                              disp=None, full_output=True)

            if numpy.any(abs(target-xhat)>=EPSILON):
                raise Exception("Failed testing for method "+m)
        