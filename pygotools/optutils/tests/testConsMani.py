from unittest import TestCase

import numpy

from pygotools.optutils.consMani import *

lb = -numpy.ones(2)
ub = numpy.ones(2)
    
box = numpy.array([ (-1.0,1.0), (-1.0,1.0) ])
G = numpy.array([[1.0,1.0]])
h = numpy.array([0.0])
        
class TestConsMani(TestCase):
    
    def testLBUBToBox(self):        
        A = lbubToBox(lb,ub)
        
        if numpy.any(abs(box-A)>=1e-8):
            raise Exception("Failed test")
    
    def testAddBoxToInequality(self):
        A, b = addBoxToInequality(box,G,h)
        targetA = numpy.array([
                              [1.0,1.0],
                              [1.0,0.0],
                              [0.0,1.0],
                              [-1.0,0.0],
                              [0.0,-1.0]
                               ])
        targetB = numpy.array([0.0, 1.0, 1.0, 1.0, 1.0])
        
        if numpy.any(abs(A-targetA)>=1e-8):
            raise Exception("Failed test")
        if numpy.any(abs(b-targetB)>=1e-8):
            raise Exception("Failed test")