from unittest import TestCase

import numpy

from pyOptimUtil.direct import RectangleObj, optimTestFun,rectOperation

class TestStandardObjFunc(TestCase):
    def test_Rosen(self):
        lb = numpy.array([-2.,-2.],float)
        ub = numpy.array([2.,2.],float)

        print("Now we start DIRECT, with scaled output")

        rectListOptim,output = rectOperation.directOptim(optimTestFun.rosen,lb,ub,
                                                 iteration=20,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

        rectOperation.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

        # rectOperation.plotParetoFront(rectListOptim)
        
    def test_GP(self):
        lb = numpy.array([-2.,-2.],float)
        ub = numpy.array([2.,2.],float)

        print("Now we start DIRECT, with scaled output for the GP function")

        rectListOptim,output = rectOperation.directOptim(optimTestFun.gp,lb,ub,
                                                 iteration=10,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

        rectOperation.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

        # rectOperation.plotParetoFront(rectListOptim)

    def test_Himmelblau(self):
        lb = numpy.array([-5.,-5.],float)
        ub = numpy.array([5.,5.],float)

        print("Now we start DIRECT, using the Himmelblau test function")
        print("This is a multimodal function")

        rectListOptim,output = rectOperation.directOptim(optimTestFun.himmelblau,lb,ub,
                                                 iteration=20,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

        rectOperation.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

        # rectOperation.plotParetoFront(rectListOptim)
        
    def test_mccormick(self):
        lb = numpy.array([-5.,-5.],float)
        ub = numpy.array([5.,5.],float)

        print("Now we start DIRECT, using the Mccormick test function")

        rectListOptim,output = rectOperation.directOptim(optimTestFun.himmelblau,lb,ub,
                                                 iteration=20,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

        rectOperation.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

        # rectOperation.plotParetoFront(rectListOptim)
        
#####