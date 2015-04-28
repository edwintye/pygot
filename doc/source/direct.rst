.. _direct:

******
DIRECT
******

DIviding RECTangle is a well known method in global optimization.  We have implemented here with a few tweaks and ideas of our own

Setup
=====

First, we are going to load the required modules and also define our objective function - the Rosenbrock function.  We use a set of relatively conservative bounds :math:`x \in [-2,2]^{2}`

.. ipython::

   In [1]: from pygot.direct import directAlg, optimTestFun, plotDirectBox, IdConditionType

   In [1]: import numpy

   In [1]: import matplotlib.pyplot as plt

   In [1]: boundSize = 2

   In [1]: lb = -numpy.ones(2) * boundSize

   In [1]: ub = numpy.ones(2) * boundSize

   In [1]: func = optimTestFun.rosen

Original form
=============

In the seminal paper by Jones et al. it uses an :math:`\varepsilon` condition to determine the dividing boxes.  We have to explicitly tell it to use the this condition via :func:`IdConditionType`, which is Soft in this case

.. ipython::

    In [1]: rectListOptim,output = directAlg.directOptim(func,lb,ub,
       ...:                                              iteration=50,
       ...:                                              numBox=1000,
       ...:                                              conditionType = IdConditionType.Soft,
       ...:                                              targetMin=0,
       ...:                                              scaleOutput=False,
       ...:                                              full_output=True)

    @savefig directSoft.png
    In [2]: plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

The plots show how the distribution of boxes.  When the condition is not set, by default, it progresses using the Pareto front condition as seen below

.. ipython::

   In [1]: from pygot.direct import directUtil

   In [2]: rectListOptim,output = directAlg.directOptim(func,lb,ub,
      ...:                                              iteration=50,
      ...:                                              numBox=1000,
      ...:                                              targetMin=0,
      ...:                                              scaleOutput=False,
      ...:                                              full_output=True)

   In [3]: potentialIndex = directUtil.identifyPotentialOptimalObjectPareto(rectListOptim)

   In [3]: print potentialIndex

   @savefig directParetoFront.png
   In [4]: directUtil.plotParetoFrontRect(rectListOptim,potentialIndex)

In this particular case, the Pareto front condition performs better.  This though, is not a guarantee and using the Pareot front usually result in a lot more function evaluations

.. ipython::
   
   @savefig directPareto.png
   In [3]: plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

   In [4]: plt.close()
