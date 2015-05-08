
__all__ = [
    'exp'
]

from abc import ABCMeta, abstractmethod
import numpy


class baseCov:

    __metaclass__ = ABCMeta

    @abstractmethod
    def f(self, phi, D):
        raise NotImplementedError()

    @abstractmethod
    def g(self, phi, D):
        raise NotImplementedError()

class exp(baseCov):

    def f(self, phi, D):
        return numpy.exp(-phi*D)

    def diffPhi(self, phi, D):
        return -D*numpy.exp(-phi*D)
    
    def diffD(self, phi, D):
        return -phi
