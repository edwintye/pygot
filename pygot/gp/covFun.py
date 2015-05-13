
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
    def diffPhi(self, phi, D):
        raise NotImplementedError()

    @abstractmethod
    def diff2Phi(self, phi, D):
        raise NotImplementedError()
        
    @abstractmethod
    def diffD(self, phi, D):
        raise NotImplementedError()
    
    @abstractmethod
    def diff2D(self, phi, D):
        raise NotImplementedError()

class exp(baseCov):

    def f(self, phi, D):
        return numpy.exp(-phi*D)

    def diffPhi(self, phi, D):
        return -D*numpy.exp(-phi*D)
    
    def diff2Phi(self, phi, D):
        return (D**2) * numpy.exp(-phi*D)
    
    def diffD(self, phi, D):
        return -phi * numpy.exp(-phi*D)
    
    def diff2D(self, phi, D):
        return phi * phi * numpy.exp(-phi*D)
