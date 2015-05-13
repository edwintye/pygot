''' gp

.. moduleauthor:: Edwin Tye <Edwin.Tye@gmail.com>

'''

from __future__ import division, print_function, absolute_import

from .gp import *
from .covFun import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test
