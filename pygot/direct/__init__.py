''' direct

.. moduleauthor:: Edwin Tye <Edwin.Tye@gmail.com>

'''
from __future__ import division, print_function, absolute_import

from .rectOperation import *
from .polyOperation import *
from .directUtil import *
from .directAlg import *

#from .PolygonObj import *
#from .RectangleObj import *

__all__ = [s for s in dir() if not s.startswith('_')]
from numpy.testing import Tester
test = Tester().test

