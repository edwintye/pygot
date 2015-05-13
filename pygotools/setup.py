# no idea what it is doing here
# copied most of it from scipy
from __future__ import division, print_function, absolute_import

import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pygotools',parent_package,top_path)
    config.add_subpackage('direct')
    config.add_subpackage('gradient')
    config.add_subpackage('convex')
    config.add_subpackage('optutils')

    if sys.version_info[0] < 3:
        config.add_subpackage('weave')
        config.add_subpackage('_build_utils')
        config.add_subpackage('_lib')
        config.make_config_py()
        return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
