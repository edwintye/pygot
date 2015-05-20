
__all__ = [
    'ip'
    ]

from .ipBar import ipBar
from .ipPD2 import ipPD2

def ip(func, grad, hessian=None, x0=None,
       lb=None, ub=None,
       G=None, h=None,
       A=None, b=None,
       maxiter=100,
       method='bar',
       disp=0, full_output=False):
    '''
    An interface to the methods provides in this subpackage.
    '''
    if method=='bar':
        return ipBar(func, grad, hessian, x0,
                     lb, ub,
                     G, h,
                     A, b,
                     maxiter,
                     disp, full_output)
    elif method=='pd':
        return ipPD2(func, grad, hessian, x0,
                     lb, ub,
                     G, h,
                     A, b,
                     maxiter,
                     method='pd',
                     disp=disp, full_output=full_output)
    elif method=='pdc':
        return ipPD2(func, grad, hessian, x0,
                     lb, ub,
                     G, h,
                     A, b,
                     maxiter,
                     method='pdc',
                     disp=disp, full_output=full_output)
