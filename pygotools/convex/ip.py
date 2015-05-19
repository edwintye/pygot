
__all__ = [
    'ip'
    ]


from .ipBar import ipBar
from .ipPD import ipPD
from .ipPDC import ipPDC


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
        return ipPD(func, grad, hessian, x0,
                     lb, ub,
                     G, h,
                     A, b,
                     maxiter,
                     disp, full_output)
    elif method=='pdc':
        return ipPDC(func, grad, hessian, x0,
                     lb, ub,
                     G, h,
                     A, b,
                     maxiter,
                     disp, full_output)
