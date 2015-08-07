
__all__ = [
    'DFP',
    'BFGS',
    'SR1'
    ]

import numpy
import scipy.linalg

def DFP(H, diffG, deltaX):
    if numpy.all(diffG==0) or numpy.all(deltaX==0):
        pass
    else:
        p = len(diffG)
        a = diffG.dot(deltaX)
        A = numpy.eye(p) - numpy.outer(diffG, deltaX) / a
        #A2 = numpy.eye(p) - numpy.outer(deltaX, diffG) / a
        # print "A1"
        # print A1
        # print "A2"
        # print A2

        #H += A1.dot(H).dot(A2)
        H += A.dot(H).dot(A.T)
        H += numpy.outer(diffG, diffG) / a
        # print "H"
        # print numpy.linalg.eig(H)[0]
    return H

def BFGS(H, diffG, deltaX):
    if numpy.all(diffG==0) or numpy.all(deltaX==0):
        return H
    else:
        rho = 1/diffG.dot(deltaX)
        if numpy.isinf(rho):
            rho = 10000.0
            
        A1 = numpy.outer(diffG,diffG) * rho
        a = H.dot(deltaX).T
        # AA = numpy.outer(a,a)
        # print a
        # print deltaX
        # AA2 = (deltaX.dot(a))
        A2 = numpy.outer(a,a) / (deltaX.dot(a))
        
        # print "diff G"
        # print diffG
        # print "delta X"
        # print deltaX
        # print "bot"
        # print diffG.dot(deltaX)
        # print "A1"
        # print A1
        # print "A2"
        # print A2
        H1 = H + (A1-A2)
        if numpy.any(scipy.linalg.eig(H1)[0]<=0):
            # print "Positive descent direction"
            # print diffG
            # print deltaX
            return H1
        else:
            return H1

def SR1(H, diffG, deltaX):
    a = diffG - H.dot(deltaX)
    LHS = abs(deltaX.dot(a))
    RHS = 1e-8 * scipy.linalg.norm(deltaX) * scipy.linalg.norm(a) 
    if LHS < RHS:
        pass
    else:
        #print numpy.outer(a,a) / a.dot(diffG)
        H += numpy.outer(a,a) / a.dot(deltaX)
    return H

