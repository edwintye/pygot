
__all__ = [
    'DFP',
    'BFGS',
    'SR1',
    'SR1Alpha'
    ]

import numpy

def DFP(H, diffG, deltaX):
    H -= H.dot(numpy.outer(diffG,diffG)).dot(H) / (diffG.dot(H).dot(diffG))
    H += numpy.outer(deltaX,deltaX) / diffG.dot(deltaX)
    return H

def BFGS(H, diffG, deltaX):
    p = len(diffG)
    a = diffG.dot(deltaX)
    A = numpy.eye(p) - numpy.outer(deltaX,diffG) / a
    H += A.T.dot(H).dot(A)
    H += numpy.outer(deltaX,deltaX) / a
    return H

def SR1(H, diffG, deltaX):
    a = deltaX - H.dot(diffG)
    #print numpy.outer(a,a) / a.dot(diffG)
    H += numpy.outer(a,a) / a.dot(diffG)
    return H

def SR1Alpha(H, diffG, deltaX):
    # TODO: check if this is a true under estimator
    p = len(diffG)
    a = deltaX - H.dot(diffG)
    #print numpy.outer(a,a) / a.dot(diffG)
    H += numpy.outer(a,a) / a.dot(diffG)
    e = numpy.linalg.eig(H)[0]
    H += numpy.eye(p) * abs(min(e))
    return H
