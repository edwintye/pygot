
__all__ = [
    'DFP',
    'BFGS',
    'SR1',
    'SR1Alpha'
    ]

import numpy

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
        pass
    else:
        A1 = numpy.outer(diffG,diffG) / diffG.dot(deltaX)
        a = H.dot(deltaX)
        A2 = numpy.outer(a,a) / (deltaX.T.dot(a))
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
        H += A1 - A2
    return H

def SR1(H, diffG, deltaX):
    if numpy.all(diffG==0) or numpy.all(deltaX==0):
        pass
    else:
        a = diffG - H.dot(deltaX)
        #print numpy.outer(a,a) / a.dot(diffG)
        H += numpy.outer(a,a) / a.dot(deltaX)
    return H

def SR1Alpha(H, diffG, deltaX):
    H = SR1(H,diffG,deltaX)
    e = numpy.linalg.eig(H)[0]
    H += numpy.eye(len(H)) * abs(min(e))
    return H
