
__all__ = ['GP']

import numpy
import scipy.spatial.distance as ssd
import scipy.stats
import pygotools.responseSurface

class InputError(Exception):
    '''
    Read the function name
    '''
    pass

class GP(object):

    def __init__(self, y, x, s, nugget=True, rbfFun=None):

        self._n = len(y)
        self._y = y

        if x is None:
            self._x1 = numpy.ones((self._n,1))
            self._numBeta = 1
        else:
            self._x1 = numpy.append(numpy.ones((self._n,1)),x,axis=1)
            n, self._numBeta = self._x1.shape

            if n!=self._n:
                raise InputError("Number of rows in x and y must be equal")

        self._s = s
        n1 , self._numSites = s.shape
        if n1 != self._n:
            raise InputError("Number of rows in s and y must be equal")

        self._D = ssd.squareform(ssd.pdist(s))

        self._phi = 1
        self._beta = numpy.ones(self._numBeta)
        self._sigma2 = 1
        if nugget:
            self._tau2 = 1
        else:
            self._tau2 = None

        if rbfFun is None:
            self._rbfFun = pygotools.responseSurface.covFun.exp()

    def negLogLike(self,theta):
        beta, phi, sigma2, tau2 = _unrollParam(theta, self._numBeta)
        G = sigma2*self._rbfFun.f(phi,self._D);
        if tau2 is None:
            H = G
        else:
            H = tau2*numpy.eye(self._n) + G;

        try:
            ll = -scipy.stats.multivariate_normal.logpdf(self._y,self._x1.dot(beta),H,False)
            self._beta = beta
            self._phi = phi
            self._sigma2 = sigma2
            self._tau2 = tau2
            return ll
        except Exception:
            return numpy.inf

    def gradient(self, theta):
        beta, phi, sigma2, tau2 = _unrollParam(theta, self._numBeta)

        G, H, quadForm, invH, DG, dPhi, dSigma, dTau = _getParamDerivativeInfo(self._y, self._x1,
                                                                         beta, phi, sigma2, tau2,
                                                                         self._D, self._n,
                                                                         self._rbfFun)
        # finished the unrolling of parameters and pre calculations
        g = numpy.zeros(len(theta))

        g[0:self._numBeta] = self._x1.T.dot(quadForm);
        g[self._numBeta] = (quadForm.T.dot(DG.dot(quadForm)) - numpy.trace(dPhi))/2; 
        g[self._numBeta+1] = (quadForm.T.dot((G/sigma2).dot(quadForm)) - numpy.trace(dSigma))/2;
        if tau2 is not None:
            g[self._numBeta+2] = (quadForm.T.dot(quadForm) - numpy.trace(dTau))/2;
        return -g

    def hessian(self, theta):
        # TODO: check the correctness of this
        beta, phi, sigma2, tau2 = _unrollParam(theta, self._numBeta)

        G, H, quadForm, invH, DG, dPhi, dSigma, dTau = _getParamDerivativeInfo(self._y, self._x1,
                                                                         beta, phi, sigma2, tau2,
                                                                         self._D, self._n,
                                                                         self._rbfFun)

        if tau2 is None:
            A = numpy.zeros((2,2))
        else:
            A = numpy.zeros((3,3))

        # phi,phi
        A[0,0] = quadForm.T.dot(2*DG.dot(dPhi)).dot(quadForm) + numpy.trace(-dPhi.dot(dPhi));
        # phi, sigma2
        A[0,1] = quadForm.T.dot(DG.dot(dSigma) + dPhi).dot(quadForm) + numpy.trace(-dPhi.dot(dSigma));

        # sigma2, sigma2
        A[1,1] = quadForm.T.dot(2*(G/sigma2).dot(dSigma)).dot(quadForm) + numpy.trace(-dSigma.dot(dSigma));
        
        if tau2 is not None:
            # phi, tau2
            A[0,2] = quadForm.T.dot(DG.dot(dTau) + dPhi).dot(quadForm) + numpy.trace(-dPhi.dot(dTau));
            # sigma2, tau2
            A[1,2] = quadForm.T.dot((G/sigma2).dot(dTau) + dSigma).dot(quadForm) + numpy.trace(-dSigma.dot(dTau));
            # tau2, tau2
            A[2,2] = quadForm.T.dot(2*dTau).dot(quadForm) + numpy.trace(-dTau.dot(dTau));

        A = A.T + A - numpy.diag(numpy.diag(A));
    
        if tau2 is None:
            B = numpy.zeros((self._numBeta,2));
        else:
            B = numpy.zeros((self._numBeta,3));

        B[:,0] = self._x1.T.dot(dPhi.dot(quadForm));
        B[:,1] = self._x1.T.dot(dSigma.dot(quadForm));
        if tau2 is not None:
            B[:,2] = self._x1.T.dot(dTau.dot(quadForm));
    
        #print B
        #print (x1.T.dot(invH)).dot(x1)

        XTX = numpy.zeros((self._numBeta,self._numBeta))
        XTX[:] = (self._x1.T.dot(invH)).dot(self._x1)

        return numpy.bmat([[ XTX, B],[B.T, 0.5*A]]);

    def getInitialGuess(self):
        if self._tau2 is None:
            theta = numpy.ones(self._numBeta+2)
        else:
            theta = numpy.ones(self._numBeta+3)
        
        theta[0:self._numBeta] = numpy.linalg.lstsq(self._x1,self._y)[0]
        return theta

    def predict(self, s0, x0=None, full_output=False):
        x, s, D, A, Ac, x0, s0 = _predictSetup(self._x1, self._s,
                                       x0, s0)

        mu = x.dot(self._beta)
        H = _getCovariance(self._phi, self._sigma2, self._tau2,
                           D, self._rbfFun)

        X = numpy.append(self._y, numpy.zeros(len(s)))

        muA, sigma2A, lambdaA = _conditionalMVN(X, mu, H, A, Ac)

        if full_output:
            return muA, sigma2A, lambdaA
        else:
            return muA

    def predictGradient(self, s0, x0=None):
        if len(s0) > 1:
            if len(s0) != self._numSites:
                raise InputError("Gradient only available for"
                                 +" a single observation")

        x, s, D, A, Ac, x0, s0 = _predictSetup(self._x1, self._s,
                                               x0, s0)

        mu = x.dot(self._beta)
        H = _getCovariance(self._phi, self._sigma2, self._tau2,
                           D, self._rbfFun)

        X = numpy.append(self._y, numpy.zeros(len(x0)))

        muA, sigma2A, lambdaA = _conditionalMVN(X, mu, H, A, Ac)
        lambdaA = lambdaA.reshape(len(lambdaA),1)

        r = s0 - self._s
        F = numpy.array(range(len(r)))
        normR = numpy.linalg.norm(r,axis=1)
        setIndex = F[normR!=0]
        
        gamma = H[:-1,-1]
        normalize = numpy.zeros(len(normR))
        normalize[setIndex] = 1/normR[setIndex]
        gammaNormalize = (gamma * normalize).reshape(len(gamma),1);

        gTemp = r[setIndex,:] * gammaNormalize[setIndex]
        grad = (-self._phi * gTemp * lambdaA[setIndex]).sum(axis=0)
        if len(self._beta) == self._numSites:
            grad += self._beta

        return grad

    def predictHessian(self, s0, x0=None):
        if len(s0) > 1:
            if len(s0) != self._numSites:
                raise InputError("Gradient only available for"
                                 +" a single observation")

        x, s, D, A, Ac, x0, s0 = _predictSetup(self._x1, self._s,
                                               x0, s0)

        mu = x.dot(self._beta)
        H = _getCovariance(self._phi, self._sigma2, self._tau2,
                           D, self._rbfFun)

        X = numpy.append(self._y, numpy.zeros(len(x0)))

        muA, sigma2A, lambdaA = _conditionalMVN(X, mu, H, A, Ac)
        lambdaA = lambdaA.reshape(len(lambdaA),1)

        r = s0 - self._s
        F = numpy.array(range(len(r)))
        normR = numpy.linalg.norm(r,axis=1)
        setIndex = F[normR!=0]
        
        D12 = D[:-1,-1].reshape(self._n,1)
        gamma = H[:-1,-1].reshape(self._n,1)
        normalize = numpy.zeros(len(normR))
        normalize[setIndex] = (1/normR[setIndex])
        normalize = normalize.reshape(len(normalize),1)
        gammaNormalize = gamma * normalize
        #gammaNormalize = (gamma * normalize).reshape(len(gamma),1);

        hessian = numpy.eye(self._numSites) * sum(self._rbfFun.diffD(self._phi,D12[setIndex]) * normalize[setIndex] * lambdaA[setIndex]);
        
        #print -self._phi*gamma[setIndex] - self._rbfFun.diffD(self._phi,D12[setIndex])

        # print r[setIndex,:].shape
        # print (numpy.sqrt(self._rbfFun.diff2D(self._phi,0) * gamma[setIndex])).shape
        # print (self._rbfFun.diffD(self._phi,0) * gammaNormalize[setIndex] * normalize[setIndex]).shape
        # print gammaNormalize[setIndex].shape
        # print normalize[setIndex].shape

        R = numpy.sqrt(self._rbfFun.diff2D(self._phi,D12[setIndex]) - self._rbfFun.diffD(self._phi,D12[setIndex]) * normalize[setIndex]) * r[setIndex,:] * normalize[setIndex]

        # print hessian
        # print R.shape
        # print lambdaA[setIndex].shape
        # print numpy.diag(lambdaA[setIndex].ravel()).shape

        print hessian
        hessian += R.T.dot(numpy.diag(lambdaA[setIndex].ravel())).dot(R);
        print hessian

        # print len(F)
        # print len(setIndex)
        # print type(F)
        # print type(setIndex)
        Ac = numpy.array(list(set(F) - set(setIndex)))
        #print Ac
        #Ac = setdiff(F,setIndex);
        #print lambdaA[Ac]
        if len(Ac) > 0:
            hessian += sum(self._rbfFun.diff2D(self._phi,0) * lambdaA[Ac] ) * numpy.eye(self._numSites);

        return hessian

def _unrollParam(theta, numBeta):

    beta = theta[0:numBeta]
    phi = theta[numBeta]
    sigma2 = theta[numBeta+1]

    if len(theta)==numBeta+1:
        tau2 = None
    else:
        tau2 = theta[numBeta+2]

    # print theta
    # print tau2
        
    return beta, phi, sigma2, tau2

def _predictSetup(xOrig, sOrig, xPred, sPred):
    n, p = xOrig.shape
    nS, pS = sOrig.shape

    if type(sPred) is not numpy.ndarray:
        # do not need further check because we
        # would not use a GP for points on a line
        sPred = numpy.array(sPred)

    if len(sPred) == sPred.size:
        sPred = sPred.reshape(1,len(sPred))

    if xPred is None:
        xPred = numpy.ones((len(sPred),1))
        if p==1:
            pass
        elif p==pS+1:
            xPred = numpy.append(xPred,sPred.copy(),axis=1)
        else:
            raise Exception("Input dimensions not correct")
    else:
        if type(xPred) is not numpy.ndarray:
            if type(xPred) in (list, tuple):
                xPred = numpy.array(xPred)
            else:
                xPred = numpy.array([xPred])
        if p==1:
            xPred = xPred.reshape(len(xPred),p)
        else:
            n1, p1 = xPred.shape
            if p1!=p:
                # TODO: extend linear tail
                raise InputError("Expecting " +str(p)+
                                 " number of parameters")
            
    #print xOrig.shape
    #print xPred.shape

    x = numpy.append(xOrig, xPred, axis=0)
    s = numpy.append(sOrig, sPred, axis=0)
    D = ssd.squareform(ssd.pdist(s))

    Ac = range(n)
    A = range(n, len(x))
    return x, s, D, A, Ac, xPred, sPred

def _getCovariance(phi, sigma2, tau2, D, rbfFun):
    G = sigma2 * rbfFun.f(phi, D);
    if tau2 is None:
        return G
    else:
        return tau2 * numpy.eye(len(G)) + G

def _getParamDerivativeInfo(y, x1, beta, phi, sigma2, tau2, D, n, rbfFun):
    G = sigma2*rbfFun.f(phi,D);
    if tau2 is None:
        H = G
    else:
        H = tau2*numpy.eye(n) + G

    W = (y-x1.dot(beta));

    invH = numpy.linalg.solve(H,numpy.eye(n))
    quadForm = invH.dot(W);

    DG = -D*G
    dPhi = invH.dot(DG);
    dSigma = invH.dot(G/sigma2); 
    dTau = invH;

    return G, H, quadForm, invH, DG, dPhi, dSigma, dTau

def _conditionalMVN(X, mu, sigma2, A, Ac):

    lambdaA = numpy.linalg.solve(sigma2[Ac][:,Ac],X[Ac]-mu[Ac])

    # print lambdaA.shape
    # print sigma2.shape
    # print A
    # print len(A)
    # print len(Ac)
    # print sigma2[A][:,Ac].dot(lambdaA)

    muA = mu[A] + sigma2[A][:,Ac].dot(lambdaA)
    sigma2A = sigma2[A][:,A] - sigma2[A][:,Ac].dot(numpy.linalg.solve(sigma2[Ac][:,Ac],sigma2[Ac][:,A]))

    return muA, sigma2A, lambdaA
