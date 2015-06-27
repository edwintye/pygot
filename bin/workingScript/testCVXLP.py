import numpy
import scipy.io

a = scipy.io.loadmat("/home/edwin/workspace/testLP/AFIRO.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/25FV47.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/ADLITTLE.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/BLEND.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/BNL2.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/DFL001.mat")
a = scipy.io.loadmat("/home/edwin/workspace/testLP/FIT1D.mat")


A = numpy.array(a['A'].todense()).astype(float)
b = numpy.array(a['b']).astype(float).ravel()
c = numpy.array(a['c']).astype(float).ravel()

lb = a['lbounds'].astype(float).ravel()
ub = a['ubounds'].astype(float).ravel()
ub[ub==1e32]=1e5



numParam = len(c)
G = numpy.append(numpy.eye(numParam), -numpy.eye(numParam), axis=0)
h = numpy.append(ub, -lb, axis=0)

numpy.linalg.matrix_rank(G)
numpy.linalg.matrix_rank(A)

q,r = numpy.linalg.qr(A)

index =  abs(numpy.diag(r))<=1e-15

q[index][:,index].shape
r[index].shape

newA = q[index][:,index].dot(r[index])

import scipy.sparse
sparseG = scipy.sparse.csc_matrix(G)

u,s,vt = scipy.sparse.linalg.svds(sparseG)

from cvxopt import sparse, spmatrix, matrix, solvers

sparse(matrix(G))

sol = solvers.lp(matrix(c), sparse(matrix(G)), matrix(h), sparse(matrix(A)), matrix(b))

sol1 = solvers.lp(matrix(c), sparse(matrix(G)), matrix(h), sparse(matrix(A)), matrix(b),'glpk')

sol1 = solvers.lp(matrix(c), sparse(matrix(G)), matrix(h), sparse(matrix(newA)), matrix(b),'glpk')

print(sol['x'])


c = matrix([-4., -5.])
G = matrix([[2., 1., -1., 0.], [1., 2., 0., -1.]])
h = matrix([3., 3., 0., 0.])
sol = solvers.lp(c, G, h)
sol = solvers.lp(c, G, h, None, None, 'glpk')
print(sol['x'])

