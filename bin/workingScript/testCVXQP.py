from cvxopt import matrix, solvers
### Simply QP
n = 4
S = matrix([[ 4e-2,  6e-3, -4e-3,    0.0 ],
            [ 6e-3,  1e-2,  0.0,     0.0 ],
            [-4e-3,  0.0,   2.5e-3,  0.0 ],
            [ 0.0,   0.0,   0.0,     0.0 ]])
pbar = matrix([.12, .10, .07, .03])
G = matrix(0.0, (n,n))
G[::n+1] = -1.0
h = matrix(0.0, (n,1))
A = matrix(1.0, (1,n))
b = matrix(1.0)

# Compute trade-off.
N = 100
mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
sol = solvers.qp(mus[0]*S, -pbar, G, h, A, b)
sol = solvers.coneqp(mus[0]*S, -pbar, G, h, [], A, b)

portfolios = [ solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]



## From SCOP to Cone LP

c = matrix([-2., 1., 5.])
G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
sol = solvers.socp(c, Gq = G, hq = h)
sol['status']


c = matrix([-2., 1., 5.])
G = matrix( [[12., 13., 12., 3., 3., -1., 1.],
             [6., -3., -12., -6., -6., -9., 19.],
             [-5., -5., 6., 10., -2., -2., -3.]])
h = matrix( [-12., -3., -2., 27., 0., 3., -42.] )
dims = {'l': 0, 'q': [3,4], 's': []}
sol2 = solvers.conelp(c, G, h, dims)



## least sqaures problem, as a QP

A = matrix([ [ .3, -.4,  -.2,  -.4,  1.3 ],
                 [ .6, 1.2, -1.7,   .3,  -.3 ],
                 [-.3,  .0,   .6, -1.2, -2.0 ] ])
b = matrix([ 1.5, .0, -1.2, -.7, .0])
m, n = A.size
I = matrix(0.0, (n,n))
I[::n+1] = 1.0
G = matrix([-I, matrix(0.0, (1,n)), I])
h = matrix(n*[0.0] + [1.0] + n*[0.0])
dimsQP = {'l': n, 'q': [n+1], 's': []}
solQP = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)

# in general nonlinear epigraph form

from cvxopt import blas
solvers.options['maxiters'] = 300

def F1(x=None, z=None):
    if x is None: return 0, matrix(0.0,(n,1))
    f = blas.nrm2(b - A*x)**2
    # df = (A.T * -b).T
    df = (2.0 * A.T * (A*x - b)).T
    # df = (2 * A.T * A * x - 2 * A.T * b).T
    if z is None: return f, df
    H = z[0] * 2.0 * A.T * A
    return f, df, H
        
solCP = solvers.cp(F1, G, h, dims)

scipy.linalg.lstsq(A,b)

A.T * (A*matrix(y) - b)

y = x
for i in range(5):
    deltaY = scipy.linalg.lstsq(A.T * A,(A.T * (A*matrix(y) - b)))[0]
    y -= deltaY
    print y

scipy.linalg.lstsq(A.T * A, A.T*b)

def F2(x=None, z=None):
    if x is None: return 0, matrix(0.0,(n,1))
    f = blas.nrm2(b - A*x)**2
    df = 2.0 * (A.T * -b).T
    if z is None: return f, df
    H = z[0] * 2.0 * A.T * A
    return f, df, H

solCP2 = solvers.cp(F2, G, h, dims)
print solCP1['x']
print solCP2['x']

# now convert it to a second order cone

c = matrix([1.0] + [0.0]*n)

hTemp = matrix([0.0,b])
GTemp = matrix(numpy.array(scipy.sparse.bmat([
    [[-1.0],None],
    [None,A]
    ]).todense()))

GTemp1 = matrix(numpy.append(numpy.append(numpy.array([0]*G.size[0]).reshape(G.size[0],1),numpy.array(G),axis=1),
                             numpy.array(GTemp),
                             axis=0))

hTemp1 = matrix(numpy.append(h,hTemp))

dims1 = {'l': n, 'q': [n+1,m+1], 's': []}

solQP = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)

solSOCP = solvers.conelp(c, GTemp1, hTemp1, dims1)

print solSOCP['x']


print solQP['x']
print solCP['x']
print solSOCP['x']

