
# Figures 8.5-7, pages 417-421.
# Centers of polyhedra.

from math import log, pi
from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spdiag, sqrt, mul, cos, sin, log
from cvxopt.modeling import variable, op
solvers.options['show_progress'] = False
try: import pylab
except ImportError: pylab_installed = False
else: pylab_installed = True

# Extreme points (with first one appended at the end)
X = matrix([ 0.55,  0.25, -0.20, -0.25,  0.00,  0.40,  0.55,
             0.00,  0.35,  0.20, -0.10, -0.30, -0.20,  0.00 ], (7,2))
m = X.size[0] - 1

# Inequality description G*x <= h with h = 1
G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
G = (X[:m,:] - X[1:,:]) * matrix([0., -1., 1., 0.], (2,2))
h = (G * X.T)[::m+1]
G = mul(h[:,[0,0]]**-1, G)
h = matrix(1.0, (m,1))

# Chebyshev center
#
# maximizse   R
# subject to  gk'*xc + R*||gk||_2 <= hk,  k=1,...,m
#             R >= 0

R = variable()
xc = variable(2)
op(-R, [ G[k,:]*xc + R*blas.nrm2(G[k,:]) <= h[k] for k in range(m) ] +
    [ R >= 0] ).solve()
R = R.value
xc = xc.value
 
if pylab_installed:
    pylab.figure(1, facecolor='w')

    # polyhedron
    for k in range(m):
        edge = X[[k,k+1],:] + 0.1 * matrix([1., 0., 0., -1.], (2,2)) * \
            (X[2*[k],:] - X[2*[k+1],:])
        pylab.plot(edge[:,0], edge[:,1], 'k')


    # 1000 points on the unit circle
    nopts = 1000
    angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
    circle = matrix(0.0, (2,nopts))
    circle[0,:], circle[1,:] = R*cos(angles), R*sin(angles)
    circle += xc[:,nopts*[0]]

    # plot maximum inscribed disk
    pylab.fill(circle[0,:].T, circle[1,:].T, facecolor = '#F0F0F0')
    pylab.plot([xc[0]], [xc[1]], 'ko')
    pylab.title('Chebyshev center (fig 8.5)')
    pylab.axis('equal')
    pylab.axis('off')


# Maximum volume enclosed ellipsoid center
#
# minimize    -log det B
# subject to  ||B * gk||_2 + gk'*c <= hk,  k=1,...,m
#
# with variables  B and c.
#
# minimize    -log det L
# subject to  ||L' * gk||_2^2 / (hk - gk'*c) <= hk - gk'*c,  k=1,...,m
#
# L lower triangular with positive diagonal and B*B = L*L'.
#
# minimize    -log x[0] - log x[2]
# subject to   g( Dk*x + dk ) <= 0,  k=1,...,m
#
# g(u,t) = u'*u/t - t
# Dk = [ G[k,0]   G[k,1]  0       0        0
#        0        0       G[k,1]  0        0
#        0        0       0      -G[k,0]  -G[k,1] ]
# dk = [0; 0; h[k]]
#
# 5 variables x = (L[0,0], L[1,0], L[1,1], c[0], c[1])

D = [ matrix(0.0, (3,5)) for k in range(m) ]
for k in range(m):
    D[k][ [0, 3, 7, 11, 14] ] = matrix( [G[k,0], G[k,1], G[k,1],
        -G[k,0], -G[k,1]] )
d = [matrix([0.0, 0.0, hk]) for hk in h]

def F(x=None, z=None):
    if x is None:
        return m, matrix([ 1.0, 0.0, 1.0, 0.0, 0.0 ])
    if min(x[0], x[2], min(h-G*x[3:])) <= 0.0:
        return None

    y = [ Dk*x + dk for Dk, dk in zip(D, d) ]

    f = matrix(0.0, (m+1,1))
    f[0] = -log(x[0]) - log(x[2])
    for k in range(m):
        f[k+1] = y[k][:2].T * y[k][:2] / y[k][2] - y[k][2]

    Df = matrix(0.0, (m+1,5))
    Df[0,0], Df[0,2] = -1.0/x[0], -1.0/x[2]

    # gradient of g is ( 2.0*(u/t);  -(u/t)'*(u/t) -1)
    for k in range(m):
        a = y[k][:2] / y[k][2]
        gradg = matrix(0.0, (3,1))
        gradg[:2], gradg[2] = 2.0 * a, -a.T*a - 1
        Df[k+1,:] =  gradg.T * D[k]
    if z is None: return f, Df

    H = matrix(0.0, (5,5))
    H[0,0] = z[0] / x[0]**2
    H[2,2] = z[0] / x[2]**2

    # Hessian of g is (2.0/t) * [ I, -u/t;  -(u/t)',  (u/t)*(u/t)' ]
    for k in range(m):
        a = y[k][:2] / y[k][2]
        hessg = matrix(0.0, (3,3))
        hessg[0,0], hessg[1,1] = 1.0, 1.0
        hessg[:2,2], hessg[2,:2] = -a,  -a.T
        hessg[2, 2] = a.T*a
        H += (z[k] * 2.0 / y[k][2]) *  D[k].T * hessg * D[k]

    return f, Df, H

sol = solvers.cp(F)
L = matrix([sol['x'][0], sol['x'][1], 0.0, sol['x'][2]], (2,2))
c = matrix([sol['x'][3], sol['x'][4]])

if pylab_installed:
    pylab.figure(2, facecolor='w')

    # polyhedron
    for k in range(m):
        edge = X[[k,k+1],:] + 0.1 * matrix([1., 0., 0., -1.], (2,2)) * \
            (X[2*[k],:] - X[2*[k+1],:])
        pylab.plot(edge[:,0], edge[:,1], 'k')


    # 1000 points on the unit circle
    nopts = 1000
    angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
    circle = matrix(0.0, (2,nopts))
    circle[0,:], circle[1,:] = cos(angles), sin(angles)

    # ellipse = L * circle + c
    ellipse = L * circle + c[:, nopts*[0]]

    pylab.fill(ellipse[0,:].T, ellipse[1,:].T, facecolor = '#F0F0F0')
    pylab.plot([c[0]], [c[1]], 'ko')
    pylab.title('Maximum volume inscribed ellipsoid center (fig 8.6)')
    pylab.axis('equal')
    pylab.axis('off')


# Analytic center.
#
# minimize  -sum log (h-G*x)
#

def F(x=None, z=None):
    if x is None: return 0, matrix(0.1, (2,1))
    y = h-G*x
    if min(y) <= 0: return None
    f = -sum(log(y))
    Df = (y**-1).T * G
    if z is None: return matrix(f), Df
    H =  G.T * spdiag(y**-2) * G
    return matrix(f), Df, z[0]*H

sol = solvers.cp(F)
xac = sol['x']
Hac = G.T * spdiag((h-G*xac)**-1) * G

if pylab_installed:
    pylab.figure(3, facecolor='w')

    # polyhedron
    for k in range(m):
        edge = X[[k,k+1],:] + 0.1 * matrix([1., 0., 0., -1.], (2,2)) * \
            (X[2*[k],:] - X[2*[k+1],:])
        pylab.plot(edge[:,0], edge[:,1], 'k')


    # 1000 points on the unit circle
    nopts = 1000
    angles = matrix( [ a*2.0*pi/nopts for a in range(nopts) ], (1,nopts) )
    circle = matrix(0.0, (2,nopts))
    circle[0,:], circle[1,:] = cos(angles), sin(angles)

    # ellipse = L^-T * circle + xc  where Hac = L*L'
    lapack.potrf(Hac)
    ellipse = +circle
    blas.trsm(Hac, ellipse, transA='T')
    ellipse += xac[:, nopts*[0]]
    pylab.fill(ellipse[0,:].T, ellipse[1,:].T, facecolor = '#F0F0F0')
    pylab.plot([xac[0]], [xac[1]], 'ko')

    pylab.title('Analytic center (fig 8.7)')
    pylab.axis('equal')
    pylab.axis('off')
    pylab.show()



X = matrix([ 1.0, 1.0, 0.0, 0.0, 1.0,
             0.0, 1.0, 1.0, 0.0, 0.0], (5,2))
m = X.size[0] - 1

# Inequality description G*x <= h with h = 1
G, h = matrix(0.0, (m,2)), matrix(0.0, (m,1))
G = (X[:m,:] - X[1:,:]) * matrix([0., -1., 1., 0.], (2,2))
h = (G * X.T)[::m+1]
G = mul(h[:,[0,0]]**-1, G)
G = mul(h[h!=0,[0,0]]**-1, G)
h = matrix(1.0, (m,1))

## setting up G and h

G = matrix(numpy.append(numpy.append(numpy.eye(2),-numpy.eye(2),axis=0),numpy.array([[-1,1]]),axis=0))

h = matrix([1.0,1.0,0.0,0.0,0.0])

from pyOptimUtil.direct import rectOperation

from pyOptimUtil.direct import polyOperation

boxBounds = [
    (0.0,1.0),
    (0.0,1.0)
    ]

sol = polyOperation.findAnalyticCenterBox(boxBounds)

A = numpy.array([[-1,1],[-1,1]],dtype="float")
b = numpy.array([0,-0.5])

x,sol,G,h = polyOperation.findAnalyticCenterBox(boxBounds,A,b,full_output=True)

# Gx \precceq h
# numpy.array(h).flatten() - numpy.array(G).dot(x)
# so we are expecting everything to be positive

print numpy.array(h).flatten() - numpy.array(G).dot(x)

redundantIndex = polyOperation.redundantConstraintBox(boxBounds,A,b)
bindingIndex = polyOperation.bindingConstraintBox(boxBounds,A,b)

origin = h - G * matrix(x)

bindingIndex,hull,newG,newh = polyOperation.bindingConstraintBox(boxBounds,A,b,full_output=True)

x1,sol1,G1,h1 = polyOperation.findAnalyticCenter(newG[bindingIndex.tolist(),:],newh[bindingIndex.tolist()],full_output=True)


# first we find the origins
origin = h - G * matrix(x1)

# construct the set of points
D = mul(origin[:,[0,0]]**-1,G)

import scipy.spatial
import matplotlib.pyplot as plt

hull = scipy.spatial.ConvexHull(D)

points = numpy.array(D)

plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
plt.show()

plt.plot(points[:,0], points[:,1], 'o')
plt.plot(x[0], x[1], 'ro')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()


newOrigin = h[bindingIndex.tolist()] - newG[bindingIndex.tolist(),:] * matrix(x1)
newD = mul(newOrigin[:,[0,0]]**-1,newG[bindingIndex.tolist(),:])
newHull = scipy.spatial.ConvexHull(newD)
newPoints = numpy.array(newD)

plt.plot(newPoints[newHull.vertices,0], newPoints[newHull.vertices,1], 'r--', lw=2)
plt.plot(newPoints[newHull.vertices[0],0], newPoints[newHull.vertices[0],1], 'ro')
plt.show()

plt.plot(newPoints[:,0], newPoints[:,1], 'o')
plt.plot(newCenter[0], newCenter[1], 'ro')
for simplex in newHull.simplices:
    plt.plot(newPoints[simplex,0], newPoints[simplex,1], 'k-')

plt.show()

newA = newHull.equations[:,0:2]
newb = -newHull.equations[:,2]

polyOperation.feasibleStartingValue(newA,newb)

polyOperation.bindingConstraint(newA,newb)

# we expect the center to always be 0 in all dimension
# because the dual polytope is always centered
newCenter = polyOperation.findAnalyticCenter(newA,newb)

primalDistance = newb - newA.dot(newCenter)
primalD = mul(matrix(primalDistance)[:,[0,0]]**-1,matrix(newA))

primalHull = scipy.spatial.ConvexHull(primalD)
primalPoints = numpy.array(primalD)


plt.plot(primalPoints[:,0], primalPoints[:,1], 'o')
#plt.plot(x[0], x[1], 'ro')
for simplex in primalHull.simplices:
    plt.plot(primalPoints[simplex,0], primalPoints[simplex,1], 'k-')

plt.show()

## testing inequality <=> verticies operations

V,hull,A,b,x0 = polyOperation.constraintToVertices(G[0:5,:],h[0:5],full_output=True)

points = hull.points
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(x[0], x[1], 'ro')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()


ATest,bTest = polyOperation.verticesToConstraint(V)

polyOperation.constraintToVertices(ATest,bTest)

polyOperation.findAnalyticCenter(ATest,bTest)
polyOperation.findAnalyticCenter(G[0:5,:],h[0:5])

hull.simplices
hull.points

##  splitting

newV = V[hull.simplices[0,:],:]
newV = numpy.append(newV,numpy.reshape(numpy.array(x0),(1,2)),axis=0)

ATest,bTest = polyOperation.verticesToConstraint(newV)

newHull = scipy.spatial.ConvexHull(newV)

newPoints = newV

plt.plot(newPoints[:,0], newPoints[:,1], 'o')
plt.plot(x0[0], x0[1], 'ro')
plt.plot(xc.value[0], xc.value[1], 'go')
for simplex in newHull.simplices:
    plt.plot(newPoints[simplex,0], newPoints[simplex,1], 'k-')

plt.show()

ATest,bTest = polyOperation.verticesToConstraint(newV)

V,hull,A,b,x0 = polyOperation.constraintToVertices(ATest,bTest,full_output=True)


### now we test on a standard rectangle case
from pyOptimUtil.direct import directObj, polyOperation, directUtil, directAlg, optimTestFun
from pyOptimUtil.direct import directObj, polyOperation, directAlg, optimTestFun
from pyOptimUtil.direct import directUtil
import pyOptimUtil.direct
import numpy
import scipy.spatial
import matplotlib.pyplot as plt

lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)

A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

polyObj = polyOperation.PolygonObj(optimTestFun.rosen,A,b)
polyObj.getMaxDistanceToVertices()

V, dualHull, G, h, x0 = polyOperation.constraintToVertices(A,b,full_output=True)

newHull = scipy.spatial.ConvexHull(V)

points = V
plt.plot(points[:,0], points[:,1], 'o')
plt.plot(x0[0], x0[1], 'ro')
for simplex in newHull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()

polyObjList = polyOperation.divideGivenPolygon(optimTestFun.rosen,polyObj)

# showing the correct initial divide
pyOptimUtil.direct.plotDirectPolygon(polyObjList)

for i in range(0,len(polyObjList)):
    print "f(x) = " +str(polyObjList[i].getFx())
    print polyObjList[i].getMaxDistanceToVertices()


# now we want to identify a polygon which we want to divide
polyOperation.identifyPotentialOptimalPolygonPareto(polyObjList)

newPolyList = polyOperation.divideGivenPolygon(optimTestFun.rosen,polyObjList[3])

pyOptimUtil.direct.plotDirectPolygon(newPolyList)

ATemp, bTemp = newPolyList[5].getInequality()
polyOperation.findAnalyticCenter(ATemp,bTemp)
V = polyOperation.constraintToVertices(ATemp,bTemp)
hull = scipy.spatial.ConvexHull(V,False)

points = V
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.show()



newPolyList = directAlg.dividePoly(optimTestFun.rosen,polyObjList)
pyOptimUtil.direct.plotDirectPolygon(newPolyList)

newNewPolyList = directAlg.dividePoly(optimTestFun.rosen,newPolyList)
pyOptimUtil.direct.plotDirectPolygon(newNewPolyList)



potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(newPolyList)

for i in range(0,len(potentialIndex)):
    print "f(x) = " +str(newPolyList[i].getFx())
    print newPolyList[i].getMaxDistanceToVertices()

pyOptimUtil.direct.plotDirectPolygon(newPolyList,potentialIndex)

x,sol,G,h = polyOperation.findAnalyticCenter(A,b,full_output=True)

import pyOptimUtil.direct

rectListOptim,output = directAlg.directOptim(optimTestFun.rosen,lb,ub,
                                                 iteration=50,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

pyOptimUtil.direct.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

# class object 
directObj = directAlg.direct(optimTestFun.rosen,lb,ub)
rectListOptim = directObj.divide(10)


directObj = directAlg.direct(optimTestFun.rosen,lb,ub,A,b)
polyListOptim = directObj.divide(5)

potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)

pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex)

polyListOptim = directObj.divide(1)

potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)

pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex)


polyObj = polyListOptim[25]

polyObj._location
hull = scipy.spatial.ConvexHull(polyObj._V)

x,sol,G,h = polyOperation.findAnalyticCenter(hull.equations[:,0:2],-hull.equations[:,2],full_output=True)


for i in potentialIndex:
    print polyListOptim[i].getLocation()
    print polyListOptim[i].hasSplit()

print polyListOptim[6].getVertices()
print polyListOptim[6].getLocation()

for o in polyListOptim:
    print o.hasSplit()


minIndex = directUtil.findLowestObjIndex(polyListOptim)

polyListOptim[minIndex].getFx()
polyListOptim[minIndex].getLocation()


directObj = directAlg.direct(optimTestFun.rosen,lb,ub,A,b)

polyListOptim = directObj.divide(1)

potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)
pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex,lb-0.5,ub+0.5)

for o in listObj:
    print o.getLocation()

polyListOptim = directObj.divide(1)
potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)
pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex,lb-0.5,ub+0.5)



## Testing the triangulation

lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)

A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

polyObj = polyOperation.PolygonObj(optimTestFun.rosen,A,b)
polyObj.getMaxDistanceToVertices()

V, dualHull, G, h, x0 = polyOperation.constraintToVertices(A,b,full_output=True)

tri = scipy.spatial.Delaunay(V)

points = V
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()


for simplex in tri.simplices:
    plt.plot(points[simplex,0], points[simplex,1], 'k-')

plt.ylim([-2.5,2.5])
plt.xlim([-2.5,2.5])
plt.show()

newV = V[tri.simplices[0],:]

newTri = scipy.spatial.Delaunay(newV)

newTri.simplices
for simplex in newTri.simplices:
    plt.plot(newV[simplex,0], newV[simplex,1], 'k-')

plt.show()

##
##
##  Note that triangulation means dividing a polygon into triangle and not further
##
##


## Center of facet


# 2D
lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)

A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

V, dualHull, G, h, x0 = polyOperation.constraintToVertices(A,b,full_output=True)

# 3D
lb = numpy.array([-2.,-2.,-2.],float)
ub = numpy.array([2.,2.,2.],float)

A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

V, dualHull, G, h, x0 = polyOperation.constraintToVertices(A,b,full_output=True)


## triangle
V = numpy.array([[2,2],[-2,-2],[2,-2]],float)

## another triangle
V = numpy.array([[0,0],[0,1],[1,0]],float)

## Tetrahegron
V = numpy.array([[0,0,0],[0,0,1],[0,1,0],[1,0,0]],float)

## our routine
hull = scipy.spatial.ConvexHull(V)

b = numpy.zeros((len(hull.simplices[:,0]),len(hull.points[0,:])))

for i in range(0,len(hull.simplices[:,0])):
    b[i] = numpy.mean(hull.points[hull.simplices[i,:],:],axis=0)


hull.points[hull.simplices[0,:],:]
polyOperation.verticesToConstraint(hull.points[hull.simplices[0,:],:])
polyOperation.PolygonObj(optimTestFun.rosen,None,None,hull.points[hull.simplices[0,:],:])

polyObj = polyOperation.PolygonObj(optimTestFun.rosen,None,None,V)

## TODO: Currently facing with a rank deficiency problem where the face of a polygon is 
## actually a hyperplane


