
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


