#%load_ext autoreload
#%autoreload 2
### now we test on a standard rectangle case
from pyOptimUtil.direct import directObj, polyOperation, directUtil, directAlg, optimTestFun, rectOperation
from pyOptimUtil.direct import directObj, polyOperation, directAlg, optimTestFun
import pyOptimUtil.direct
import numpy
import scipy.spatial
import matplotlib.pyplot as plt

lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)

lb = numpy.array([-2.,-2.,-2],float)
ub = numpy.array([2.,2.,2],float)


rectListOptim,output = directAlg.directOptim(optimTestFun.rosen,lb,ub,
                                                 iteration=50,
                                                 numBox=1000,
                                                 targetMin=0,
                                                 scaleOutput=False,
                                                 full_output=True)

index = directUtil.findLowestObjIndex(rectListOptim)
rectListOptim[index].getFx()
rectListOptim[index].getLocation()

pyOptimUtil.direct.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)

# class object 
directObj = directAlg.direct(optimTestFun.rosen,lb,ub)
rectListOptim,output = directObj.divide(50,numBox=10000,full_output=True)

potentialIndex = directUtil.identifyPotentialOptimalObjectPareto(rectListOptim)
pyOptimUtil.direct.directUtil.plotParetoFrontRect(rectListOptim,potentialIndex)

pyOptimUtil.direct.plotDirectBox(rectListOptim,lb,ub,scaleOutput=False)


# in terms of inequalities 
A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

directObj = directAlg.direct(optimTestFun.gp,lb,ub,A,b)
polyListOptim,output = directObj.divide(10,numBox=2000,full_output=True)

potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)

pyOptimUtil.direct.directUtil.plotParetoFrontPoly(polyListOptim,potentialIndex)

pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex)

polyListOptim = directObj.divide(1)

potentialIndex = polyOperation.identifyPotentialOptimalPolygonPareto(polyListOptim)

pyOptimUtil.direct.plotDirectPolygon(polyListOptim,potentialIndex)


index = directUtil.findLowestObjIndex(polyListOptim)
polyListOptim[index].getFx()
polyListOptim[index].getLocation()

for o in polyListOptim:
    print "f(x) = " +str(o.getFx())+ " Location :" 
    print o.getLocation()
    print o._simplexGrad
    print str(o._decreaseDirectionFromParent) +"  "+ str(o.hasSplit())
    
listD = numpy.zeros(len(polyListOptim))
for i in range(0,len(polyListOptim)):
    listD[i] = polyListOptim[i].getMeasure()

G,h = polyListOptim[207].getInequality()
G.dot(polyListOptim[207].getLocation()) - h.flatten()

polyObj = polyOperation.PolygonObj(optimTestFun.rosen,A,b)

polyList = polyOperation.divideGivenPolygon(optimTestFun.rosen,polyObj)



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


lb = numpy.array([-2.,-2.],float)
ub = numpy.array([2.,2.],float)
A,b = polyOperation.addBoxToInequalityLBUB(lb,ub)

V, dualHull, G, h, x0 = polyOperation.constraintToVertices(A,b,full_output=True)

V = numpy.array([[0,0],[0,1],[1,0]],float)

hull = scipy.spatial.ConvexHull(V)

A = hull.equations[:,:-1]
b = hull.equations[:,-1:]
V = hull.points

G * matrix(V.T) - h[:,matrix(0,(4,1))] >= 1e-8

abs(A.dot(V.T) + b)>=1e-8

numpy.where(abs(A.dot(V.T) - numpy.reshape(b,(4,1)))>=1e-8)

diffA = abs(A.dot(V.T) - numpy.reshape(b,(4,1)))>=1e-8
diffAIndex = numpy.linspace(0,3,4)

planeIndex = numpy.zeros((6,8))
for i in range(0,6):
    planeIndex[i] = diffAIndex[diffA[i]]

newV = V[diffAIndex[diffA[i]].tolist(),:]
newV = numpy.append(newV,numpy.reshape(numpy.array(x0),(1,3)),axis=0)

newHull = scipy.spatial.ConvexHull(newV)


polyObj = polyOperation.PolygonObj(optimTestFun.rosen,None,None,newV)
