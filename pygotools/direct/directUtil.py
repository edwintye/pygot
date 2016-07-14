
__all__ = [
    'IdConditionType',
    'identifyPotentialOptimalObjectPareto',
    'plotDirectBox',
    'plotDirectPolygon',
    'plotParetoFrontRect',
    'plotParetoFrontPoly'
    ]

from enum import Enum

import numpy
import matplotlib.pyplot as plt
import scipy.spatial

# cannot do this because of the reverse importation
#from rectOperation import identifyPotentialOptimalRectanglePareto

class IdConditionType(Enum):
    '''
    This is an Enum describing the different conditions allowed when identifying
    the rectangles with potential.
    
    The following four types of transitions are available.
    
    Pareto = boxes that define the Pareto front of f(x) and size of box
    
    Strong = Strong condition using the estimated Lipschitz constant
    
    Soft = Soft condition using the estimated Lipschitz constant
    
    '''
    Pareto = 'Pareto'
    Strong = 'Strong'
    Soft = 'Soft'
    
def findLowestObjIndex1(rectList):
    # now we wish to find the rectangle with the lowest objective function
    minObj = rectList[0].getFx()
    minObjIndex = 0
    for i in range(1,len(rectList)):
        if (rectList[i].getFx() < minObj):
            minObjIndex = i
            minObj = rectList[i].getFx()
    
    return minObjIndex
    
def findLowestObjIndex2(rectList):
    # now we wish to find the rectangle with the lowest objective function
    s = len(rectList)
    a = numpy.zeros(s)
    for i in range(0,s):
        a[i] = rectList[i].getFx()

    return numpy.argmin(a)

# apparently, this is the fastest way to find argmin
def findLowestObjIndex(rectList):
    # now we wish to find the rectangle with the lowest objective function
    a = list()
    for r in rectList:
        a.append(r.getFx())

    return a.index(min(a))


def findLowestObjIndex4(rectList):
    # now we wish to find the rectangle with the lowest objective function
    a = list()
    for r in rectList:
        a.append(r.getFx())

    return numpy.argmin(numpy.array(a))

def findHighestObjIndex2(rectList):
    # now we wish to find the rectangle with the highest objective function
    maxObj = rectList[0].getFx()
    maxObjIndex = 0
    for i in range(1,len(rectList)):
        if (rectList[i].getFx() > maxObj):
            maxObjIndex = i
            maxObj = rectList[i].getFx()
    
    return maxObjIndex

def findHighestObjIndex(rectList):
    # now we wish to find the rectangle with the highest objective function
    a = list()
    for r in rectList:
        a.append(r.getFx())

    return a.index(max(a))

def identifyPotentialOptimalObjectPareto(objList,EPSILON=1e-4,uniqueDecimal=4,includeMin=False):
    ## we find the rectangle with the lowest objective function value
    minIndex = findLowestObjIndex(objList)
    minFx = objList[minIndex].getFx()
    # the target value
    c = minFx - EPSILON * abs(minFx)

    # holders
    uniqueMeasureList = list()
    fxList = list()
    # unrolling the information
    for directObj in objList:
        uniqueMeasureList.append(directObj.getMeasure())
        fxList.append(directObj.getFx())

    fxList = numpy.array(fxList)
    # round to only accurate to 8 decimal place to so that our 
    # "uniqueness" is not ruined by floating point precision
    uniqueMeasureList = numpy.around(numpy.array(uniqueMeasureList),uniqueDecimal)
    # put it in reverse order
    uniqueMeasureArray = numpy.unique(uniqueMeasureList)[::-1]

    # extracting the information out for the largest rectangles
    oldV = uniqueMeasureArray[0]
    # OVIL is the oldVIndexList
    OVIL = numpy.where(uniqueMeasureList==oldV)[0]
    oldFx = min(fxList[OVIL])
    #print("Iteration = " +str(0)+ " with volume " +str(oldV))

    # holder
    listPotentialOptimalIndex = list()
    # always split one of the largest polygon
    listPotentialOptimalIndex.append(OVIL[numpy.argmin(fxList[OVIL])])
    # time to find our points on the Pareto front
    for i in range(1,len(uniqueMeasureArray)):
        # find the minimum at 
        V = uniqueMeasureArray[i]
        indexTempList = numpy.where(uniqueMeasureList==V)[0]
        index = indexTempList[numpy.argmin(fxList[indexTempList])]
        fx = fxList[index]
        
        # the pareto condition
        # which boils down to 
        # (f(x)^{t} - c) \over V^{t} \le (f(x)^{t-1} - c) \over V^{t-1}
        # the two have the same gradient when it is an equality sign
        # and the new point is accepted if f(x)^{t} is a smaller gradient
        # (or angle) relative to the measure V
        if fx <= c + (V/oldV) * (oldFx - c):
            listPotentialOptimalIndex.append(index)
            # if we have a new point, we have a new gradient
            oldV = V
            oldFx = fx

    # if we want to include the minimum, then lets go for it!
    if includeMin:
        if minIndex not in listPotentialOptimalIndex:
            listPotentialOptimalIndex.append(minIndex)
        
    return listPotentialOptimalIndex

def plotDirectBox(rectList, paretoIndex=None):
    '''
    Plot the boxes return by the DIRECT algorithm given a two dimensional problem.

    Parameters
    ==========
    rectList: list
        list of rectangles
    lb: array like
        lower bounds
    ub: array like
        upper bounds

    '''

    if (rectList[0].getLB().size != 2):
        raise Exception("Can only plot objective function of 2 dimension")
    
#     lb = numpy.zeros(2)
#     ub = numpy.zeros(2)
#     for rectObj in rectList:
#         
#     # the whole area
#     x = list()
#     y = list()
#     x.append(lb[0])
#     y.append(lb[1])
#     
#     x.append(lb[0])
#     y.append(ub[1])
#     
#     x.append(ub[0])
#     y.append(ub[1])
#     
#     x.append(ub[0])
#     y.append(lb[1])
#     
#     x.append(lb[0])
#     y.append(lb[1])

    
    f = plt.figure()

    # plt.plot(x,y,color='black')
    lb = numpy.Inf*numpy.ones(2)
    ub = numpy.Inf*numpy.ones(2)
    for rectObj in rectList:
        for i in range(2):
            if lb[i] > rectObj.getLB()[i]:
                lb[i] = rectObj.getLB()[i]
            if ub[i] < rectObj.getUB()[i]:
                ub[i] = rectObj.getUB()[i]
    
    addBox(lb, ub)
    
    for rectObj in rectList:
        addBox(rectObj.getLB(), rectObj.getUB(), rectObj.getLocation())
    
    rectLowestObjIndex = findLowestObjIndex(rectList)
    location = rectList[rectLowestObjIndex].getLocation()
    plt.plot(location[0],location[1],'rD')
    
    # index = identifyPotentialOptimalRectanglePareto(rectList)
    if paretoIndex is not None:
        print("Number of potentially optimal rectangle = " +str(len(paretoIndex)))
        for i in paretoIndex:
            location = rectList[i].getLocation()
            plt.plot(location[0],location[1],'bo')
        
    plt.show()
    #plt.savefig("rect")

def addBox(lb, ub, location=None, color='black'):
    x = list()
    y = list()
        
    x.append(lb[0])
    y.append(lb[1])
    
    x.append(lb[0])
    y.append(ub[1])
        
    x.append(ub[0])
    y.append(ub[1])
        
    x.append(ub[0])
    y.append(lb[1])
        
    x.append(lb[0])
    y.append(lb[1])

    plt.plot(x,y,color=color)
    if location is not None:
        plt.plot(location[0], location[1], 'ro')
        
def addCircle(x, y, r):
    ax = plt.gca()
    circle = plt.Circle((x,y), r)
    circle.set_facecolor('none')
    circle.set_edgecolor('black')
    circle.set_alpha(1.0)
    ax.add_artist(circle)

def plotDirectPolygon(polyList, paretoIndex=None, lb=None, ub=None):
    '''
    Plot the boxes return by the DIRECT algorithm given a two dimensional problem.

    Parameters
    ==========
    polyList: list
        list of polygon

    '''

    f = plt.figure()

    A,b = polyList[0].getInequality()
    if (len(A[0,:]) != 2):
        raise Exception("Can only plot objective function of 2 dimension")        

    for polyObj in polyList:
        if polyObj.hasSplit() == False:
            V = polyObj.getVertices()
            x0 = polyObj.getLocation()
            hull = scipy.spatial.ConvexHull(V)

            # plot a single polygon
            points = V
            # plt.plot(points[:,0], points[:,1], 'o')
            # plt.plot(x0[0], x0[1], 'go')
            for simplex in hull.simplices:
                plt.plot(points[simplex,0], points[simplex,1], 'k-')
    
    rectLowestObjIndex = findLowestObjIndex(polyList)
    location = polyList[rectLowestObjIndex].getLocation()
    plt.plot(location[0],location[1],'rD')
    
    # index = identifyPotentialOptimalRectanglePareto(rectList)
    if paretoIndex is not None:
        print("Number of potentially optimal rectangle = " +str(len(paretoIndex)))
        for i in paretoIndex:
            location = polyList[i].getLocation()
            plt.plot(location[0],location[1],'bo')
        
    if lb is not None:
        plt.xlim([lb[0],ub[0]])
        plt.xlim([lb[1],ub[1]])
    plt.show()

def plotParetoFrontRect(rectList, paretoIndex=None):
    '''
    Plot the Pareto front formed by the input boxes

    Parameters
    ==========
    rectList: list
        list of rectangles

    '''

    f = plt.figure()

    for rect in rectList:
        plt.plot(rect.getMeasure(),rect.getFx(),'bo')

    if paretoIndex is not None:
        for i in paretoIndex:
            plt.plot(rectList[i].getMeasure(),rectList[i].getFx(),'ro')

    #plt.ylim(0,1000)
    #plt.xlim(0,0.04)
    plt.show()

def plotParetoFrontPoly(polyList, paretoIndex=None):
    '''
    Plot the Pareto front formed by the input boxes

    Parameters
    ==========
    rectList: list
        list of rectangles

    '''

    f = plt.figure()

    for poly in polyList:
        if poly.hasSplit()==False:
            plt.plot(poly.getMeasure(),poly.getFx(),'bo')

    if paretoIndex is not None:
        for i in paretoIndex:
            plt.plot(polyList[i].getMeasure(),polyList[i].getFx(),'ro')

    #plt.ylim(0,1000)
    #plt.xlim(0,0.04)
    plt.show()



