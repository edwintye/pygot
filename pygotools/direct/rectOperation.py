
__all__ = [
           'RectangleObj',
           'identifyPotentialOptimalRectangleLipschitz',
           'identifyPotentialOptimalRectanglePareto'
           ]

import numpy
import copy
#from directObj import RectangleObj
from directUtil import findLowestObjIndex, findHighestObjIndex
from directUtil import identifyPotentialOptimalObjectPareto

class RectangleObj(object):

    def __init__(self, fx, lb, ub, location=None, g=None, H=None):
        self._ub = None
        self._lb = None
        
        if type(lb) is list:
            lb = numpy.array(lb)
            
        if type(lb) is numpy.ndarray:
            self._dimension = len(lb)
        else:
            raise Exception("Expecting lower bounds to be of type numpy.ndarray")
        
        if numpy.any(lb > ub):
            for i in range(self._dimension):
                if lb[i] > ub[i]:                
                    lbT = lb[i]
                    lb[i] = ub[i]
                    ub[i] = lbT
        self.setLB(lb)
        self.setUB(ub)

        if location is not None:
            self._checkDimension(location)
            self.setLocation(location)
        else:
            self._location = self.computeLocation(lb, ub)

        self._volume = self.computeVolume(lb, ub)

        self._fx = fx
        self._g = g
        self._H = H

    def getFx(self):
        return copy.deepcopy(self._fx)
    
    def setFx(self, value):
        self._fx = copy.deepcopy(value)
    
    def getGradient(self):
        return self._g

    def setGradient(self, g):
        self._g = g

    def getHessian(self):
        return self._H

    def setHessian(self, H):
        self._H = H
    
    #@property
    def getLB(self):
        return copy.deepcopy(self._lb)
    
    def setLB(self, value):
        self._checkDimension(value)
        self._lb = copy.deepcopy(value)
        return None
    
    def getUB(self):
        return copy.deepcopy(self._ub)
    
    def setUB(self,value):
        self._checkDimension(value)
        self._ub = copy.deepcopy(value)
        return None
    
    def getLocation(self):
        return copy.deepcopy(self._location)
    
    def setLocation(self,value):
        self._location = value
        return None
    
    def getVolume(self):
        return copy.deepcopy(self._volume)
    
    def setVolume(self,value):
        self._volume = copy.deepcopy(value)
        return None
    
    def computeVolume(self,lb,ub):
        diffBound = numpy.abs(ub - lb)
        return diffBound.prod()
    
    def getMeasure(self):
        return copy.deepcopy(self._volume)
    
    def computeLocation(self,lb,ub):
        '''
        Assume that the location of the box is the center
        of the box
        '''
        return (ub+lb)/2    
    
    def _checkDimension(self,value):
        if type(value) is numpy.ndarray:
            if len(value) != self._dimension:
                raise Exception("Dimension of input do not conform to the object")
        else:
            raise Exception("Expecting type numpy.ndarray")


def divideGivenRectangle(func, rectObj, scaleLB, boundDiff):
    # TODO: This process is in serial.  Should be able to make a parallel 
    # process out of it fairly simply (saying that, I gave up after 10 min)
    # find out which of the dimensionsn
    operateDimensionIndex = identifyDimensionToSplit(rectObj)
        
    # holder for the list of rectangles
    rectList = list()

    # now we loop through the dimensions
    for j in range(0,len(operateDimensionIndex)):
        # find out the dimension which we wish to operate on 
        i = operateDimensionIndex[j]
        # print("index = " +str(i))
            
        # manipulating the central rectangle
        newLocation = rectObj.getLocation()
        newLB = rectObj.getLB()
        newUB = rectObj.getUB()

        # Plus side, which we name it as 1
        ub = rectObj.getUB()
        lb = rectObj.getLB()
        lb[i] = lb[i] + (ub[i]-lb[i])*(2.0/3.0)
        
        location = rectObj.getLocation()
        location[i] = (ub[i] + lb[i])/2.0
        
        # print("plus location = " +str(location))
        # print("plus ub = " +str(ub))
        # print("plus lb = " +str(lb))
        
        fx = func(inverseScaleLocation(location, scaleLB, boundDiff))
        rectObj1 = RectangleObj(fx,lb,ub,location)
            
        # Negative side, which we name it as 2
        location = rectObj.getLocation()
        ub = rectObj.getUB()
        lb = rectObj.getLB()
        # this is the line that is different to the one above
        ub[i] = lb[i] + (ub[i]-lb[i])*(1.0/3.0)
        location[i] = (ub[i] + lb[i])/2.0
        
        # print("minus location = " +str(location))
        # print("minus ub = " +str(ub))
        # print("minus lb = " +str(lb))
        
        fx = func(inverseScaleLocation(location, scaleLB, boundDiff))
        rectObj2 = RectangleObj(fx,lb,ub,location)  

        # fix the original object
        newUB[i] = rectObj1.getLB()[i]
        newLB[i] = rectObj2.getUB()[i]
        
        # print("center location = " +str(newLocation))
        # print("center ub = " +str(newUB))
        # print("center lb = " +str(newLB))
        
        newFx = rectObj.getFx()
        rectObj = RectangleObj(newFx,newLB,newUB,newLocation)            
        # add new object per dimension
        # need to make a deep copy because they are objects
        rectList.append(copy.deepcopy(rectObj1))
        rectList.append(copy.deepcopy(rectObj2))

    ### finish looping all the "longest" dimensions
    # now we add in the original obj, after adjustment for 
    # all the different dimension
    rectList.append(copy.deepcopy(rectObj))
        
    return rectList

def scaleLocation(location,scaleLB,boundDiff):
    '''
    Scale the input to unit box scale.

    Parameters
    ----------
    location: array like
        input location
    scaleLB: array like
        the translation from center of original box to unit box
    boundDiff: array like
        the difference in each of the bounds

    Returns
    -------
    array like

    '''

    return (location - scaleLB) / boundDiff

def inverseScaleLocation(location, scaleLB, boundDiff):
    '''
    Reverse the scaling operation of one input.

    Parameters
    ----------
    location: array like
        input location
    scaleLB: array like
        the translation from center of original box to unit box
    boundDiff: array like
        the difference in each of the bounds

    Returns
    -------
    array like

    '''

    return location * boundDiff + scaleLB

    
def inverseScaleBounds(rectObj, scaleLB, boundDiff):
    '''
    Reverse the scaling operation for the whole object.

    Parameters
    ----------
    rectObj: :class:`RectObj`
        our rectangle object
    scaleLB: array like
        the translation from center of original box to unit box
    boundDiff: array like
        the difference in each of the bounds

    Returns
    -------
    :class:`.RectangleObj`

    '''

    rectObj.setLB(inverseScaleLocation(rectObj.getLB(), scaleLB, boundDiff))
    rectObj.setUB(inverseScaleLocation(rectObj.getUB(), scaleLB, boundDiff))
    rectObj.setLocation(inverseScaleLocation(rectObj.getLocation(), scaleLB, boundDiff))
    return rectObj
    
def identifyDimensionToSplit(rectObj):
    '''
    Given a rectangle, locate the longest dimension(s) which we would like to split

    Parameters
    ----------
    rectObj: :class:`RectObj`
        our rectangle object

    '''
    
    EPSILON = numpy.sqrt(numpy.finfo(numpy.float).eps)
    # find out which of the dimensions have the largest side
    # first, all the dimension length
    boundsDiff = abs(rectObj.getUB() - rectObj.getLB()) 
    # identify the dimension with the max length
    
    #maxLengthDimensionIndex = boundsDiff.argmax()
    maxLength = boundsDiff.max()
    # the number of side with that "max" length
    
    # holder
    indexList = list()
    for i in range(0,len(boundsDiff)):
        # safeguard against floating point error
        if boundsDiff[i] >= (maxLength - EPSILON):
            indexList.append(i)
    
    return numpy.array(indexList,int)

def identifyDimensionToSplit2(rectObj):
    return numpy.argmax(rectObj.getDistance())
    
def identifyPotentialOptimalRectanglePareto(rectList,EPSILON=1e-4,includeMin=False):
    return identifyPotentialOptimalObjectPareto(rectList,EPSILON=1e-4,uniqueDecimal=8,includeMin=includeMin)

def identifyPotentialOptimalRectangleLipschitz(rectList,lipschitzConstant=None,strongCondition=False,targetMin=0,EPSILON=1e-4):
    ## we find the rectangle with the lowest objective function value
    rectLowestObjIndex = findLowestObjIndex(rectList)
    # if a lipschitzConstant,K in the literature, is not provided, it is 
    # estimated by using the lowest bound
    if lipschitzConstant is None:
        rectHighestObjIndex = findHighestObjIndex(rectList)
        lipschitzConstantMax = (rectList[rectHighestObjIndex].getFx() - (1-EPSILON)*rectList[rectLowestObjIndex].getFx())/rectList[rectHighestObjIndex].getMeasure()

        # print("Max Index: " +str(rectHighestObjIndex)+ ", Min Index: "+str(rectLowestObjIndex))
        
        # print(" Max obj: " +str(rectList[rectHighestObjIndex].getFx())+ "\n" +
        #       " Min Obj: " +str(rectList[rectLowestObjIndex].getFx())+ "\n" +
        #       " Max Volumne: " +str(rectList[rectHighestObjIndex].getVolumne()) +"\n")

        # print("lb of max K = " +str(lipschitzConstantMax))
        # print("lb of min K = " +str(EPSILON*rectList[rectLowestObjIndex].getFx()/rectList[rectHighestObjIndex].getVolumne()))

        #rectangleLowestObjIndex2 = findRectangleLowestObjIndex(newRectList)
        ## lipschitzConstant <- abs(rectList[[rectangleLowestObjIndex]]$fx - newRectList[[rectangleLowestObjIndex2]]$fx)/sqrt(sum((rectList[[rectangleLowestObjIndex]]$location - newRectList[[rectangleLowestObjIndex2]]$location)^2))
        ## the convergence depends on how the lipschitz constant is estimated
        
        ### just the function value assuming that f(x)=0 is possible
        ## lipschitzConstant <- rectList[[rectangleLowestObjIndex]]$fx
        
        ### scales the f(x) with the current volumne
        ### so that it increases as the volumne goes down
        ## lipschitzConstant <- rectList[[rectangleLowestObjIndex]]$fx/rectList[[rectangleLowestObjIndex]]$V
        ### scales the f(x) with the diagonal Euclidean distance of the rectangle
        ## TODO: other size/distance measure
        #squareDist = (rectList[rectLowestObjIndex].getUB() - rectList[rectLowestObjIndex].getLB())**2
        # our distance measure
        D = numpy.linalg.norm(rectList[rectLowestObjIndex].getUB() - rectList[rectLowestObjIndex].getLB())
        #print("Square distance is = " + str(squareDist))
        #print("Which came from index " + str(rectLowestObjIndex))
    
        #lipschitzConstant = abs(rectList[rectLowestObjIndex].getFx()-targetMin)/ math.sqrt(squareDist.sum())
        lipschitzConstant = abs(rectList[rectLowestObjIndex].getFx()-targetMin)/ D
        #lipschitzConstant <- max(lipschitzConstant,100)
        
        ### other random crap
        ## lipschitzConstant <- lipschitzConstantMax/rectList[[rectangleHighestObjIndex]]$V
        ## lipschitzConstant <- 1000
        ## lipschitzConstant <- lipschitzConstantMax
        
        # print("K used = " +str(lipschitzConstant)+ " with distance "+ str(math.sqrt(squareDist.sum())))

    ### finish estimating the lipschitz Constant

    # we need to make a deep copy to prevent the object inside getting changed
    newRectList = copy.deepcopy(rectList)
    #newRectList = list(rectList)
    newRectList[rectLowestObjIndex] = None
    newRectList.remove(None)

    # we are assuming that we will always partition the rectangle
    # currently with the lowest value
    listPotentialOptimalIndex = list()
    listPotentialOptimalIndex.append(rectLowestObjIndex)

    if strongCondition is False:
        for i in range(0,len(rectList)):
            LHS = rectList[rectLowestObjIndex].getFx() - lipschitzConstant*rectList[rectLowestObjIndex].getMeasure()
            # test whether this is the index for the current lowest obj value
            if i!=rectLowestObjIndex:
                ## print(rectList[[rectangleLowestObjIndex]]$fx - lipschitzConstant*rectList[[rectangleLowestObjIndex]]$V < rectList[[i]]$fx - lipschitzConstant*rectList[[i]]$V)
                if (LHS > rectList[i].getFx() - lipschitzConstant*rectList[i].getMeasure()):
                    listPotentialOptimalIndex.append(i)
    else:  #strong definition
        for j in range(0,len(rectList)):
            numTrue = 0
            # pre-compute the LHs of the inequality
            # we have flipping the measure to 1/d_{j} (NOT)
            LHS = rectList[j].getFx() - lipschitzConstant*rectList[j].getVolume()
            for i in range(0,len(rectList)):
                if (i!=j):
                    if (LHS < (rectList[i].getFx() - lipschitzConstant * rectList[i].getMeasure())):
                        numTrue += 1       

        # now test
            ##print(numTrue)
        if (numTrue==(len(rectList)-1)):
            # if it passes the first condition, then we test for the seoncd
            if (LHS < (1-EPSILON)*rectList[rectLowestObjIndex].getFx()):
                listPotentialOptimalIndex.append(j)
                 
    return(listPotentialOptimalIndex)

