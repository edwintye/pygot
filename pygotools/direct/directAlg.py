
__all__ = [
    'direct',
    'directOptim'
    ]

import functools

import numpy
import copy
from scipy.optimize import approx_fprime

from rectOperation import RectangleObj
from rectOperation import divideGivenRectangle
from rectOperation import identifyPotentialOptimalRectangleLipschitz
from rectOperation import identifyPotentialOptimalRectanglePareto
from rectOperation import inverseScaleLocation, inverseScaleBounds

from polyOperation import triangulatePolygon, divideGivenPolygon
from polyOperation import identifyPotentialOptimalPolygonPareto
from polyOperation import PolygonObj

from directUtil import IdConditionType, findLowestObjIndex, plotDirectPolygon
from rectCut import LipschitzGradient, _boxWithinNewtonStep

from pygotools.optutils.consMani import addBoxToInequality, addLBUBToInequality
from pygotools.gradient.finiteDifference import forwardGradCallHessian

TOLERENCE = 1E-4
abstol = 1e-8
reltol = 1e-8
eps = numpy.sqrt(numpy.finfo(float).eps)

class direct(object):
    '''
    DIRECT object
    
    Parameters
    ----------
    func: callable
        objective function
    lb: array like
        lower bounds for the parameters
    ub: array like
        upper bounds for the parameters
    grad: callable
        gradient of the objective function
    hess: callable
        hessian of the objective function
    A: array like, optional
        matrix of A in Ax<=b 
    b: array like, optional
        vector of b in Ax<=b
    conditionType: enum, optional
        of class :class:`.IdConditionType`.  Defaults to Pareto like condition
        to select the next set of boxes
    targetMin: float, optional
        the target (best possible) minimum, i.e. 0 for a square/absolute loss
    EPSILON: float, optional
        tolerence for \varepsilon-optimal condition
    TOLERENCE: float, optional
        parameter to determine how local the search it
        
    '''

    def __init__(self,
                 func,
                 lb,
                 ub,
                 grad=None,
                 hess=None,
                 A=None,
                 b=None,
                 conditionType=None,
                 targetMin=0.0,
                 EPSILON=1e-4,
                 TOLERENCE=1e-4):
        '''     
        Constructor
        
        '''

        self._func = copy.deepcopy(func)
        if grad is None:
            grad = functools.partial(approx_fprime, f=func, epsilon=eps)
        if hess is None:
            hess = functools.partial(forwardGradCallHessian, grad, h=eps)

        self._grad = copy.deepcopy(grad)
        self._hess = copy.deepcopy(hess)
        
        self._numParam = len(lb)

        # we always scale the bounds
        self._boundDiff = ub-lb
        self._scaleLB = lb
        self._lb = lb
        self._ub = ub

        self._locationFunc = functools.partial(inverseScaleLocation, scaleLB=self._scaleLB, boundDiff=self._boundDiff)
        
        # other information
        self._A = A
        self._b = b
        self._conditionType = conditionType
        self._targetMin = targetMin
        self._EPSILON = EPSILON
        self._TOLERENCE = TOLERENCE

        if A is None:
            self._rectType = True
        else:
            self._rectType = False
            # and we go and find the set of inequalities that define our
            # convex polygon
            self._G,self._h = addLBUBToInequality(lb, ub, A, b)

        self._objList = None
        self._objCutList = None

    def initialDivide(self):
        '''
        The first divide from the origin
        
        Returns
        =======
        list:
            a list of objects, either of type :class:`.PolygonObj` or 
            :class:`.RectangleObj` depending on what type of equalities
            were used
        '''       
        
        if self._rectType:
            return self.initialDivideRect()
        else:
            return self.initialDividePoly()
    
    def initialDivideRect(self):
        '''
        The first divide from the origin for box constraints
        
        Returns
        =======
        list:
            a list of objects, of type :class:`.RectangleObj` 
            
        '''     
        # lb are ub are standardized to [0,1]
        boxLB = numpy.zeros(self._numParam)
        boxUB = numpy.ones(self._numParam)
        
        # the center if obviously the average between the lb and ub
        location = (boxUB + boxLB) / 2
        # find f(x)
        fx = self._func(self._locationFunc(location))
        # initialize the object
        rectObj = RectangleObj(fx, boxLB, boxUB, location)
        # and divide it the first time
        return divideGivenRectangle(self._func, rectObj, self._scaleLB, self._boundDiff)

    def initialDividePoly(self):
        '''
        The first divide from the origin under linear equalities.  This performs
        a triangulation on the polygon.
        
        Returns
        =======
        list:
            a list of objects, of type :class:`.PolygonObj` 
            
        '''     
        
        polyObj = PolygonObj(self._func, self._G, self._h)
        return triangulatePolygon(self._func, polyObj)

    def divide(self, iteration=50, numBox=1000, scaleOutput=False, exclusion=None, includeMin=False, includeLocal=False, full_output=False):
        '''
        Dividing up the parameter space with respect to the type of bounds

        Parameters
        ----------
        iteration: int, optional
            maximum number of iterations allowed. Defaults to 50
        numBox: int, optional
            maximum number of boxes allowed. Defaults to 1000
        scaleOutput: bool, optional
            the rectangles will be scaled to between bounds 0 and 1 if True.  
            Note that we scale only when box constraints are present
        exclusion: str, optional
            defaults to None.  Takes 'exact' or 'conservative' which will
            find the underestimator of each box using :func:`. LipschitzGradient`
            to determine whether the box should be a candidate for further division
        includeMin: bool, optional
            if the box with the minimum objective value will be divided
        includeLocal: bool, optional
            whether the box reached using a Newton step from the the box
            with the minimum objective value will be divided
        full_output: bool, optional
            if extra information is required

        Returns
        -------
        objList: list
            list of the objects after dividing
        infodict : dict, only returned if full_output == True
            Dictionary containing additional output information

            =========  ============================================================
            key        meaning
            =========  ============================================================
            'message'  reason for stopping
            'fx'       progress of the objective value
            'iter'     number of iterations taken
            'scale'    if the bounds on the output has been scaled
            'newBox'   number of new box in each iteration
            'numBox'   final number of box
            'bi'       index of the box with the smallest objective value
            'x'        the set of parameters with the smallest objective value
            =========  ============================================================  

        '''     
        
        # holders in case we want more output information
        if exclusion is not None:
            if exclusion.lower() in ('exact', 'conservative'):
                if self._objCutList is None:
                    self._objCutList = list()
            else:
                raise Exception("Exclusion type not recognized")

        output = dict()
        minFxList = list()
        newBox = list()
        numRepeat = 0

        if includeMin:
            if includeLocal:
                maxRepeat = 8
            else:
                maxRepeat = 5
        elif includeLocal:
            maxRepeat = 3 
        else:
            maxRepeat = 10
                    
        # define some arbitrary large number
        oldFStar = numpy.Inf
        if self._objList is None:
            self._objList = self.initialDivide()
            newBox.append(len(self._objList))
        # else:
        #     directObjList = copy.deepcopy(self._objList)

        fStar1 = min([box.getFx() for box in self._objList])
        fStar2 = numpy.Inf
        if self._objCutList is not None:
            if len(self._objCutList) > 0:
                fStar2 = min([box.getFx() for box in self._objCutList])
        fStar = min(fStar1, fStar2)
        ## plain old iteration
        currentNumBox = len(self._objList)
        for k in range(0,iteration):
            ## to do, Pareto front for the size of boxes vs f(x)
            ## identification
            # print includeMin

            if self._rectType:
                directObjList = _divideRect(self._func, self._objList, self._scaleLB, self._boundDiff,
                                            conditionType=self._conditionType,
                                            targetMin=self._targetMin, includeMin=includeMin)
            else:
                directObjList = _dividePoly(self._func, self._objList,
                                            includeMin=includeMin, includeLocal=includeLocal)
      
            # now we have the list of box, after split
            minIndex = findLowestObjIndex(directObjList)
            fx = directObjList[minIndex].getFx()
            if fx < fStar:
                fStar = fx

            if len(directObjList) == currentNumBox:
                # this will only happen if we are using a Lipschitz type condition
                # TODO: check whether algorithm is correct
                output['message'] = "Cannot identify a suitable box to split" 
                break

            # check out other conditions that will lead us to stop
            # now we want to know the number of boxes after the iteration
            currentNumBox = len(self._objList + self._objCutList)
            # more information recording
            newBox.append(currentNumBox)

            # we know that fx should always be smaller or equal to oldFx
            # print (oldFx - fx)
            if (currentNumBox >= numBox):
                output['message'] = "Exceeded the number of Box, " + str(numBox)
                break
            elif (oldFStar - fStar) <= abstol or ((oldFStar - fStar)/fStar) <= reltol:
                numRepeat += 1
                if numRepeat >= maxRepeat:
                    output['message'] = ('No major improvement in f(x) in ' +str(maxRepeat)+'consecutive iteration.'
                                         + ' Absolute difference is ' +str(oldFStar - fStar)
                                         + ' and relative difference is ' + str((oldFStar - fStar)/fStar)
                                         )
                    break
            elif (abs(self._targetMin - fStar) < self._TOLERENCE):
                # this is a special condition because if we have a target, then it 
                # should be respected (aka I ain't going to question your prior)
                output['message'] = 'Reached the target minimum (within epsilon)'
                break
            else:
                numRepeat = 0

            oldFStar = fStar
            minFxList.append(oldFStar)
            
            if self._rectType:
                if exclusion is not None:
                    self._objList = list()
                    exList = LipschitzGradient(directObjList, fStar=fStar,
                                               func=self._func, grad=self._grad, hess=self._hess, 
                                               locationFunc=self._locationFunc,
                                               epsilon=self._EPSILON, LType=exclusion)

                    for i, ex in enumerate(exList):
                        if ex == True:
                            box = directObjList[i]
                            c = self._locationFunc(box.getLocation())

#                             print "\n Cut box"
#                             print c
#                             print _boxWithinNewtonStep(box.getLB(), box.getUB(), c, box.getGradient(), 
#                                                        max(abs(numpy.linalg.eigvalsh(box.getHessian()))))
                            self._objCutList.append(directObjList[i])
                        else:
                            self._objList.append(directObjList[i])
            # print len(directObjList)
#             print "\n Current cut list"
#             for box in self._objCutList:
#                 print self._locationFunc(box.getLocation())
        # end for iteration

        # make a copy within the object itself
        # so we can operate more on it
        # this object is always scaled
        # self._objList = copy.deepcopy(directObjList)

        # scale it if we have to
        if self._rectType:
            if scaleOutput == False:
                boxList = copy.deepcopy(self._objList) + copy.deepcopy(self._objCutList)
                for i in range(len(boxList)):
                    boxList[i] = inverseScaleBounds(boxList[i], self._scaleLB, self._boundDiff)

        fx = numpy.array([box.getFx() for box in boxList])
#         print len(self._objList)
#         print len(self._objCutList)
#         print "\n Final cut list"
#         for box in self._objCutList:
#             print self._locationFunc(box.getLocation())

        if full_output:
            output['scale'] = scaleOutput
            if iteration > 0:
                output['iter'] = k+1
                output['numBox'] = len(boxList)
                output['newBox'] = numpy.diff(numpy.array(newBox))
                output['bi'] = numpy.argmin(fx)
                output['x'] = numpy.min(fx)
                output['fx'] = numpy.array(minFxList.append(fStar))
            if 'message' not in output:
                output['message'] = 'Reached the maximum number of iterations allowed'
            return boxList, output
        else:
            return boxList

def _divideRect(func, rectList, scaleLB, boundDiff, conditionType=None, targetMin=0, includeMin=False): 
    
    if conditionType is None:
        potentialOptimalRectangle = numpy.array(identifyPotentialOptimalRectanglePareto(rectList, includeMin=includeMin))
    else:
        if isinstance(conditionType, IdConditionType):   
            if conditionType == IdConditionType.Pareto:
                potentialOptimalRectangle = numpy.array(identifyPotentialOptimalRectanglePareto(rectList, includeMin))
            elif conditionType == IdConditionType.Strong:
                potentialOptimalRectangle = numpy.array(identifyPotentialOptimalRectangleLipschitz(rectList, strongCondition=True, targetMin=targetMin))
            elif conditionType == IdConditionType.Soft:
                potentialOptimalRectangle = numpy.array(identifyPotentialOptimalRectangleLipschitz(rectList, strongCondition=False, targetMin=targetMin))
            else: 
                raise Exception("Expecting input of type IdConditionType")

#     print type(rectList)
#     print len(rectList)
#     print potentialOptimalRectangle
    # sort and order
    potentialOptimalRectangle.sort()
    # invert the sort order, which is sort of wtf because numpy doesn't have a 
    # sort in decreasing order
    potentialOptimalRectangle = potentialOptimalRectangle[::-1]

    ## this will only happen if we are not using the Pareto front condition
    if (len(potentialOptimalRectangle) == 0):
        return rectList
        
    # print "Num potential = "+str(len(potentialOptimalRectangle))
    for actingIndex in potentialOptimalRectangle:
        # print "Index:"+str(actingIndex)
        # extract the object
        rectObj = rectList[actingIndex]
        newRectList = divideGivenRectangle(func, rectObj, scaleLB, boundDiff)
        # now remove the object that we have split
        rectList.remove(rectObj)
        rectList += newRectList
        
    return rectList

def _dividePoly(func, polyList, includeMin=False, includeLocal=False):
    # the list of feasible index
    polyObjListIndex = identifyPotentialOptimalPolygonPareto(polyList, includeMin)

    numPoly = len(polyList)
    # arbitrary minimum
    currentMin = 1e10
    minIndex = None
    if includeLocal:
        for i in range(0,numPoly):
            o = polyList[i]
            if o.hasSplit()==False and o.isDirectionFromParent():
                # print "pass"
                # print "Current = "+str(currentMin)+ " and new = " +str(o.getFx())
                if o.getFx()<=currentMin:
                    currentMin = o.getFx()
                    minIndex = i
        # now equipped with the information, we can finally proceed
        if minIndex is not None:
            if minIndex not in polyObjListIndex:
                polyObjListIndex.append(minIndex)

    polyOperatedList = list()
    
    for i in polyObjListIndex:
        o = polyList[i]
        polyOperatedList.append(o)

        newPolyObjList = divideGivenPolygon(func,o)
        #plotDirectPolygon(newPolyObjList)
        # add to the list
        polyList += newPolyObjList    

    for i in range(0,len(polyOperatedList)):
        polyList.remove(polyOperatedList[i])

    return polyList

def directOptim(func, lb, ub, iteration=50, numBox=200, conditionType=None, targetMin=0.0, EPSILON=1e-4, TOLERENCE=1e-4, scaleOutput=False, full_output=False):
    '''
    DIRECT algorithm.

    Parameters
    ----------
    func: callable
        Cost function 
    lb: array like
        lower bounds of the parameters space
    ub: callable
        upper bounds of the parameters space
    iteration: int
        maximum number of iterations allowed
    numBox: int
        maximum number of boxes allowed 
    strongCondition: bool
        if the box selection criteria should be strong
    targetMin: float
        the target (best possible) minimum, i.e. 0 for
        a square/absolute loss
    EPSILON: float
        parameter to determine how local the search it
    TOLERENCE: float
        tolerence for :math:`\varepsilon`-optimal condition
    scaleOutput: bool
        the rectangles will be scaled to between bounds
        0 and 1 if True
    full_output: bool
        if extra information is required

    Returns
    -------
    rectList: list
        list of the rectangles after divide
    infodict : dict, only returned if full_output == True
        Dictionary containing additional output information

        =========  ==============================================
        key        meaning
        =========  ==============================================
        'message'  reason for stopping
        'iter'     number of iterations taken
        'scale'    if the bounds on the output has been scaled
        'numBox'   final number of box
        'mbi'      index of the box with the smallest objective
                   value
        =========  ==============================================

    '''

    # holders
    output = dict()
    minFxList = list()
    numRepeat = 0
    
    if iteration is None:
        raise Exception("Integer number of iterations required")
        
    if numBox is None:
        numBox = 200

    ## we need to find the inputed bound so that we can rescale it to \left[0,1\right]
    boundDiff = ub-lb
    scaleLB = -lb
    
    # the number of parameters are!!! drum roll please...
    numParam = len(lb)
    
    ## reintroduce the bounds
    lb = numpy.zeros(numParam)
    ub = numpy.ones(numParam)
    
    location = (ub-lb)/2
    
    fx = func(inverseScaleLocation(location, scaleLB, boundDiff))
    rectObj = RectangleObj(fx, lb, ub, location)
       
    # initial divide
    rectList = divideGivenRectangle(func,rectObj,scaleLB,boundDiff)
    oldFx = 1e6

    ## plain old iteration
    for k in range(0,iteration):
        ## to do, Pareto front for the size of boxes vs f(x)
        ## identification
        rectList = _divideRect(func, rectList, scaleLB, boundDiff,
                               conditionType=conditionType, targetMin=targetMin)
        minIndex = findLowestObjIndex(rectList)
        fx = rectList[minIndex].getFx()
        
        if (len(rectList) >= numBox):
            output['message'] = "Exceeded the number of Box, " + str(numBox)
            break

        if (abs(targetMin - fx) < EPSILON):
            output['message'] = "Reached the target minimum (within epsilon)"
            break

        # we know that fx should always be smaller or equal to oldFx
        if (oldFx - fx) < -TOLERENCE:
            output['message'] = "Local minimum found"
            minFxList.append(fx)
            numRepeat = 0
            break
        elif (oldFx == fx):
            minFxList.append(oldFx)
            numRepeat += 1
            if numRepeat >= 10:
                output['message'] = "No improvement in f(x) in 10 consecutive iteration"
                break
        else:
            oldFx = fx
            minFxList.append(oldFx)
        
    # end for iteration
    if scaleOutput == False:
        for i in range(0, len(rectList)):
            rectList[i] = inverseScaleBounds(rectList[i], scaleLB, boundDiff)        
    
    if full_output:
        output['scale'] = scaleOutput
        output['iter'] = k+1
        output['numBox'] = len(rectList)
        output['mbi'] = minIndex
        output['fx'] = numpy.array(minFxList)
        if 'message' not in output:
            output['message'] = 'Reached the maximum number of iterations allowed'
        return (rectList,output)
    else:
        return rectList

