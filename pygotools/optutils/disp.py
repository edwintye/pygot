import numpy

class Disp(object):
	
    def __init__(self,level):
        
        self._level = level
        self._h()
        #self._x(numpy.arange(5))
	
    def d(self, i, x, fx=None, deltaX=None, grad=None):
        print str(i).ljust(5),
        
        if self._level==0:
            pass
        elif self._level==1:
            self._x(x)
        elif self._level==2:
            self._fx(fx)
        elif self._level==3:
            self._fx(fx)
            self._x(x)
        elif self._level==4:
            self._fx(fx)
            self._deltaXGrad(deltaX.dot(grad))
	   	
        print ""
	   	
    def _h(self):
        if self._level==0:
            pass
        elif self._level==1:
            print "iter  Parameters"
        elif self._level==2:
            print "iter  f(x)"
        elif self._level==3:
            print "iter  f(x)     Parameters"
        elif self._level==4:
            print "iter  f(x)     Newton Decrement"
        
    def _fx(self, fx):
     	print "{0:0.2g}".format(fx).ljust(8),

    def _x(self, x):
        print ["%0.2f"% s for s in x.tolist()],
		
    def _deltaXGrad(self, a):
        print "{0:0.2g}".format(a).ljust(4),
