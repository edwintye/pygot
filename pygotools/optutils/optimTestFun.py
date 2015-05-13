import math

### f_\min = 0
### x_\min = [0,0]
def ackley(theta):
    x = theta[0]
    y = theta[1]
    
    return -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x**2 + y**2))) - math.exp(0.5 * (math.cos(2.0*math.pi*x) + math.cos(2.0*math.pi*y))) + math.exp(1) + 20

### f_\min = 0
### x_\min = [3,0.5]
def beale(theta):
    x = theta[0]
    y = theta[1]
    
    obj = (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
    return obj
    
### f_\min = 0
### x_\min = [1,3]
def booth(theta):
    x = theta[0]
    y = theta[1]
    
    return (x + 2.0*y - 7)**2 + (2.0*x + y - 5)**2

### f_\min = 0
### x_\min = [-10,1]
def bukin(theta):
    x = theta[0]
    y = theta[1]
    
    return 100.0*math.sqrt(abs(y-0.001*x**2)) + 0.01 * abs(x+10.0)

### f_\min = -1
### x_\min = [\pi,\pi]
def easom(theta):
    x = theta[0]
    y = theta[1]
    
    return -math.cos(x) * math.cos(y) * math.exp(-((x-math.pi)**2 + (y-math.pi)**2)) 

### f_\min = 3
### x_\min = [0,-1]
def gp(x):
    x1 = x[0]
    x2 = x[1]
    
    obj = (1.0+(x1+x2+1.0)**2 * (19-14*x1+3*x1**2-14 * x2+6.0*x1*x2+3*x2**2)) * (30+(2*x1-3*x2)**2 * (18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2))
    return obj

### f_\min = 0
### x_\min = [3,2] or [-2.805118,3.131312] or [-3.779310,-3.283186] or [3.584428,-1.848126]
def himmelblau(theta):
    x = theta[0]
    y = theta[1]
    
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2

### f_\min = 0
### x_\min = [1,1]
def levi(theta):
    x = theta[0]
    y = theta[1]
    
    return math.sin(3.0*math.pi*x)**2 + (x-1.0)**2 * (1 + math.sin(3.0*math.pi*y)**2) + (y - 1.0)**2 * (1.0-math.sin(2.0*math.pi*y)**2)

### f_\min = 0
### x_\min = [0,0]
def matyas(theta):
    x = theta[0]
    y = theta[1]
    
    return 0.26 * (x**2 + y**2) - 0.48*x*y

### f_\min = -1.9133
### x_\min = [-0.54719,-1.54719]
def mccormick(theta):
    x = theta[0]
    y = theta[1]
    
    return math.sin(x+y) + (x-y)**2 - 1.5 * x + 2.5 * y + 1.0

### f_\min = 0
### x_\min = [1,1] or n_{i} = 1 \forall i \in Dimension whatever
def rosen(x):
    numParam = len(x)
    
    obj = 0
    for i in range(0,numParam-1):
        obj += 100.0 * (x[i+1] - x[i]**2)**2 + (x[i] - 1.0)**2
    
    return obj

### f_\min = -39 d
### x_\min = numpy.ones((d,)) * -2.9035
def styblinski(x):
    obj = 0.0
    for i in range(0,len(x)):
        obj += x[i]**4 - 16.0 * x[i]**2 + 5.0 * x[i]
    
    return obj/2.0

