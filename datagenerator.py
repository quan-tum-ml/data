import scipy.stats as ss #for normal distribution
import numpy as np


def eins_a(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    cols = 1
    rows = 2
    centroids= np.zeros(shape=(cols*rows,2))
    start = [0.3,0.5]
    inc= 0.4
    Xvals, yvals = [], []
    count=0
    for c in range(0,cols):
        for r in range(0,rows):
            centroids[count][0] = start[0] + r * inc
            centroids[count][1] = start[1] 
            count +=1

    radius = 0.25

    for n in range(samples):

        t = 2*np.pi*np.random.random()
        u = np.random.random()+np.random.random()
        if u>1:
            r = 2-u 
        else: 
            r = u

        a = r*np.cos(t)*radius
        b = r*np.sin(t)*(radius+0.125)
        dot = np.random.randint(0, 2)
        
        if dot in [0,2,5,7,8,10,13,15]:
            yvals.append(0)
        else:
            yvals.append(1)

       
        x = centroids[dot][0]+a - 0.5
        y = centroids[dot][1]+b - 0.5
        
        Xvals.append([x, y])
        
    return np.array(Xvals), np.array(yvals)



def eins_b(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    centroids = [[0.4,0.2],[0.6,0.5],[0.4,0.8]]

    radius = 0.135
    Xvals, yvals = [], []
    for n in range(samples):  

        x = np.arange(-np.pi,np.pi,0.01)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 1.8) - ss.norm.cdf(xL, scale = 1.8)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1

        dot = np.random.randint(0, 3)
        if dot in [0,2]:
            yvals.append(0)
            t = np.random.choice(x, size = 1, p = prob)
        else:
            yvals.append(1)
            t = np.random.choice(x, size = 1, p = prob)+np.pi


        u = np.random.random()+np.random.random()
        if u>1:
            r = 2-u 
        else: 
            r = u

        a = r*np.cos(t)*(radius+0.15)
        b = r*np.sin(t)*radius

        x = centroids[dot][0]+a - 0.5
        y = centroids[dot][1]+b - 0.5

             
        Xvals.append([x[0], y[0]])
        

    
    return np.array(Xvals), np.array(yvals)


def eins_c(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    
    centroids = [[0.3,0.15],[0.7,0.383333],[0.3,0.623333],[0.7,0.85]]

    radius = 0.155

    Xvals, yvals = [], []


    for n in range(samples):  

        x = np.arange(-np.pi,np.pi,0.01)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 1.5) - ss.norm.cdf(xL, scale = 1.5)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1

        dot = np.random.randint(0, 4)
        if dot in [0,2]:
            yvals.append(0)
            t = np.random.choice(x, size = 1, p = prob)
        else:
            yvals.append(1)
            t = np.random.choice(x, size = 1, p = prob)+np.pi

        #t  ###Todo, das darf keine gleichverteilung sein.... biased für eine hälfte
        u = np.random.random()+np.random.random()
        if u>1:
            r = 2-u 
        else: 
            r = u

        a = r*np.cos(t)*(radius+0.15)
        b = r*np.sin(t)*radius

        x = centroids[dot][0]+a -0.5
        y = centroids[dot][1]+b -0.5
        
        Xvals.append([x[0], y[0]])
        
    return np.array(Xvals), np.array(yvals)        


def zwei_a(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    centroids = [[0.3,0.7],[0.7,0.7],[0.3,0.3],[0.7,0.3]]

    radius = 0.15
    Xvals, yvals = [], []

    for n in range(samples):

        t = 2*np.pi*np.random.random()
        u = np.random.random()+np.random.random()
        if u>1:
            r = 2-u 
        else: 
            r = u

        a = r*np.cos(t)*radius
        b = r*np.sin(t)*radius

        dot = np.random.randint(0, len(centroids))

        if dot == 0 or dot == 3:
            yvals.append(0)
        else:
            yvals.append(1)


        x = centroids[dot][0]+a -0.5
        y = centroids[dot][1]+b -0.5

        Xvals.append([x, y])
        
    return np.array(Xvals), np.array(yvals)


def zwei_b(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    centroids = [[0.2,0.2],[0.5,0.2],[0.8,0.2],[0.2,0.5],[0.5,0.5],[0.8,0.5],[0.2,0.8],[0.5,0.8],[0.8,0.8]]

    radius = 0.12
    Xvals, yvals = [], []

    for n in range(samples):

        t = 2*np.pi*np.random.random()
        u = np.random.random()+np.random.random()
        if u>1:
            r = 2-u 
        else: 
            r = u

        a = r*np.cos(t)*radius
        b = r*np.sin(t)*radius
        dot = np.random.randint(0, len(centroids))


        if dot%2 == 0:
            yvals.append(0)
        else:
            yvals.append(1)

        x = centroids[dot][0]+a -0.5
        y = centroids[dot][1]+b -0.5

        Xvals.append([x, y])
        
    return np.array(Xvals), np.array(yvals)



def zwei_c(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    """
    Generates a dataset of points with 1/0 labels inside a given radius.

    Args:
        samples (int): number of samples to generate
        center (tuple): center of the circle
        radius (float: radius of the circle

    Returns:
        Xvals (array[tuple]): coordinates of points
        yvals (array[int]): classification labels
    """
    Xvals, yvals = [], []

    cols = 4
    rows = 4
    centroids = np.zeros(shape=(cols * rows, 2))
    start = [0.2, 0.2]
    inc = 0.2
    inp = []
    labelz = []
    count = 0
    for c in range(0, cols):
        for r in range(0, rows):
            centroids[count][0] = start[0] + r * inc
            centroids[count][1] = start[1] + c * inc
            count += 1

    radius = 0.08

    for n in range(samples):

        t = 2 * np.pi * np.random.random()
        u = np.random.random() + np.random.random()
        if u > 1:
            r = 2 - u
        else:
            r = u

        a = r * np.cos(t) * radius
        b = r * np.sin(t) * radius
        dot = np.random.randint(0, len(centroids))
        label = -1
        if dot in [0, 2, 5, 7, 8, 10, 13, 15]:
            yvals.append(0)

        else:
            yvals.append(1)

        x = centroids[dot][0] + a -0.5
        y = centroids[dot][1] + b -0.5

        Xvals.append([x, y])

    return np.array(Xvals), np.array(yvals)




def drei_a(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    
    Xvals, yvals = [], []
    
    for n in range(samples):
        x = np.random.uniform(0, 1)-0.5
        y = np.random.uniform(0, 1)-0.5
        
        if np.sqrt(x*x+y*y) <= 0.2:
            yvals.append(0)
        else:
            yvals.append(1)
            
        Xvals.append([x, y])
        
    return np.array(Xvals), np.array(yvals)



def drei_b(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    
    Xvals, yvals = [], []

    
    for n in range(samples):

        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)

        x=a-0.25
        y=b-0.25

        x1=a-0.75
        y1=b-0.75

        if np.sqrt(x*x+y*y) <= 0.15:
            yvals.append(0)


        elif np.sqrt(x1*x1+y1*y1) <= 0.15:   
            yvals.append(0)

        else:
            yvals.append(1)

        Xvals.append([a-0.5, b-0.5])
        
        
    return np.array(Xvals), np.array(yvals)



def drei_c(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)):
    Xvals, yvals = [], []

    for n in range(samples):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)

        x=a-0.25
        y=b-0.25

        x1=a-0.75
        y1=b-0.75

        x2=a-0.25
        y2=b-0.75

        x3=a-0.75
        y3=b-0.25

        if np.sqrt(x*x+y*y) <= 0.15:
            yvals.append(0)

        elif np.sqrt(x1*x1+y1*y1) <= 0.15:   
            yvals.append(0)

        elif np.sqrt(x2*x2+y2*y2) <= 0.15:   
            yvals.append(0)

        elif np.sqrt(x3*x3+y3*y3) <= 0.15:   
            yvals.append(0)

        else:
            yvals.append(1)
            
        Xvals.append([a-0.5, b-0.5])
        
        
    return np.array(Xvals), np.array(yvals)

