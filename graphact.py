import numpy as np
import matplotlib.pyplot as plt
import math
t = np.arange(-10., 10., .1)
print(math.e)
def sigmoid(x):
    return 1/(1+math.e**(-x))
def interpl(x,alpha,beta):
    y=alpha*x*sigmoid(x)+beta*x*(1-sigmoid(x))
    return y
plt.plot(t,interpl(t,10,1))
plt.show()
