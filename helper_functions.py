import numpy as np
from matplotlib import pyplot as plt
import pdb

def sigmoid_generator(t_step, shift, sign):
    y = sign*1/(1+np.exp(-(t_step-shift)*10000))
    return y

t_range = np.arange(-5, 5, .0001)

plt.plot(t_range,sigmoid_generator(t_range, 2, -1))
plt.show()