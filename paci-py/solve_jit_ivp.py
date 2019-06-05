from scipy.integrate import ode, solve_ivp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from paci_jit_generator import wrapper

import pdb

from csv import writer
from io import StringIO

import time

from jitcode import jitcode, y

Y_names = ['Vm', 'Ca_SR', 'Cai', 'g', 'd', 'f1', 'f2',
           'fCa', 'Xr1', 'Xr2', 'Xs', 'h', 'j', 'm',
           'Xf', 'q', 'r', 'Nai', 'm_L', 'h_L',
           'RyRa', 'RyRo', 'RyRc']
Y_units = ['V', 'mM', 'mM', '-', '-', '-', '-',
           '-', '-', '-', '-', '-', '-', '-',
           '-', '-', '-', 'mM', '-', '-',
           '-', '-', '-']
Y_names_units = {'Names': Y_names, 'Units': Y_units}
Y_names_units = pd.DataFrame(data=Y_names_units)

# SS originale
'''
Y = np.array([-0.070, 0.32, 0.0002,
              0, 0, 1,
              1, 1, 0,
              1, 0, 0.75,
              0.75, 0, 0.1,
              1, 0, 9.2,
              0, 0.75, 0.3,
              0.9, 0.1])
'''

# SS a 800
Y0 = [-0.0749228904740065, 0.0936532528714175, 3.79675694306440e-05,
              0, 8.25220533963093e-05, 0.741143500777858,
              0.999983958619179, 0.997742015033076, 0.266113517200784,
              0.434907203275640, 0.0314334976383401, 0.745356534740988,
              0.0760523580322096, 0.0995891726023512, 0.0249102482276486,
              0.841714924246004, 0.00558005376429710, 8.64821066193476,
              0.00225383437957339, 0.0811507312565017, 0.0387066722172937,
              0.0260449185736275, 0.0785849084330126]

# Current blockers
tDrugApplication = 10000
INaFRedMed = 1
ICaLRedMed = 1
IKrRedMed = 1
IKsRedMed = 1

tic = time.time()





t_span = [0, 6]





x = wrapper()

system_of_ODEs = wrapper()
ODE = jitcode(system_of_ODEs)



import csv
with open('ode_system.py')

for i in range(0, len(system_of_ODEs)):
  print(system_of_ODEs)



pdb.set_trace()

ODE.set_integrator('vode')
ODE.set_initial_value(Y0)
# ODE.set_integrator('BDF', max_step=1e-3)

times = np.arange(0, 30, .001)
data = np.zeros((len(times),23))

tic = time.time()
i = 0

for t in times:
  data[i,:] = ODE.integrate(t)
  i+=1

# output.seek(0)
# sol = pd.read_csv(output)

# sol = jitcode.solve_ivp(wrapper, t_span, Y0, method='BDF', max_step=1e-3)
elapsed = time.time() - tic
print(elapsed)



plt.plot(times, data[:,0])
plt.show()

# plt.plot(times, data[:,0])
# plt.show()
pdb.set_trace()
np.savetxt("../pythonSolutionJit.csv", data, delimiter=",")
np.savetxt("../pySolutionTimeJit.csv", times, delimiter=",")