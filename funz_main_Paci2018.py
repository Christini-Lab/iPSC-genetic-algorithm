from scipy.integrate import ode
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Paci2018 import Paci2018

from csv import writer
from io import StringIO

import pdb
import time



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

# Set initial conditions
t0 = 0
t1 = .5

# Current blockers
tDrugApplication = 10000
INaFRedMed = 1
ICaLRedMed = 1
IKrRedMed = 1
IKsRedMed = 1

# Call function with ODE solver
# solver = ode(lambda t,y: Paci2018(t, y, tDrugApplication,
#                                              INaFRedMed,
#                                              ICaLRedMed,
#                                              IKrRedMed,
#                                              IKsRedMed)).set_integrator('vode', method='bdf',
#                                               max_step=1e-3,
#                                               first_step=2e-5) #could set order=15
# Five Seconds, 1739
# solver = ode(Paci2018).set_integrator('vode', method='bdf',
#                                               max_step=1e-3,
                                              # first_step=2e-5) #could set order=15
# 4.62 seconds, 1739


solver = ode(Paci2018).set_integrator('vode', method='bdf',
                                              max_step=1e-2,
                                              min_step=1e-8,
                                              first_step=2e-4) #could set order=15

# solver.set_solout(solout)
solver.set_f_params(tDrugApplication,
                    INaFRedMed,
                    ICaLRedMed,
                    IKrRedMed,
                    IKsRedMed)
solver.set_jac_params(tDrugApplication,
                    INaFRedMed,
                    ICaLRedMed,
                    IKrRedMed,
                    IKsRedMed)
solver.set_initial_value(Y0, t0)

a = np.empty(shape=(100000,23))
sol = pd.DataFrame(columns=Y_names_units['Names'])
t_sol = np.empty(100000)
i = 0 

output = StringIO()
csv_writer = writer(output)

tic = time.time()
while solver.successful() and solver.t < t1:
    solver.integrate(t1, step=True)
    csv_writer.writerow([solver.t, solver.y])
    print(solver.t)
    

elapsed = time.time() - tic
print(elapsed)
print(sol.shape)

pdb.set_trace()
output.seek(0)
sol = pd.read_csv(output)


#solver.integrate(2e-5, step=True)

# At this point, I should have everything stored in sol and t_sol

plt.plot(sol.ix[:,0])






# function to be called after each successful integration



# def solout(t, y):
#     sol.loc[sol.shape[0]] = y
#     t_sol = t_sol.append(t)

# scipy.integrate.ode(f).set_integrator('vode', method='bdf', order=15)