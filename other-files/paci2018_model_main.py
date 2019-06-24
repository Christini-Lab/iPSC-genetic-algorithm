"""Runs a Paci2018 model."""
import argparse
import time

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import Paci2018
from scipy import integrate

parser = argparse.ArgumentParser()

# Duration, in seconds, of integration.
parser.add_argument('duration', action='store', type=int)

Y_NAMES = ['Vm', 'Ca_SR', 'Cai', 'g', 'd', 'f1', 'f2',
           'fCa', 'Xr1', 'Xr2', 'Xs', 'h', 'j', 'm',
           'Xf', 'q', 'r', 'Nai', 'm_L', 'h_L',
           'RyRa', 'RyRo', 'RyRc']
Y_UNITS = ['V', 'mM', 'mM', '-', '-', '-', '-',
           '-', '-', '-', '-', '-', '-', '-',
           '-', '-', '-', 'mM', '-', '-',
           '-', '-', '-']
Y_NAMES_UNITS_DF = pd.DataFrame(data={'Names': Y_NAMES, 'Units': Y_UNITS})
Y_INITIAL = [-0.0749228904740065, 0.0936532528714175, 3.79675694306440e-05,
              0, 8.25220533963093e-05, 0.741143500777858,
              0.999983958619179, 0.997742015033076, 0.266113517200784,
              0.434907203275640, 0.0314334976383401, 0.745356534740988,
              0.0760523580322096, 0.0995891726023512, 0.0249102482276486,
              0.841714924246004, 0.00558005376429710, 8.64821066193476,
              0.00225383437957339, 0.0811507312565017, 0.0387066722172937,
              0.0260449185736275, 0.0785849084330126]


def main():
  args = parser.parse_args()

  start_time = time.time()
  solution = integrate.solve_ivp(
    Paci2018.wrapper,
    [0, args.duration],
    Y_INITIAL,
    method='BDF',
    max_step=1e-3)

  seconds_elapsed = time.time() - start_time
  print('Seconds elapsed: {}'.format(seconds_elapsed))
  print(solution.t)
  print(solution.y[0])
  plt.plot(solution.t, solution.y[0])
  plt.show()

if __name__ == '__main__':
  main()