# -*- coding: utf-8 -*-

from wfomc import *
import combinatorial_experiments
import timeit
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import argparse

def measure_time(function_name, time_limit = 100, starting_n = 1, stopping_n = 100):
  times = []
  n_list = []
  measured_time = 0
  n = starting_n
  while (measured_time < time_limit) and (n <= stopping_n):
    funct = getattr(combinatorial_experiments, function_name)
    start_time = timeit.default_timer()
    _ = funct(n)
    measured_time = timeit.default_timer()-start_time
    times.append(measured_time)
    n_list.append(n)
    print(n, measured_time)
    n += 1
  return times, n_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default='count_smoker_example', type=str, help='function to be timed')
    parser.add_argument('--time_limit', default=100, type=float, help='time limit')
    parser.add_argument('--starting_n', default=1, type=int, help='starting n')
    parser.add_argument('--last_n', default=100, type=int, help='stopping n')
    args = parser.parse_args()
    times_list, n_list = measure_time(args.function, args.time_limit, args.starting_n, args.last_n)
    with open('results/{}.csv'.format(args.function), 'a') as f:
      for i, t in enumerate(times_list):
        f.write("{},{:.3f}\n".format(i+args.starting_n, t))
    #plt.plot(n_list, times_list)
    #plt.xlabel('n')
    #plt.ylabel('time (s)')
    #plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    #plt.savefig('plots/{}_{}.png'.format(args.function, args.time_limit))
    #plt.show()