# -*- coding: utf-8 -*-

from wfomc import *
from collections import Counter
from math import factorial
import experiments_solutions
import argparse
from inspect import signature

def count_smoker_example(n):
  language = Language(['S'], ['F'])
  formula = Formula('(Sx & Fxy) >> Sy')
  return WFOMC_no_axiom(language, formula, n)

def count_DAGs(n):
  language = Language(['P'], ['R'])
  formula = Formula('Px', 'DAG', 'R')
  return WFOMC(language, formula, n)

def count_DAGs_one_source(n):
  language = Language(['I', 'P'], ['R'], {'P':(1,-1)})
  formula = Formula('(Ix >> ~ Ryx) & (Px | ~ Ryx) & (Px | ~ Ix)', 'DAG', 'R', {'Ix': 1})
  return WFOMC(language, formula, n)

def count_DAGs_one_source_one_sink(n):
  language = Language(['I', 'T', 'P', 'Q'], ['R'], {'P':(1,-1), 'Q':(1,-1)})
  formula = Formula('(Ix >> ~ Ryx) & (Px | ~ Ryx) & (Px | ~ Ix) & (Tx >> ~ Rxy) & (Qx | ~ Rxy) & (Qx | ~ Tx)', 'DAG', 'R', {'Ix': 1, 'Tx':1})
  return WFOMC(language, formula, n)

def count_DAGs_two_sources(n):
  language = Language(['I', 'P'], ['R'], {'P':(1,-1)})
  formula = Formula('(Ix >> ~ Ryx) & (Px | ~ Ryx) & (Px | ~ Ix)', 'DAG', 'R', {'Ix': 2})
  return WFOMC(language, formula, n)

def count_DAGs_k_edges(n, k):
  language = Language(['P'], ['R'])
  formula = Formula('Px', 'DAG', 'R', cardinality_constraints_binary={'R': ('=', k)})
  return WFOMC(language, formula, n)

def count_DAGs_edges(n):
  return count_DAGs_k_edges(n, 2*n)

def count_DAGs_k_edges_one_source(n, k):
  language = Language(['I', 'P'], ['R'], {'P':(1,-1)})
  formula = Formula('(Ix >> ~ Ryx) & (Px | ~ Ryx) & (Px | ~ Ix)', 'DAG', 'R', {'Ix': 1}, {'R': ('=', k)})
  return WFOMC(language, formula, n)

def count_DAGs_edges_one_source(n):
  return count_DAGs_k_edges_one_source(n, 2*n)

def count_connected_graphs(n):
  language = Language(['P'], ['R'])
  formula = Formula('Px', 'connected', 'R')
  return WFOMC(language, formula, n)

def count_3_colored_connected_graphs(n):
  language = Language(['V', 'G', 'B'], ['R'])
  formula = Formula('((Vx & Rxy) >> ~ Vy) & ((Gx & Rxy) >> ~ Gy) & ((Bx & Rxy) >> ~ By) & (Vx | Gx | Bx) & (~(Vx & Gx)) & (~(Vx & Bx)) & (~(Gx & Bx))', 'connected', 'R')
  return WFOMC(language, formula, n)

def count_connected_graphs_d_edges(n, d):
  language = Language(['P'], ['R'])
  formula = Formula('Px', 'connected', 'R', cardinality_constraints_binary={'R':('=', 2*d)})
  return WFOMC(language, formula, n)

def sized_partitions(n, k, m = None):
  """Partition n into k parts with a max part of m.
  Yield non-increasing lists.  m not needed to create generator.
  https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning
  """
  if k == 1:
    yield [n]
    return
  for f in range(n-k+1 if (m is None or m > n-k+1) else m, (n-1)//k, -1):
    for p in sized_partitions(n-f, k-1, f): yield [f] + p

def count_graphs_k_connected_components(n,k):
  language = Language([], ['R'])
  formula = Formula('Rxy | ~ Rxy', 'connected', 'R')
  sum_ = 0
  for partition in sized_partitions(n, k):
    prod_ = multinomial_coefficient(partition)
    for subset_size, count in Counter(partition).items():
      prod_ *= WFOMC(language, formula, subset_size) ** count
      prod_ = prod_ // factorial(count)
    sum_ += prod_
  return sum_

def count_graphs_k_edges(n, k):
  language = Language([], ['R'])
  formula = Formula('(~ Rxx) & (Rxy >> Ryx)', axiom_predicate = 'R', cardinality_constraints_binary = {'R': ('=', 2*k)})
  return WFOMC(language, formula, n)

def count_trees(n):
  language = Language([], ['R'])
  formula = Formula('Rxy | ~ Rxy', 'tree', 'R')
  return WFOMC(language, formula, n)

def count_forests(n):
  language = Language([], ['R'])
  formula = Formula('Rxy | ~ Rxy', 'forest', 'R')
  return WFOMC(language, formula, n)

def count_forests_without_isolated_vertices(n):
  language = Language(['S'], ['R'], {'S':(1,-1)})
  formula = Formula('Sx | (~Rxy)', 'forest', 'R')
  return WFOMC(language, formula, n)


def count_forests_k_edges(n, k):
  language = Language([], ['R'])
  formula = Formula('Rxy | ~ Rxy', 'forest', 'R', cardinality_constraints_binary={'R': ('=', 2*k)})
  return WFOMC(language, formula, n)

def count_weakly_connected_directed_graphs(n):
  language = Language([], ['S', 'R'])
  formula = Formula('(Sxy >> (Rxy | Ryx)) & ((Rxy | Ryx) >> Sxy)', 'connected', 'S')
  return WFOMC(language, formula, n)

def count_weakly_connected_oriented_graphs(n):
  language = Language([], ['S', 'R'])
  formula = Formula('(Sxy >> (Rxy | Ryx)) & ((Rxy | Ryx) >> Sxy) & ~(Rxy & Ryx)', 'connected', 'S')
  return WFOMC(language, formula, n)


def count_graphs_edges(n):
  language = Language([], ['R'])
  k =  (n*(n-1))//2
  formula = Formula('(Rxy >> Ryx) & ~Rxx', cardinality_constraints_binary={'R': ('=', k)})
  return WFOMC(language, formula, n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', default='DAGs', type=str, help='function to be tested')
    parser.add_argument('--starting_n', default=1, type=int, help='starting n')
    parser.add_argument('--last_n', default=10, type=int, help='last n')
    parser.add_argument('--profiler', action='store_true', help='use profiler instead of checking correctness')
    parser.add_argument('--extended', action='store_true', help='profiler on all functions, not only the wfomc ones')
    #parser.add_argument('--k', default=-1, type=int, help='d or k')
    args = parser.parse_args()
    count_function = locals()['count_'+args.function]
    if args.profiler:
      from cProfile import Profile
      from pstats import SortKey, Stats
      with Profile() as profile:
        print(f"{count_function(args.last_n)}")
        if args.extended:
          Stats(profile).strip_dirs().sort_stats("tottime").print_stats()
        else:
          Stats(profile).strip_dirs().sort_stats("tottime").print_stats("wfomc")
    else:
      for n in range(args.starting_n, args.last_n+1):
        solution_function = getattr(experiments_solutions, 'solution_'+args.function)
        try:
          if len(signature(count_function).parameters) == 1:
            print(n, count_function(n), solution_function(n)) 
          else:
            print(n, [count_function(n, k) for k in range(1,n)],  [solution_function(n, k) for k in range(1,n)])
        except:
          break

