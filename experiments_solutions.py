# -*- coding: utf-8 -*-

from wfomc import *
from collections import Counter
from math import factorial


def solution_smoker_example(n):
  sum = 0
  for k in range(n+1):
    sum += comb(n, k, exact=True) * (2 **(n**2 - k*(n-k)))
  return sum

def solution_DAGs(n):
  A = [1]
  for i in range(1, n+1):
    sum_ = 0
    for l in range(i):
      sum_ += (-1)**(i-l+1) * comb(i, l, exact=True) * (2**(l*(i-l))) * A[l]
    A.append(sum_)
  return A[n]

def solution_DAGs_one_source_one_sink(n):
  solutions = [1, 2, 12, 216, 10600, 1306620, 384471444, 261548825328, 402632012394000,
               1381332938730123060, 10440873023366019273820, 172308823347127690038311496,
               6163501139185639837183141411320, 474942255590583211554917995123517868,
               78430816994991932467786587093292327531620]
  return solutions[n-1]

def solution_DAGs_one_source(n):
  solutions = [1, 2, 15, 316, 16885, 2174586, 654313415, 450179768312, 696979588034313,
               2398044825254021110, 18151895792052235541515, 299782788128536523836784628,
               10727139906233315197412684689421]
  return solutions[n-1]

def solution_DAGs_two_sources(n):
  #https://oeis.org/A003026
  solutions = [1, 9, 198, 10710, 1384335, 416990763, 286992935964, 444374705175516,
               1528973599758889005, 11573608032229769067465, 191141381932394665770442818,
               6839625961762363728765713227698]
  return solutions[n-2]

def solution_DAGs_k_sources(n, k):
  #https://oeis.org/A361718
  solutions = [1, 0, 1, 0, 2, 1, 0, 15, 9, 1, 0, 316, 198, 28, 1, 0, 16885, 10710,
               1610, 75, 1, 0, 2174586, 1384335, 211820, 10575, 186, 1, 0, 654313415,
               416990763, 64144675, 3268125, 61845, 441, 1, 0, 450179768312, 286992935964,
               44218682312, 2266772550, 43832264, 336924, 1016, 1]
  return solutions[n*(n+1)//2 + k]

def solution_DAGs_k_edges(n, k):
  #https://oeis.org/A081064
  solutions = [1, 1, 1, 2, 1, 6, 12, 6, 1, 12, 60, 152, 186, 108, 24, 1, 20, 180,
               940, 3050, 6180, 7960, 6540, 3330, 960, 120, 1, 30, 420, 3600, 20790, 83952,
               240480, 496680, 750810, 838130, 691020, 416160, 178230, 51480, 9000, 720,
               1, 42, 840, 10570, 93030, 601944]
  if k > n*(n-1)//2:
    return 0
  sum_ = 0
  for n_prime in range(n):
    sum_ += n_prime * (n_prime-1) //2  +1
  return solutions[sum_ + k]

def solution_DAGs_edges(n):
  return solution_DAGs_k_edges(n, 2*n)

def solution_DAGs_k_edges_one_source(n, k):
  #https://oeis.org/A350487
  solutions = [	1, 0, 2, 0, 0, 9, 6, 0, 0, 0, 64, 132, 96, 24, 0, 0, 0, 0,
                625, 2640, 4850, 4900, 2850, 900, 120, 0, 0, 0, 0, 0, 7776, 55800,
                186480, 379170, 516660, 491040, 328680, 152640, 46980, 8640, 720,
                0, 0, 0, 0, 0, 0, 117649, 1286670, 6756120, 22466010]
  if k > n*(n-1)//2:
    return 0
  sum_ = 0
  for n_prime in range(1, n):
    sum_ += n_prime * (n_prime-1) //2  +1
  return solutions[sum_ + k]

def solution_DAGs_edges_one_source(n):
  return solution_DAGs_k_edges_one_source(n, 2*n)

def solution_connected_graphs(n):
  #https://oeis.org/A001187
  solutions = [1, 1, 1, 4, 38, 728, 26704, 1866256, 251548592, 66296291072, 34496488594816,
               35641657548953344, 73354596206766622208, 301272202649664088951808, 2471648811030443735290891264,
               40527680937730480234609755344896, 1328578958335783201008338986845427712]
  return solutions[n]

def solution_3_colored_connected_graphs(n):
  #https://oeis.org/A322279
  #https://oeis.org/A002028
  solutions = [1, 3, 6, 42, 618, 15990, 668526, 43558242, 4373213298, 677307561630,
               162826875512646, 61183069270120842, 36134310487980825258, 33673533885068169649830,
               49646105434209446798290206, 116002075479856331220877149042, 430053223599741677879550609246498,
               2531493110297317758855120762121050990]
  return solutions[n]

def solution_connected_graphs_d_edges(n, d):
  #https://oeis.org/A062734
  solutions = [1, 0, 1, 0, 0, 3, 1, 0, 0, 0, 16, 15, 6, 1, 0, 0, 0, 0, 125, 222, 205,
               120, 45, 10, 1, 0, 0, 0, 0, 0, 1296, 3660, 5700, 6165, 4945, 2997, 1365,
               455, 105, 15, 1, 0, 0, 0, 0, 0, 0, 16807, 68295, 156555, 258125, 331506,
               343140, 290745, 202755, 116175, 54257, 20349]
  sum_ = 0
  for n_prime in range(1, n):
    sum_ += n_prime * (n_prime-1) //2  +1
  return solutions[sum_ + d]

def solution_graphs_k_connected_components(n, k):
  #https://oeis.org/A143543
  sol_list = [1, 1, 1, 4, 3, 1, 38, 19, 6, 1, 728, 230, 55, 10, 1, 26704, 5098,
              825, 125, 15, 1, 1866256, 207536, 20818, 2275, 245, 21, 1, 251548592,
              15891372, 925036, 64673, 5320, 434, 28, 1, 66296291072, 2343580752,
              76321756, 3102204, 169113, 11088, 714, 36, 1]
  return sol_list[n*(n-1)//2 + k - 1]

def solution_graphs_k_edges(n, k):
  #https://oeis.org/A084546
  return comb(n*(n-1)//2, k, exact = True)

def solution_trees(n):
  return int(n ** (n-2))

def solution_forests(n):
  #https://oeis.org/A001858
  if n <= 1:
    return 1
  sum_ = 0
  for k in range(1, n+1):
    sum_ += int(comb(n-1, k-1, exact = True) * (k ** (k-2))) * solution_forests(n-k)
  return sum_

def solution_forests_without_isolated_vertices(n):
  #https://oeis.org/A105784
  solutions = [0, 1, 3, 19, 155, 1641, 21427, 334377, 6085683, 126745435, 2975448641,
               77779634571, 2241339267037, 70604384569005, 2414086713172695, 89049201691604881,
               3525160713653081279, 149075374211881719939, 6707440248292609651513, 319946143503599791200675]
  return solutions[n-1]


def solution_weakly_connected_directed_graphs(n):
  #https://oeis.org/A003027
  solutions = [1, 3, 54, 3834, 1027080, 1067308488, 4390480193904, 72022346388181584,
               4721717643249254751360, 1237892809110149882059440768, 1298060596773261804821355107253504,
               5444502293680983802677246555274553481984, 91343781554246596956424128384394531707099632640]
  return solutions[n-1]


def solution_weakly_connected_oriented_graphs(n):
  #https://oeis.org/A054941
  solutions = [1, 2, 20, 624, 55248, 13982208, 10358360640, 22792648882176,
               149888345786341632, 2952810709943411146752, 174416705255313941476193280,
               30901060796613886817249881227264, 16422801513633911416125344647746244608,
               26183660776604240464418800095675915958222848]
  return solutions[n-1]


def solution_forests_k_edges(n, k):
  #https://oeis.org/A138464
  solutions = [1, 
               1, 1, 
               1, 3, 3, 
               1, 6, 15, 16, 
               1, 10, 45, 110, 125, 
               1, 15, 105, 435, 1080, 1296, 
               1, 21, 210, 1295, 5250, 13377, 16807, 
               1, 28, 378, 3220, 18865, 76608, 200704, 262144, 
               1, 36, 630, 7056, 55755, 320544, 1316574, 3542940, 4782969, 
               1, 45, 990, 14070, 143325, 1092105, 6258000, 26100000, 72000000, 100000000]
  return solutions[n*(n-1)//2 + k - 1]