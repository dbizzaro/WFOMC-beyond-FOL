# -*- coding: utf-8 -*-

from wfomc import *
import combinatorial_experiments
import timeit
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import argparse


def plot_times(list_functions, maximum_n, list_names=None):
    if list_names is None:
        list_names = list_functions
    for i, funct_name in enumerate(list_functions):
        n_list, times_list = [], []
        with open('results/{}.csv'.format(funct_name), 'r') as f:
            for line in f:
                info = line.strip().split(',')
                n_list.append(int(info[0]))
                times_list.append(float(info[1]))
        n_list = n_list[1:maximum_n+1]
        times_list = times_list[1:maximum_n+1]
        plt.plot(n_list, times_list, label=list_names[i])
        #print(n_list, times_list)
    plt.xlabel('number of nodes', fontsize = 'large')
    plt.ylabel('time (s)', fontsize = 'large')
    plt.xticks(fontsize = 'large')
    plt.yticks(fontsize = 'large')
    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.yscale("log")
    if len(list_functions)>1:
        plt.legend(fontsize='medium')
    plt.tight_layout()
    plt.savefig('plots/log_{}.png'.format('_'.join(list_functions)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions', default=['count_DAGS_edges',
                                                'count_DAGs_edges_one_source',
                                                'count_DAGs_one_source',
                                                'count_DAGs_one_source_one_sink',
                                                'count_trees',
                                                'count_3_colored_connected_graphs',
                                                'count_forests_without_isolated_vertices'],
                        nargs = '+', type=str, help='functions to plot')
    parser.add_argument('--maximum_n', default=150, type=int, help='maximum n')
    parser.add_argument('--names', default=['DAGs with 2n edges (ex. 5)',
                                           'DAGs with 2n edges and one source (ex. 6)',
                                           'DAGs with one source (ex. 6)',
                                           'DAGs with one source and one sink (ex. 6)',
                                           'trees (ex. 8)',
                                           '3-colored connected graphs (ex. 9)',
                                           'forests without isolated vertices (ex. 10)'],
                        nargs='+', type=str, help='names in the legend')
    #                                   default=['DAGs with 2n edges (ex. 4)',
    #                                             'DAGs with 2n edges and one source (ex. 5)',
    #                                             'DAGs with one source (ex. 5)',
    #                                             'DAGs with one source and one sink (ex. 5)',
    #                                             'trees (ex. 7)',
    #                                             '3-colored connected graphs (ex. 8)',
    #                                             'forests without isolated vertices (ex. 9)'],
    args = parser.parse_args()
    if len(args.functions) > 1:
        plot_times(args.functions, args.maximum_n, args.names)
    else:
        plot_times(args.functions, args.maximum_n)
