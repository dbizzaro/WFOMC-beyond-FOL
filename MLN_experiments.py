from wfomc import *
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import symengine
import sympy
import numpy as np
import timeit
import argparse


class MLN_experiment: 
    def __init__(self, experiment_name, weights, predicates_to_rescale = [], cardinality_constraints_binary = {}, n_digits = 3):
        self.experiment_name = experiment_name
        self.weights = weights
        self.predicates_to_rescale = predicates_to_rescale
        self.cardinality_constraints_binary = cardinality_constraints_binary
        self.n_digits = n_digits
        
        if experiment_name == 'DAG-edges':
            self.unary_predicates = []
            self.binary_predicates = ['R']
            self.expression = "~Rxx & (Rxy >> ~Ryx)"
            self.axiom = 'DAG'
            self.axiom_predicate = 'R'
            self.query_predicate = 'R'
        elif experiment_name == 'DAG-edges-smokers':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> ~Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'DAG'
            self.axiom_predicate = 'F'
            self.query_predicate = 'F'
        elif experiment_name == 'hard-DAG':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> ~Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'DAG'
            self.axiom_predicate = 'F'
            self.query_predicate = 'S'
        elif experiment_name == 'connected-edges':
            self.unary_predicates = []
            self.binary_predicates = ['R']
            self.expression = "~Rxx & (Rxy >> Ryx)"
            self.axiom = 'connected'
            self.axiom_predicate = 'R'
            self.query_predicate = 'R'
        elif experiment_name == 'connected-edges-smokers':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'connected'
            self.axiom_predicate = 'F'
            self.query_predicate = 'F'
        elif experiment_name == 'hard-connected':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'connected'
            self.axiom_predicate = 'F'
            self.query_predicate = 'S'
        elif experiment_name == 'soft-connected':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P', 'W', 'Z']
            self.expression = "~Fxx & (Fxy >> Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy)) & \
                                (Zxy >> ((Wxy >> Fxy) & (Wxy << Fxy))) & \
                                (Zxy << ((Wxy >> Fxy) & (Wxy << Fxy)))"
            self.axiom = 'connected'
            self.axiom_predicate = 'W'
            self.query_predicate = 'S'
        elif experiment_name == 'forest-edges':
            self.unary_predicates = []
            self.binary_predicates = ['R']
            self.expression = "~Rxx & (Rxy >> Ryx)"
            self.axiom = 'forest'
            self.axiom_predicate = 'R'
            self.query_predicate = 'R'
        elif experiment_name == 'forest-edges-smokers':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'forest'
            self.axiom_predicate = 'F'
            self.query_predicate = 'F'
        elif experiment_name == 'hard-forest':
            self.unary_predicates = ['S']
            self.binary_predicates = ['F', 'P']
            self.expression = "~Fxx & (Fxy >> Fyx) & \
                                (Pxy >> ((Fxy & Sx) >> Sy)) & \
                                (Pxy << ((Fxy & Sx) >> Sy))"
            self.axiom = 'forest'
            self.axiom_predicate = 'F'
            self.query_predicate = 'S'

    def without_axiom(self, cardinality_constraints_binary = {}):
        if cardinality_constraints_binary != {}:
            experiment = MLN_experiment(self.experiment_name, self.weights, self.predicates_to_rescale, cardinality_constraints_binary, self.n_digits)
        else:
            experiment = MLN_experiment(self.experiment_name, self.weights, self.predicates_to_rescale, self.cardinality_constraints_binary, self.n_digits)
        experiment.axiom = ''
        return experiment

    def produce_MLN_formula(self, expression, auxiliary_predicate, wmc_weight):
        new_object = MLN_experiment(self.experiment_name, self.weights, self.predicates_to_rescale, self.cardinality_constraints_binary, self.n_digits)
        new_object.expression += ' & ((' + expression + ') >> ' + auxiliary_predicate + 'xy) & (' + auxiliary_predicate + 'xy >> (' + expression + '))'
        new_object.weights[auxiliary_predicate] = wmc_weight
        new_object.binary_predicates.append(auxiliary_predicate)
        return new_object


    def with_soft_constraint(self, auxiliary_1 = 'Z', auxiliary_2 = 'W', weight_1 = (3,0), weight_2 = (0,0)):
        new_object = self.produce_MLN_formula(f'({auxiliary_2}xy >> {self.query_predicate}xy) & ({auxiliary_2}xy << {self.query_predicate}xy)', auxiliary_1, weight_1)
        if weight_2 != None:
            new_object.weights[auxiliary_2] = weight_2
        else:
            new_object.weights[auxiliary_2] = new_object.weights[new_object.query_predicate]
        new_object.binary_predicates.append(auxiliary_2)
        new_object.axiom_predicate = auxiliary_2
        return new_object

    
    
    def compute_weights(self, n):
        def approximate_weight(value, n_digits):
            fraction = sympy.Rational(sympy.exp(value).evalf(n_digits))
            return symengine.Rational(fraction.numerator, fraction.denominator)  
    
        
        self.rescaled_weights = self.weights.copy()
        for predicate in self.predicates_to_rescale:
            self.rescaled_weights[predicate] = (self.weights[predicate][0]/n, self.weights[predicate][0]/n)
        #self.exponentiated_weights = {k: (exp(v[0]), exp(v[1])) for k, v in self.rescaled_weights.items()}
        #fractions assure that we don't run into numerical precision (or overflow) issues (at the cost of more computation)
        self.exponentiated_weights = {k: (approximate_weight(v[0], self.n_digits), approximate_weight(v[1], self.n_digits)) \
                                      for k, v in self.rescaled_weights.items()}

    
    def unary_query_experiment(self, n):
        self.compute_weights(n)
        values_list = []
        language = Language(self.unary_predicates, self.binary_predicates, weights = self.exponentiated_weights)
        formula = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=self.cardinality_constraints_binary)
        #formula_axiom = Formula(self.expression, self.axiom, self.axiom_predicate) #cardinality_constraints_binary=self.cardinality_constraints_binary)
        total = WFOMC(language, formula, n)
        #total_axiom = WFOMC(language, formula_axiom, n)
        for cardinality_query in range(n+1):
            print(cardinality_query)
            formula_query = Formula(self.expression, self.axiom, self.axiom_predicate, {self.query_predicate+'x': cardinality_query}, self.cardinality_constraints_binary)
            #formula_query_axiom = Formula(self.expression, self.axiom, self.axiom_predicate, {self.query_predicate+'x': cardinality_query}) #cardinality_constraints_binary=self.cardinality_constraints_binary)
            value = float(sympy.Rational(WFOMC(language, formula_query, n)/total))
            #value_axiom = float(sympy.Rational(WFOMC(language, formula_query_axiom, n)/total_axiom))
            values_list.append(value)
        return values_list
        #values_array = np.array(values_list)
        #np.save(f'results/{self.experiment_name}_{n}_{self.weights}_{self.predicates_to_rescale}_{self.cardinality_constraints_binary}.npy', values_array)
        #self.bar_plot(values_array)
        #return values_array
    

    def binary_query_experiment(self, n, max_cardinality = None, only_even = False):
        self.compute_weights(n)
        values_list = []
        language = Language(self.unary_predicates, self.binary_predicates, weights = self.exponentiated_weights)
        formula = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=self.cardinality_constraints_binary)
        #formula_axiom = Formula(self.expression, self.axiom, self.axiom_predicate) #cardinality_constraints_binary=self.cardinality_constraints_binary)
        total = WFOMC(language, formula, n)
        #total_axiom = WFOMC(language, formula_axiom, n)
        if max_cardinality is None:
            max_cardinality = n*(n-1)            
        for cardinality_query in range(max_cardinality+1):
            if (cardinality_query % 2 == 0) or not only_even: 
                if self.query_predicate not in self.cardinality_constraints_binary.keys() or eval(f"{cardinality_query} {self.cardinality_constraints_binary[self.query_predicate][0]} {self.cardinality_constraints_binary[self.query_predicate][1]}"):
                    if (self.axiom != 'forest' or cardinality_query < 2*n) and (self.axiom != 'connected' or cardinality_query > 2*n-3):
                        print(cardinality_query)
                        new_cardinality_constraints_binary = self.cardinality_constraints_binary.copy()
                        new_cardinality_constraints_binary[self.query_predicate] = ('=', cardinality_query)
                        formula_query = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=new_cardinality_constraints_binary)
                        #formula_query_axiom = Formula(self.expression, self.axiom, self.axiom_predicate, {self.query_predicate+'x': cardinality_query}) #cardinality_constraints_binary=self.cardinality_constraints_binary)
                        value = float(sympy.Rational(WFOMC(language, formula_query, n)/total))
                        #value_axiom = float(sympy.Rational(WFOMC(language, formula_query_axiom, n)/total_axiom))
                    else:
                        value = 0
                else:
                    value = 0
                values_list.append(value)
        return values_list
        #values_array = np.array(values_list)
        #np.save(f'results/{self.experiment_name}_{n}_{self.weights}_{self.predicates_to_rescale}_{self.cardinality_constraints_binary}.npy', values_array)
        #self.bar_plot(values_array)
        #return values_array
    
    def alternative_computation_experiment(self, n):
        # does not work when there are cardinality constraints on binary predicates!
        self.compute_weights(n)
        language = Language(self.unary_predicates, self.binary_predicates, weights = self.exponentiated_weights).set_symbolic_weight(self.query_predicate)
        formula = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=self.cardinality_constraints_binary)
        #formula_axiom = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=self.cardinality_constraints_binary)
        polynomial = WFOMC(language, formula, n)
        #polynomial_axiom = WFOMC(language, formula_axiom, n)      
        list0, sum_0 = assign_weight(self.query_predicate, self.exponentiated_weights[self.query_predicate], n, polynomial)
        #list1, sum_1 = assign_weight(self.query_predicate, self.exponentiated_weights[self.query_predicate], n, polynomial_axiom)
        list0 = [float(sympy.Rational(value/sum_0)) for value in list0]
        return list0
        #list1 = [float(sympy.Rational(value/sum_1)) for value in list1]
        #values_list = list(zip(list0, list1))
        #values_array = np.array(values_list)
        #np.save(f'results/{self.experiment_name}_{n}_{self.weights}_{self.predicates_to_rescale}_{self.cardinality_constraints_binary}.npy', values_array)
        #self.bar_plot(values_array)
        #return values_array
    
    def runtime(self, n):
        # runtime for computing the partition function (not a query)
        start_time = timeit.default_timer()
        self.compute_weights(n)
        language = Language(self.unary_predicates, self.binary_predicates, weights = self.exponentiated_weights)
        formula_axiom = Formula(self.expression, self.axiom, self.axiom_predicate, cardinality_constraints_binary=self.cardinality_constraints_binary)
        _ = WFOMC(language, formula_axiom, n)
        return timeit.default_timer()-start_time



def expected_value(probabilities):
    values = np.arange(0, probabilities.shape[1])
    return np.dot(probabilities, values)


def runtime_experiment(MLNs, n):
    times_list = []
    for i in range(n+1):
        times = [mln.runtime(i) for mln in MLNs]
        print(i, times)
        times_list.append(times)
    values_array = np.array(times_list[1:])
    return values_array

def query_experiment(MLNs, n, unary=True, max_cardinality = None, only_even = False):
    values_list = []
    for mln in MLNs:
        if unary:
            values = mln.unary_query_experiment(n)
        else:
            values = mln.binary_query_experiment(n, max_cardinality = max_cardinality, only_even = only_even)
        print(values)
        values_list.append(values)
    values_array = np.array(values_list).T
    return values_array

def plot_runtimes(values_array, experiment_name, legend, title=False):
    fontsize = 'x-large'
    fontsize_legend = 'large'
    plt.close('all')
    n = len(values_array)
    x = np.arange(len(values_array)) + 1
    #plt.plot(x, values_array[:, 0], label=f'With hard connected axiom', color='C0')
    #plt.plot(x, values_array[:, 1], label=f'With soft connected axiom', color='C1')
    for i,item_legend in enumerate(legend):
        plt.plot(x, values_array[:, i], label=item_legend)
    plt.xlabel('domain size', fontsize=fontsize)
    plt.ylabel('time (s)', fontsize=fontsize)
    #plt.xticks(x, x)
    plt.legend(fontsize=fontsize_legend)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)  
    plt.yscale("log")
    if title:
        plt.title('Runtimes', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'plots/runtimes_{experiment_name}_{n}.png')


def produce_plot_title(weights, n, predicates_to_rescale=[]):
    #plt.rcParams['text.usetex'] = True
    str_list=[]
    for k, v in weights.items():
        if k in predicates_to_rescale:
            str_list.append(f'$w_{k}$ = {v[0]:.1f}\n')
        else:
            str_list.append(f'$w_{k}$ = {v[0]:.1f}')
    str_weights = ',   '.join(str_list)
    if predicates_to_rescale:
        return f'DA-MLN; n={n}; \n' + str_weights
    else:
        #return f'n={n};   ' + str_weights
        return str_weights

def bar_plot(values_array, experiment_name, legend, weights, query_predicate, title=True, predicates_to_rescale=[], y_lim = None):
    fontsize = 'x-large'
    figure = plt.figure(figsize=(8,6))
    n = len(values_array) - 1
    width = 1/(len(legend))
    separation = 0.2
    x = np.arange(len(values_array))
    x_new = x * (1 + separation)
    for i,item_legend in enumerate(legend):
        plt.bar(x_new+i*width, values_array[:, i], width, label=item_legend)
    if title:
        plt.title(produce_plot_title(weights, n, predicates_to_rescale), fontsize='x'+fontsize)
    if y_lim is not None:
        plt.ylim(0,y_lim)
    plt.xlabel('m', fontsize=fontsize)
    plt.ylabel(f'P(|{query_predicate}|=m)', fontsize=fontsize)
    plt.xticks(x_new + (len(legend)-1)*width/2, x, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize)  
    plt.tight_layout()
    figure.savefig(f'plots/{experiment_name}_{n}_{weights}.png')
    plt.close()

def alternative_plot(values_array, n, experiment_name, legend, weights, query_predicate, title=True, predicates_to_rescale=[], below=False, max_x = 1000000, only_even = False):
    fontsize = 'x-large'
    fontsize_legend = 'large'
    plt.close('all')
    x = np.arange(min(len(values_array), max_x))
    for i,item_legend in enumerate(legend):
        plt.plot(x, values_array[:max_x, i], label=item_legend)
        if below:
            plt.fill_between(x, values_array[:max_x, i], alpha=0.2)
    if title:
        plt.title(produce_plot_title(weights, n, predicates_to_rescale), fontsize=fontsize)
    #plt.plot(x, values_array[:, 0], label=f'Without {self.axiom} axiom', color='C0')
    #plt.plot(x, values_array[:, 1], label=f'With {self.axiom} axiom', color='C1')
    if only_even:
        scaling = '/2'
    else:
        scaling = ''
    plt.xlabel('m', fontsize=fontsize)
    plt.ylabel(f'P(|{query_predicate}|{scaling}=m)', fontsize=fontsize)
    #plt.xticks(x, x)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)  
    plt.tight_layout()
    plt.savefig(f'plots/{experiment_name}_{n}_{weights}.png')


def plot_expected_values(values_array, n, starting_n, experiment_name, legend, weights, query_predicate, title=False, only_even=False):
    fontsize = 'x-large'
    fontsize_legend = 'x-large'
    plt.close('all')
    x = np.arange(starting_n, n+1)
    for i,item_legend in enumerate(legend):
        plt.plot(x, values_array[:, i], label=item_legend)
    if title:
        plt.title(produce_plot_title(weights, n), fontsize=fontsize)
    #plt.plot(x, values_array[:, 0], label=f'Without {self.axiom} axiom', color='C0')
    #plt.plot(x, values_array[:, 1], label=f'With {self.axiom} axiom', color='C1')
    if only_even:
        scaling = '/2'
    else:
        scaling = ''
    plt.xlabel('domain size', fontsize=fontsize)
    plt.ylabel(f'E[|{query_predicate}|{scaling}]', fontsize=fontsize)
    #plt.xticks(x, x)
    plt.xticks(fontsize=fontsize)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize=fontsize_legend)  
    plt.tight_layout()
    plt.savefig(f'plots/{experiment_name}_{n}_{weights}.png')

def runtime_connected_hard(n):
    experiment_name = 'smokers-hard'
    MLN_hard = MLN_experiment('hard-connected', {'S': (0, 0), 'F': (-log(8), 0), 'P': (log(20), 0)})
    #MLN_soft = MLN_experiment('soft-connected', {'S': (0, 0), 'F': (-log(3), 0), 'R': (log(3), 0), 'Z': (log(3), 0)})
    MLN_baseline1 = MLN_hard.without_axiom({})
    MLN_baseline2 = MLN_hard.without_axiom({'F': ('>', 2*n-3)})
        #legend = ['hard connected', 'soft connected', 'cardinality constraint |F|>2n-3', 'nothing']
    legend = ['Connected(F)', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2>n-2']
    values_array = runtime_experiment([MLN_hard, MLN_baseline1, MLN_baseline2], n)
    np.save(f'results/{experiment_name}.npy', values_array)
    plot_runtimes(values_array, experiment_name, legend)

def runtime_connected(n):
    experiment_name = 'connected_all'
    MLN_hard = MLN_experiment('hard-connected', {'S': (0, 0), 'F': (-log(8), 0), 'P': (log(20), 0)})
    MLN_soft = MLN_experiment('soft-connected', {'S': (0, 0), 'F': (-log(8), 0), 'P': (log(20), 0), 'Z': (log(8), 0)})
    MLN_baseline1 = MLN_hard.without_axiom({})
    MLN_baseline2 = MLN_hard.without_axiom({'F': ('>', 2*n-3)})
    legend = ['hard-connectivity', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2>n-2', 'soft-connectivity']
    values_array = runtime_experiment([MLN_hard, MLN_baseline1, MLN_baseline2, MLN_soft], n)
    np.save(f'results/{experiment_name}.npy', values_array)
    plot_runtimes(values_array, experiment_name, legend)

def hard_connected(n):
    experiment_name = 'hard-connected'
    MLN_hard = MLN_experiment('hard-connected', {'S': (0, 0), 'F': (-1, 0), 'P': (2, 0)})
    MLN_baseline2 = MLN_hard.without_axiom({'F': ('>', 2*n-3)})
    MLN_baseline1 = MLN_hard.without_axiom()
    legend = ['Connected(F)', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2>n-2']
    values_array = query_experiment([MLN_hard, MLN_baseline1, MLN_baseline2], n)
    bar_plot(values_array, experiment_name, legend, MLN_hard.weights, 'S', title=True, y_lim=0.4)


def DAG_edges(n):
    experiment_name = 'DAG-edges'
    MLN_hard = MLN_experiment('DAG-edges', {'R': (-1, 0)})
    #MLN_baseline1 = MLN_hard.without_axiom({'R': ('', n-2)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['DAG(R)', '¬R(x,x) ∧ (R(x,y) → ¬R(y,x))', 'soft-DAG']
    values_array = query_experiment([MLN_hard, MLN_baseline], n, False, n*(n-1)//2)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'R', title=False)

def DAG_edges_expectations(n, starting_n = 5):
    experiment_name = 'DAG-edges-expectations'
    MLN_hard = MLN_experiment('DAG-edges', {'R': (-1, 0)})
    #MLN_baseline1 = MLN_hard.without_axiom({'R': ('', n-2)})
    MLN_soft = MLN_hard.with_soft_constraint()
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['DAG(R)', '¬R(x,x) ∧ (R(x,y) → ¬R(y,x))']
    arrays = []
    for i in range(starting_n, n+1):
        exp_values = expected_value(query_experiment([MLN_hard, MLN_baseline, MLN_soft], i, False, i*(i-1)//2).T)
        print(i, exp_values)
        arrays.append(exp_values)
    expectations_array = np.array(arrays)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', expectations_array)
    plot_expected_values(expectations_array, n, starting_n, experiment_name, legend, MLN_hard.weights, 'R', title=False, only_even=False)

#arr =np.load("results/DAG-edges-expectations_25_{'R': (-1, 0)}.npy")
#plot_expected_values(arr, 25, 5, 'DAG-edges-expectations', ['DAG(R)', '¬R(x,x) ∧ (R(x,y) → ¬R(y,x))'], {'R': (-1, 0)}, 'R', title=False, only_even=False)



def DAG_edges_smokers(n):
    experiment_name = 'DAG-edges-smokers'
    MLN_hard = MLN_experiment('DAG-edges-smokers',  {'S': (0, 0), 'F': (-1, 0), 'P': (3, 0)})
    #MLN_baseline1 = MLN_hard.without_axiom({'R': ('', n-2)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['DAG(R)', '¬F(x,x) ∧ (F(x,y) → ¬F(y,x))']
    values_array = query_experiment([MLN_hard, MLN_baseline], n, False, n*(n-1)//2)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', values_array)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'P', title=False)

def hard_DAG(n):
    experiment_name = 'hard-DAG'
    MLN_hard = MLN_experiment('hard-DAG', {'S': (0, 0), 'F': (-2, 0), 'P': (3, 0)})
    #MLN_baseline2 = MLN_hard.without_axiom({'F': ('>', 2*n-3)})
    MLN_baseline1 = MLN_hard.without_axiom()
    legend = ['DAG(F)', '¬F(x,x) ∧ (F(x,y) → ¬F(y,x))']
    values_array = query_experiment([MLN_hard, MLN_baseline1], n)
    bar_plot(values_array, experiment_name, legend, MLN_hard.weights, 'S', title=True, y_lim=0.4)

def connected_edges(n):
    experiment_name = 'connected-edges'
    MLN_hard = MLN_experiment('connected-edges', {'R': (-1, 0)})
    MLN_baseline1 = MLN_hard.without_axiom({'R': ('>', 2*n-3)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['Connected(R)', '¬R(x,x) ∧ (R(x,y) → R(y,x))', '¬R(x,x) ∧ (R(x,y) → R(y,x)) ∧ |R|/2>n-2']
    values_array = query_experiment([MLN_hard, MLN_baseline, MLN_baseline1], n, False, n*(n-1), True)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'R', title=False, max_x = int(4.5*n), only_even=True)

def connected_edges_expectations(n, starting_n = 5):
    experiment_name = 'connected-edges-expectations'
    MLN_hard = MLN_experiment('connected-edges', {'R': (-1, 0)})
    MLN_baseline = MLN_hard.without_axiom()
    MLN_soft = MLN_hard.with_soft_constraint()
    legend = ['Connected(R)', '¬R(x,x) ∧ (R(x,y) → R(y,x))', '¬R(x,x) ∧ (R(x,y) → R(y,x)) ∧ |R|/2>n-2', 'soft-connectivity']
    arrays = []
    for i in range(starting_n, n+1):
        MLN_baseline1 = MLN_hard.without_axiom({'R': ('>', 2*i-3)})
        exp_values = expected_value(query_experiment([MLN_hard, MLN_baseline, MLN_baseline1, MLN_soft], i, False, i*(i-1), True).T)
        print(i, exp_values)
        arrays.append(exp_values)
    expectations_array = np.array(arrays)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', expectations_array)
    plot_expected_values(expectations_array, n, starting_n, experiment_name, legend, MLN_hard.weights, 'R', title=False, only_even=True)

def connected_edges_smokers(n):
    experiment_name = 'connected-edges-smokers'
    MLN_hard = MLN_experiment('connected-edges-smokers',  {'S': (0, 0), 'F': (-1, 0), 'P': (3, 0)})
    MLN_baseline1 = MLN_hard.without_axiom({'F': ('>', 2*n-3)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['Connected(F)', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2>n-2']
    values_array = query_experiment([MLN_hard, MLN_baseline, MLN_baseline1], n, False, n*(n-1), True)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', values_array)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'F', title=False, max_x = int(4.5*n), only_even=True)

def forest_edges(n):
    experiment_name = 'forest-edges'
    MLN_hard = MLN_experiment('forest-edges', {'R': (-1, 0)})
    MLN_baseline1 = MLN_hard.without_axiom({'R': ('<', 2*n-1)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['Forest(R)', '¬R(x,x) ∧ (R(x,y) → R(y,x))', '¬R(x,x) ∧ (R(x,y) → R(y,x)) ∧ |R|/2<n']
    values_array = query_experiment([MLN_hard, MLN_baseline, MLN_baseline1], n, False, n*(n-1), only_even=True)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'R', title=False, max_x = int(4.5*n), only_even = True)

def forest_edges_expectations(n, starting_n = 5):
    experiment_name = 'forest-edges-expectations'
    MLN_hard = MLN_experiment('forest-edges', {'R': (-1, 0)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['forest(R)', '¬R(x,x) ∧ (R(x,y) → R(y,x))', '¬R(x,x) ∧ (R(x,y) → R(y,x)) ∧ |R|/2<n']
    arrays = []
    for i in range(starting_n, n+1):
        MLN_baseline1 = MLN_hard.without_axiom({'R': ('<', 2*i-1)})
        exp_values = expected_value(query_experiment([MLN_hard, MLN_baseline, MLN_baseline1], i, False, i*(i-1), True).T)
        print(i, exp_values)
        arrays.append(exp_values)
    expectations_array = np.array(arrays)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', expectations_array)
    plot_expected_values(expectations_array, n, starting_n, experiment_name, legend, MLN_hard.weights, 'R', title=False, only_even=True)

def forest_edges_smokers(n):
    experiment_name = 'forest-edges-smokers'
    MLN_hard = MLN_experiment('forest-edges-smokers',  {'S': (0, 0), 'F': (-1, 0), 'P': (3, 0)})
    MLN_baseline1 = MLN_hard.without_axiom({'F': ('<', 2*n-1)})
    MLN_baseline = MLN_hard.without_axiom()
    legend = ['Forest(F)', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2<n']
    values_array = query_experiment([MLN_hard, MLN_baseline, MLN_baseline1], n, False, n*(n-1), True)
    alternative_plot(values_array, n, experiment_name, legend, MLN_hard.weights, 'F', title=False, max_x = int(4.5*n), only_even=True)

def hard_forest(n):
    experiment_name = 'hard-forest'
    MLN_hard = MLN_experiment('hard-forest', {'S': (0, 0), 'F': (-2, 0), 'P': (3, 0)})
    MLN_baseline2 = MLN_hard.without_axiom({'F': ('<', 2*n-1)})
    MLN_baseline1 = MLN_hard.without_axiom()
    legend = ['Forest(F)', '¬F(x,x) ∧ (F(x,y) → F(y,x))', '¬F(x,x) ∧ (F(x,y) → F(y,x)) ∧ |F|/2<n']
    values_array = query_experiment([MLN_hard, MLN_baseline1, MLN_baseline2], n)
    np.save(f'results/{experiment_name}_{n}_{MLN_hard.weights}.npy', values_array)
    bar_plot(values_array, experiment_name, legend, MLN_hard.weights, 'S', title=True, y_lim=0.4)

def plot_bar(experiment_name, legend, weights, query, n, title=True, y_lim=0.4):
    values_array = np.load(f'results/{experiment_name}_{n}_{weights}.npy')
    bar_plot(values_array, experiment_name, legend, weights, query, title=title, y_lim=y_lim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=20, type=int)
    parser.add_argument('--function', default='hard_connected', type=str, help='function to be tested')
    #parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    
    locals()[args.function](args.n)


    #experiment_name = 'hard-DAG'
    #n = 30
    #weights = {'A': (0, 0), 'R': (5, 0), 'C':(-log(n), 0)}
    #predicates_to_rescale = []
    #experiment = MLN_experiment(experiment_name, weights, predicates_to_rescale) 
    #experiment.scalable_experiment(n) 


    #experiment_name = 'hard-connected'
    #n = args.n
    #weights = {'S': (0, 0), 'F': (-2, 0), 'R': (2, 0)}
    #predicates_to_rescale = []
    #cardinality_constraints_binary = {'F': ('>', n-2)}
    #experiment = MLN_experiment(experiment_name, weights, predicates_to_rescale, cardinality_constraints_binary) 

