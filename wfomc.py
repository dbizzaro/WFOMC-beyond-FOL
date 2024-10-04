# -*- coding: utf-8 -*-

from sympy import *
import copy
import itertools
from scipy.special import comb
import symengine
import math

X, Y = 'xy'
VARIABLES = [X, Y] # remember to not use x and y inside the predicates' names
AND = '&'
OR = '|'
NOT = '~'
IMPLIES = '>>'

AUXILIARY_PREDICATE= 'V'

class Language:
  def __init__(self, unary_predicates, binary_predicates, weights = {}):
    #remember that it is better to use names for predicates which have just one capital letter
    #should not be possible to be confused with other parts of the logical language, e.g. variables

    self.unary_predicates = unary_predicates
    self.binary_predicates = binary_predicates
    self.weights = weights

    #define one_variable_atoms
    self.one_variable_atoms = []
    for variable in VARIABLES:
      types_with_variable = []
      for predicate in self.unary_predicates:
        types_with_variable.append(predicate + variable)
      for predicate in self.binary_predicates:
        types_with_variable.append(predicate + variable +  variable)
      self.one_variable_atoms.append(types_with_variable)

    #define two_variable_atoms
    self.two_variables_atoms = []
    for predicate in binary_predicates:
      self.two_variables_atoms.append(predicate + VARIABLES[0] + VARIABLES[1])
      self.two_variables_atoms.append(predicate + VARIABLES[1] + VARIABLES[0])

    #define all_atoms
    self.all_atoms = self.one_variable_atoms[0] + self.one_variable_atoms[1] + self.two_variables_atoms

    #define u and b
    self.u = len(self.one_variable_atoms[0])
    self.b = len(self.two_variables_atoms)

    #set weights unary predicates
    self.set_weights(weights)

  def set_weights(self, weights): #the argument passe should be a dictionary {predicate:(weight_affermative, weight_negation)}
    self.unary_weights = []
    self.binary_weights = []
    for atom in self.unary_predicates:
      self.unary_weights.append(weights.get(atom, (1,1)))
    for atom in self.binary_predicates:
      self.binary_weights.append(weights.get(atom, (1,1)))

  def get_weight_binary_atom(self, atom):
    predicate = atom[0]
    index_predicate = self.binary_predicates.index(predicate)
    return self.binary_weights[index_predicate]

  def set_symbolic_weights(self, formula):
    new_weights = copy.copy(self.weights)
    for predicate in formula.cardinality_constraints_binary.keys():
      predicate_symbol = symengine.Symbol(predicate)
      new_weights[predicate] = (predicate_symbol, 1)
    return Language(self.unary_predicates, self.binary_predicates, new_weights)

  def set_symbolic_weight(self, predicate):
    new_weights = copy.copy(self.weights)
    predicate_symbol = symengine.Symbol(predicate)
    new_weights[predicate] = (predicate_symbol, 1)
    return Language(self.unary_predicates, self.binary_predicates, new_weights)


class Formula:
  def __init__(self, formula_string, axiom = None, axiom_predicate = None, cardinality_constraints_unary = {}, cardinality_constraints_binary = {}):
    self.formula_string = formula_string
    self.parsed_formula = parse_expr(formula_string)
    self.used_symbols = {str(symbol) for symbol in self.parsed_formula.free_symbols}
    self.axiom = axiom
    self.axiom_predicate = axiom_predicate
    self.cardinality_constraints_unary = cardinality_constraints_unary
    self.cardinality_constraints_binary = cardinality_constraints_binary

  def set_constraint_tree(self, n, keep_other_constraints = True):
    if keep_other_constraints:
      new_dict_constraints_binary = copy.copy(self.cardinality_constraints_binary)
    else:
      new_dict_constraints_binary = {}
    new_dict_constraints_binary[self.axiom_predicate] = ('=', 2 * n - 2)
    return Formula(self.formula_string, 'connected', self.axiom_predicate, self.cardinality_constraints_unary, new_dict_constraints_binary)

  def extended_string(self):
    # the variables should not be part of any predicate name
    formula1 = self.formula_string.replace(X, Y) #substitute x with y
    formula2 = self.formula_string.replace(Y, X) #substitute y with x
    formula3 = self.formula_string.translate(str.maketrans(X+Y, Y+X)) #swap x and y
    return f'({self.formula_string}) {AND} ({formula1}) {AND} ({formula2}) {AND} ({formula3})'

  def extended_formula(self):
    return Formula(self.extended_string(), self.axiom, self.axiom_predicate, self.cardinality_constraints_unary, self.cardinality_constraints_binary)

  def add(self, additional_string):
    new_formula_string = f'({additional_string}) {AND} ({self.formula_string})'
    return Formula(new_formula_string, self.axiom, self.axiom_predicate, self.cardinality_constraints_unary, self.cardinality_constraints_binary)

  def set_language_dependent_cardinality_constraints(self, language):
    self.constraints_dict = {language.one_variable_atoms[0].index(predicate): n_constraint for predicate, n_constraint in self.cardinality_constraints_unary.items()}

  def equivalent_formula_with_all_useful_predicates(self, language):
    '''add terms "a or not a" for all a unary not in formula'''
    new_formula_string = self.formula_string
    for atom in language.one_variable_atoms[0] + language.one_variable_atoms[1]: #+ self.binary_atoms_for_cardinality:
      if atom not in self.used_symbols:
        new_formula_string = f"({new_formula_string}) {AND} ({atom} {OR} {NOT} {atom})"
    return Formula(new_formula_string, self.axiom, self.axiom_predicate, self.cardinality_constraints_unary, self.cardinality_constraints_binary)
  

  def add_auxiliary_predicate(self, language):
    new_formula = self.add(f"{AUXILIARY_PREDICATE}{X}{Y} {IMPLIES} {self.axiom_predicate}{X}{Y}").add(f"{self.axiom_predicate}{X}{Y} {IMPLIES} {AUXILIARY_PREDICATE}{X}{Y}")
    new_language = Language(language.unary_predicates, language.binary_predicates + [AUXILIARY_PREDICATE], {**language.weights, AUXILIARY_PREDICATE: (symengine.Symbol(AUXILIARY_PREDICATE), 1)})
    return new_formula, new_language

  def add_cardinality_constraint_unary(self, predicate, n_constraint):
    return Formula(self.formula_string, self.axiom, self.axiom_predicate, {**self.cardinality_constraints_unary, predicate: n_constraint}, self.cardinality_constraints_binary)

###### Combinatorial functions/generators

#https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
def multinomial_coefficient(lst):
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

#https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python
#https://stackoverflow.com/questions/62344469/find-all-combinations-of-n-positive-numbers-adding-up-to-k-in-python?noredirect=1&lq=1
def vectors_k(total_sum, length):
  if length == 1:
    yield (total_sum,)
  else:
    for value in range(total_sum + 1):
      for permutation in vectors_k(total_sum - value, length - 1):
        yield (value,) + permutation


def smaller_tuples(n, k, length, idx=0):
  if length == 1:
    if n <= k[idx]:
      yield (n,)
  else:
    max_value = min(n, k[idx])
    remaining_max_sum = sum(k[idx + 1:idx + length])
    for value in range(max_value + 1):
      remaining_sum = n - value
      if remaining_sum <= remaining_max_sum:
        for permutation in smaller_tuples(remaining_sum, k, length - 1, idx + 1):
          yield (value,) + permutation



###### Weight functions

def compute_weight_one_type(language, one_type):
  prod_ = 1
  n_unary_predicates = len(language.unary_predicates)
  for predicate_position in range(language.u):
    int_value = 0 if one_type[predicate_position] else 1
    if predicate_position < n_unary_predicates:
      prod_ *= language.unary_weights[predicate_position][int_value]
    else:
      prod_ *= language.binary_weights[predicate_position - n_unary_predicates][int_value]
  return prod_ 

def compute_weight_two_table(language, model, used_symbols):
  prod_ = 1
  for atom in language.two_variables_atoms:
    weights_predicate = language.get_weight_binary_atom(atom)
    if atom in used_symbols:
      int_value = 0 if model[parse_expr(atom)] else 1
      prod_ *= weights_predicate[int_value]
    else:
      prod_ *= weights_predicate[0] + weights_predicate[1]
  return prod_

def weight_k(weights, k):
  prod_ = 1
  for i, weight in enumerate(weights):
    prod_ *= weight ** k[i]
  return prod_



###### n_ij computation and representation

def compute_all_n_ij(language, formula):
  equivalent_formula = formula.equivalent_formula_with_all_useful_predicates(language)
  solutions_dict = {}
  all_models = satisfiable(equivalent_formula.parsed_formula, all_models=True)
  for model in all_models:
    if model:
      one_type_i = tuple([True if model[Symbol(atom)] else False for atom in language.one_variable_atoms[0]])
      one_type_j = tuple([True if model[Symbol(atom)] else False for atom in language.one_variable_atoms[1]])
      #two_table_for_cardinality = tuple([True if model[Symbol(atom)] else False for atom in formula.binary_atoms_for_cardinality])
      #solutions_dict[(one_type_i, one_type_j, two_table_for_cardinality)] = compute_weight_two_table(language, model, equivalent_formula.used_symbols_extended) + solutions_dict.get((one_type_i, one_type_j, two_table_for_cardinality), 0)
      solutions_dict[(one_type_i, one_type_j)] = compute_weight_two_table(language, model, formula.used_symbols) + solutions_dict.get((one_type_i, one_type_j), 0)
  return solutions_dict


def order_one_types(language, n_ij_dict):
  actual_one_types = tuple({elem[0] for elem in n_ij_dict.keys()} | {elem[1] for elem in n_ij_dict.keys()})
  actual_u = len(actual_one_types)
  weights = []
  for one_type in actual_one_types:
    weights.append(compute_weight_one_type(language, one_type))
  return actual_one_types, actual_u, weights

def produce_n_ij_matrix(n_ij_dict, actual_one_types, actual_u):
  n_ij_matrix = tuple([tuple([n_ij_dict[(actual_one_types[i], actual_one_types[j])] for j in range(actual_u)]) for i in range(actual_u)])
  #print(n_ij_matrix)
  return n_ij_matrix



###### Cardinality constraints

def k_satisfies_cardinality_constraints(formula, k, actual_one_types):
  for predicate_index, constraint_number in formula.constraints_dict.items():
    count_ = 0
    for one_type_index, one_type in enumerate(actual_one_types):
      if one_type[predicate_index]:
        count_ += k[one_type_index]
    if count_ != constraint_number:
      return False
  return True

def impose_binary_cardinality_constraints(language, formula, polynomial, n):
  if not formula.cardinality_constraints_binary: #if no cardinality constraints on binary predicates
    return polynomial
  polynomial = symengine.expand(polynomial)
  monomials = [1]
  weights = [1]
  for predicate, constraint in formula.cardinality_constraints_binary.items():
    if constraint[0] == '=':
      monomials = [monomial * symengine.Symbol(predicate)**constraint[1] for monomial in monomials]
      weights = [weight * language.get_weight_binary_atom(predicate)[0]**constraint[1] for weight in weights]
    elif constraint[0] == '<':
      monomials = [monomial * symengine.Symbol(predicate)**exponent for monomial in monomials for exponent in range(constraint[1])]
      weights = [weight * language.get_weight_binary_atom(predicate)[0]**exponent for weight in weights for exponent in range(constraint[1])]
    elif constraint[0] == '>':
      max_degree = n**2
      monomials = [monomial * symengine.Symbol(predicate)**exponent for monomial in monomials for exponent in range(constraint[1], max_degree+1)]
      weights = [weight * language.get_weight_binary_atom(predicate)[0]**exponent for weight in weights for exponent in range(constraint[1], max_degree + 1)]
  polynomial_dict = polynomial.as_coefficients_dict()
  sum_ = 0
  for monomial, weight in zip(monomials, weights):
    sum_ += polynomial_dict.get(monomial, 0) * weight
  return sum_


def impose_single_constraint(predicate, predicate_weight, cardinality, polynomial):
  polynomial = symengine.expand(polynomial)
  return polynomial.coeff(symengine.Symbol(predicate), cardinality) * predicate_weight**cardinality
  

def assign_weight(predicate, predicate_weight, n, polynomial, maximum_cardinality = None):
  #used for computing the WFOMCs for each cardinality of a unary predicate (from 0 to n) all at once
  #useful for querying MLNs
  #not working when there are cardinality constraints on binary predicates
  polynomial = symengine.expand(polynomial)
  values_list = []
  if maximum_cardinality is None:
    maximum_cardinality = n
  for i in range(maximum_cardinality+1):
    monomial = symengine.Symbol(predicate)
    values_list.append(polynomial.coeff(monomial, i) * (predicate_weight[0] ** i) * (predicate_weight[1] ** (n - i)))
  return values_list, sum(values_list)



###### WFOMC

def WFOMC_given_k(n_ij_matrix, k, actual_u):
  prod_ = multinomial_coefficient(k)
  for i,j in itertools.combinations_with_replacement(range(actual_u), 2):
    if i != j:
      k_ij = k[i] * k[j]
    else:
      k_ij = k[i] * (k[i] - 1) // 2
    prod_ *= (n_ij_matrix[i][j] ** k_ij)
  return prod_

def WFOMC_no_axiom(language, formula, n):
  formula = formula.extended_formula()
  formula.set_language_dependent_cardinality_constraints(language)
  language_with_symbolic_weights = language.set_symbolic_weights(formula)
  n_ij_dict = compute_all_n_ij(language_with_symbolic_weights, formula)
  actual_one_types, actual_u, weights = order_one_types(language_with_symbolic_weights, n_ij_dict)
  n_ij_matrix = produce_n_ij_matrix(n_ij_dict, actual_one_types, actual_u)
  if actual_u == 0:
    return 0
  sum_ = 0
  for k in vectors_k(n, actual_u):
    if k_satisfies_cardinality_constraints(formula, k, actual_one_types):
      sum_ += WFOMC_given_k(n_ij_matrix, k, actual_u) * weight_k(weights, k)
  return impose_binary_cardinality_constraints(language, formula, sum_, n)


def number_of_extensions(n_ij_matrix_for_extensions, k_prime, k_second, actual_u, C):
  prod_ = 1
  if k_prime in C:
    for j, k_j in enumerate(k_second):
      prod_ *= C[k_prime][j] ** k_j
  else:
    C[k_prime] = [math.prod([n_ij_matrix_for_extensions[i][j]**k_i for i, k_i in enumerate(k_prime)]) for j in range(actual_u)]
    return number_of_extensions(n_ij_matrix_for_extensions, k_prime, k_second, actual_u, C)
  return prod_


###### DAGs

def WFOMC_DAG_Psi_m(n_ij_matrix_for_extensions, n_ij_matrix_for_sources, A, m, s, actual_u, B, C):
  sum_ = 0
  for s_prime in smaller_tuples(m, s, actual_u):
    s_second = tuple([s[i]-s_prime[i] for i in range(len(s))])
    if s_prime in B:
      WFOMC_on_sources = B[s_prime]
    else:
      WFOMC_on_sources = WFOMC_given_k(n_ij_matrix_for_sources, s_prime, actual_u)
      B[s_prime] = WFOMC_on_sources
    sum_ += number_of_extensions(n_ij_matrix_for_extensions, s_prime, s_second, actual_u, C) * A[s_second] * WFOMC_on_sources
  return sum_

def WFOMC_DAG_given_k(n_ij_dict_for_extensions, n_ij_dict_for_sources, k, actual_u, A, B, C):
  for p in itertools.product(*[range(n+1) for n in k]):
    if p not in A:
      abs_p = sum(p)
      sum_ = 0
      for l in range(abs_p):
        wfomc_psi_m = WFOMC_DAG_Psi_m(n_ij_dict_for_extensions, n_ij_dict_for_sources, A, abs_p - l, p, actual_u, B, C)
        sum_ += (-1) ** (abs_p - l + 1) * comb(abs_p, l, exact=True) * wfomc_psi_m
      if isinstance(sum_, symengine.Basic):
        sum_ = symengine.expand(sum_)
      A[p] = sum_
  return A[k]

def WFOMC_DAG(language, formula, n):
  assert formula.axiom_predicate in language.binary_predicates #axiom_relation should be one of the binary predicates
  formula = formula.extended_formula().add(f'{NOT} {formula.axiom_predicate}{X}{X} {AND} {NOT} {formula.axiom_predicate}{Y}{Y}')
  formula.set_language_dependent_cardinality_constraints(language)
  language_with_symbolic_weights = language.set_symbolic_weights(formula)
  n_ij_dict = compute_all_n_ij(language_with_symbolic_weights, formula)
  actual_one_types, actual_u, weights = order_one_types(language_with_symbolic_weights, n_ij_dict)
  formula_for_extensions = formula.add(f'{NOT} {formula.axiom_predicate}{Y}{X}')
  n_ij_dict_for_extensions = compute_all_n_ij(language_with_symbolic_weights, formula_for_extensions)
  n_ij_matrix_for_extensions = produce_n_ij_matrix(n_ij_dict_for_extensions, actual_one_types, actual_u)
  formula_for_sources = formula_for_extensions.add(f'{NOT} {formula.axiom_predicate}{X}{Y}')
  n_ij_dict_for_sources = compute_all_n_ij(language_with_symbolic_weights, formula_for_sources)
  n_ij_matrix_for_sources = produce_n_ij_matrix(n_ij_dict_for_sources, actual_one_types, actual_u)
  sum_ = 0
  A = {tuple([0 for _ in range(actual_u)]): 1}
  B = {}
  C = {}
  for k in vectors_k(n, actual_u):
    if k_satisfies_cardinality_constraints(formula, k, actual_one_types):
      wmc_k = WFOMC_DAG_given_k(n_ij_matrix_for_extensions, n_ij_matrix_for_sources, k, actual_u, A, B, C)
      sum_ += wmc_k * weight_k(weights, k)
  return impose_binary_cardinality_constraints(language, formula, sum_, n)


###### Connected

def WFOMC_connected_Psi_m(n_ij_matrix_for_extensions, n_ij_matrix, A, m, s, actual_u, B, C):
  sum_ = 0
  for s_prime in smaller_tuples(m, s, actual_u):
    s_second = tuple([s[i]-s_prime[i] for i in range(len(s))])
    if s_second in B:
      WFOMC_on_connected = B[s_second]
    else:
      WFOMC_on_connected = WFOMC_given_k(n_ij_matrix, s_second, actual_u)
      B[s_second] = WFOMC_on_connected
    n_extensions = number_of_extensions(n_ij_matrix_for_extensions, s_prime, s_second, actual_u, C)
    sum_ += n_extensions * A[s_prime] * WFOMC_on_connected
  return sum_

def WFOMC_connected_given_k(n_ij_matrix_for_extensions, n_ij_matrix, k, actual_u, A, B, C):
  for p in itertools.product(*[range(n+1) for n in k]):
    if p not in A:
      abs_p = sum(p)
      sum_ = 0
      for m in range(1, abs_p):
        wfomc_psi_m = WFOMC_connected_Psi_m(n_ij_matrix_for_extensions, n_ij_matrix, A, m, p, actual_u, B, C)
        sum_ += comb(abs_p, m, exact=True) * m * wfomc_psi_m
      if isinstance(sum_, int): #to distiguish when we have symbolic weights
        divided_sum = sum_ // abs_p
      else:
        divided_sum = sum_ / abs_p
      result_ = WFOMC_given_k(n_ij_matrix, p, actual_u) - divided_sum
      if isinstance(result_, symengine.Basic):
        result_ = symengine.expand(result_)
      A[p] = result_
  return A[k]

def WFOMC_connected(language, formula, n):
  assert formula.axiom_predicate in language.binary_predicates #axiom_predicate should be one of the binary predicates
  formula = formula.add(f'{NOT} {formula.axiom_predicate}{X}{X} {AND} ({formula.axiom_predicate}{X}{Y} {IMPLIES} {formula.axiom_predicate}{Y}{X})').extended_formula()
  formula.set_language_dependent_cardinality_constraints(language)
  language_with_symbolic_weights = language.set_symbolic_weights(formula)
  n_ij_dict = compute_all_n_ij(language_with_symbolic_weights, formula)
  actual_one_types, actual_u, weights = order_one_types(language_with_symbolic_weights, n_ij_dict)
  n_ij_matrix = produce_n_ij_matrix(n_ij_dict, actual_one_types, actual_u)
  formula_for_extensions = formula.add(f'{NOT} {formula.axiom_predicate}{X}{Y}')
  n_ij_dict_for_extensions = compute_all_n_ij(language_with_symbolic_weights, formula_for_extensions)
  n_ij_matrix_for_extensions = produce_n_ij_matrix(n_ij_dict_for_extensions, actual_one_types, actual_u)
  sum_ = 0
  A = {tuple([0 for _ in range(actual_u)]): 0}
  B = {}
  C = {}
  for k in vectors_k(n, actual_u):
    if k_satisfies_cardinality_constraints(formula, k, actual_one_types):
      wmc_k = WFOMC_connected_given_k(n_ij_matrix_for_extensions, n_ij_matrix, k, actual_u, A, B, C)
      sum_ += wmc_k * weight_k(weights, k)
  return impose_binary_cardinality_constraints(language, formula, sum_, n)


###### Forest

def WFOMC_forest_Psi_m(n_ij_matrix_for_extensions, n_ij_matrix, A, m, s, actual_u, axiom_predicate, B, C):
  sum_ = 0
  for s_prime in smaller_tuples(m, s, actual_u):
    s_second = tuple([s[i]-s_prime[i] for i in range(len(s))])
    if s_prime in B:
      WFOMC_on_tree = B[s_prime]
    else:
      A_trees = {tuple([0 for _ in range(actual_u)]): 0}
      polynomial_on_tree = WFOMC_connected_given_k(n_ij_matrix_for_extensions, n_ij_matrix, s_prime, actual_u, A_trees, {}, {})
      WFOMC_on_tree = impose_single_constraint(axiom_predicate, 1, 2 * m - 2, polynomial_on_tree)
      B[s_prime] = WFOMC_on_tree
    sum_ += number_of_extensions(n_ij_matrix_for_extensions, s_prime, s_second, actual_u, C) * A[s_second] * WFOMC_on_tree
  return sum_

def WFOMC_forest_given_k(n_ij_matrix_for_extensions, n_ij_matrix, k, actual_u, axiom_predicate, A, B, C):
  for p in itertools.product(*[range(n+1) for n in k]):
    if p not in A:
      abs_p = sum(p)
      sum_ = 0
      for m in range(1, abs_p + 1):
        wfomc_psi_m = WFOMC_forest_Psi_m(n_ij_matrix_for_extensions, n_ij_matrix, A, m, p, actual_u, axiom_predicate, B, C)
        sum_ += comb(abs_p-1, m-1, exact=True) * wfomc_psi_m
      A[p] = symengine.expand(sum_)
  return A[k]

def WFOMC_forest(language, formula, n):
  assert formula.axiom_predicate in language.binary_predicates #axiom_predicate should be one of the binary predicates
  formula, language = formula.add_auxiliary_predicate(language)
  formula = formula.add(f'{NOT} {formula.axiom_predicate}{X}{X} {AND} ({formula.axiom_predicate}{X}{Y} {IMPLIES} {formula.axiom_predicate}{Y}{X})').extended_formula()
  formula.set_language_dependent_cardinality_constraints(language)
  language_with_symbolic_weights = language.set_symbolic_weights(formula)#.set_symbolic_weight(formula.axiom_predicate)
  n_ij_dict = compute_all_n_ij(language_with_symbolic_weights, formula)
  actual_one_types, actual_u, weights = order_one_types(language_with_symbolic_weights, n_ij_dict)
  n_ij_matrix = produce_n_ij_matrix(n_ij_dict, actual_one_types, actual_u)
  formula_for_extensions = formula.add(f'{NOT} {formula.axiom_predicate}{X}{Y}')
  n_ij_dict_for_extensions = compute_all_n_ij(language_with_symbolic_weights, formula_for_extensions)
  n_ij_matrix_for_extensions = produce_n_ij_matrix(n_ij_dict_for_extensions, actual_one_types, actual_u)
  sum_ = 0
  A = {tuple([0 for _ in range(actual_u)]): 1}
  B = {}
  C = {}
  for k in vectors_k(n, actual_u):
    if k_satisfies_cardinality_constraints(formula, k, actual_one_types):
      wmc_k = WFOMC_forest_given_k(n_ij_matrix_for_extensions, n_ij_matrix, k, actual_u, AUXILIARY_PREDICATE, A, B, C)
      sum_ += wmc_k * weight_k(weights, k)
  return impose_binary_cardinality_constraints(language, formula, sum_, n)



###### Wrapper

def WFOMC(language, formula, n):
  if formula.axiom == 'DAG':
    WFOMC_function = WFOMC_DAG
  elif formula.axiom == 'connected':
    WFOMC_function = WFOMC_connected
  elif formula.axiom == 'tree':
    WFOMC_function = WFOMC_connected
    formula = formula.set_constraint_tree(n)
  elif formula.axiom == 'forest':
    WFOMC_function = WFOMC_forest
  else:
     WFOMC_function = WFOMC_no_axiom
  return WFOMC_function(language, formula, n)