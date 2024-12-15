import warnings
import time
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import sympy as sp
from scipy.integrate import solve_ivp

from evaluators.parameter_estimation import estimate_parameters
from population.initial_generation import beautify_system
from utils.functions import get_functions, funcs_to_str, f
from utils.load_systems import load_systems, create_ode_function
from utils.mapping import get_term_map, get_solved_map, convert_system_to_hash, get_individual_solved, \
    add_individual_solved, reset_solved
from utils.models import SIR, lotka
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.symbolic_utils import o, create_variable

tt = sp.Symbol('t')

class Config_Test:
    def __init__(self):
        self.target = lotka()

        self.G = 20  # Number of generations
        self.N = 100  # Maximum number of population
        self.M = 2  # Maximum number of equations
        self.I = 2  # Maximum number of terms per equation
        self.J = 2  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("4,5")
        self.ivp_method = 'Radau'
        self.minimize_method = 'L-BFGS-B' #'Nelder-Mead' / L-BFGS-B / CG, COBYLA, COBYQA, TNC - fast

        self.elite_rate = 0.1
        self.crossover_rate = 0.2
        self.mutation_rate = 0.5
        self.new_rate = 0.2

        self.system_load_dir = '../data/analysis/computation_equations.txt'  #data/sir_equations.txt'
        self.system_save_dir = '../data/analysis/computation_equations.txt'

        self.DEBUG = False



def generate_term(variables, config, non_empty, funcs):
    random.shuffle(variables)  # O(logN)
    term = None
    var_list = []
    j = 0
    if non_empty or random.randint(0, 1) == 1: # equation to have at least one term
        for var in variables:
            if j == 0 or random.randint(0, 1) == 1:
                func = f(random.choices(funcs, k=1)[0])
                var = func(var)
                j += 1
                if config.allow_composite and j < config.J:  # limit applying composite to only once
                    if random.randint(0, 1) == 1:
                        func = f(random.choices(funcs, k=1)[0])
                        var = func(var)
                        j += 1
                var_list.append(var)
                if j == config.J: break

    if var_list:
        term = var_list[0]
        for var in var_list[1:]:
            operator = o(random.randint(0, 1))
            term = operator(term, var)

    return term


def is_redundant(system, population):
    for p in population:
        check = True
        for (rhs1, rhs2) in zip(system, p):
            if sp.simplify(sum(rhs1[1]) - sum(rhs2[1])) != 0:
                check = False
                break
        if check:
            # print("reduandant: ", system, "<>", p )
            return True
    return False


def generate_systems(N, config, use_hash, funcs):
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)
    systems_hash_list = []
    systems = []
    n = 0
    while n < N:
        system = []
        for m in range(config.M):
            terms = []
            for i in range(config.I):
                term = generate_term(v, config, i == 0, funcs)
                if term is not None and not term in terms:
                    terms.append(term)
            system.append([sp.diff(variables[m], tt), terms])

        if not use_hash :
            if not is_redundant(system, systems):
                systems.append(system)
                n += 1

        else:
            s_hash = convert_system_to_hash(system)
            if not s_hash in systems_hash_list:
                systems.append(system)
                systems_hash_list.append(s_hash)
                n += 1

    return systems


def generate_population(config, use_hash, funcs):
    print("\n#### GENERATE INITIAL POPULATION ####")
    systems = generate_systems(config.N, config, use_hash, funcs)
    for i, system in enumerate(systems):
        if config.DEBUG: print(f"generate_system {i}: {system}")

    # Save the system as NumPy functions
    save_systems_as_numpy_funcs(systems, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return systems


def generate_new_population(history, population, config, funcs, use_hash):
    valid_entries = [(solved[0], ind) for solved, ind in zip(history, population) if not math.isinf(solved[0].fun)]

    new_population = []
    new_population_hash_list = []

    sorted_population = []
    scores = []
    for solved, ind in sorted(valid_entries, key=lambda x: x[0].fun):
        sorted_population.append(ind)
        scores.append(solved.fun)
        if config.DEBUG:
            print(f'score: {solved.fun} | system: {ind}')

    ranked_indices = np.argsort(scores)  # Indices of sorted scores
    ranks = np.argsort(ranked_indices) + 1  # Assign ranks (1 = best)

    gamma = 0.8 # todo - use as hyperparmeter
    weights = [gamma ** (rank - 1) for rank in ranks]
    # weights = [(max(scores) - score) / (max(scores) - min(scores) + 1e-6) for score in scores]
    probabilities = np.array(weights) / sum(weights)

    # elites
    num_parents = max(2, int(len(population) * config.elite_rate))
    for system in sorted_population[:num_parents]:
        new_population.append(system)
        if use_hash:
            s_hash = convert_system_to_hash(system)
            new_population_hash_list.append(s_hash)

    # crossover
    n = 0
    n_crossover = int(len(population) * config.crossover_rate / 2)
    while n < n_crossover:
        parent1, parent2 = random.choices(sorted_population, weights=probabilities, k=2)
        child = crossover(parent1, parent2, config)
        if use_hash:
            s_hash = convert_system_to_hash(child)
            if not s_hash in new_population_hash_list:
                solved = get_individual_solved(s_hash)
                if not solved or not math.isinf(solved.fun):
                    new_population.append(child)
                    new_population_hash_list.append(s_hash)
                    n += 1
        else:
            if not is_redundant(child, new_population):
                new_population.append(child)
                n += 1

    # mutation
    n = 0
    n_mutation = int(len(population) * config.mutation_rate)
    while n < n_mutation:
        parent = random.choices(sorted_population, weights=probabilities, k=1)
        child = mutate(parent[0], config, funcs)
        if use_hash:
            s_hash = convert_system_to_hash(child)
            if not s_hash in new_population_hash_list:
                solved = get_individual_solved(s_hash)
                if not solved or not math.isinf(solved.fun):
                    new_population.append(child)
                    new_population_hash_list.append(s_hash)
                    n += 1
        else:
            if not is_redundant(child, new_population):
                new_population.append(child)
                n += 1
    # new
    n = 0
    n_new = int(len(population) - len(new_population))
    while n < n_new:
        child = generate_systems(1, config, use_hash, funcs)[0]
        if use_hash:
            s_hash = convert_system_to_hash(child)
            if not s_hash in new_population_hash_list:
                solved = get_individual_solved(s_hash)
                if not solved or not math.isinf(solved.fun):
                    new_population.append(child)
                    new_population_hash_list.append(s_hash)
                    n += 1
        else:
            if not is_redundant(child, new_population):
                new_population.append(child)
                n += 1

    print("\n#### GENERATE NEW POPULATION ####")
    beautified_systems = [beautify_system(p) for p in new_population]
    for i, system in enumerate(beautified_systems):
        print(f"new_systems {i}: {system}")

    save_systems_as_numpy_funcs(new_population, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return new_population


def crossover(parent1, parent2, config):
    child = copy.deepcopy(parent1)
    parent1_copy = copy.deepcopy(parent1)
    parent2_copy = copy.deepcopy(parent2)

    for i in range(len(child)):
        # Randomly pick an equation from parent1 and parent2
        eq1_idx = random.randint(0, len(parent1_copy) - 1)
        eq2_idx = random.randint(0, len(parent2_copy) - 1)
        eq1 = parent1_copy[eq1_idx]
        eq2 = parent2_copy[eq2_idx]

        # eq1 = parent1_copy.pop(eq1_idx)
        # eq2 = parent2_copy.pop(eq2_idx)

        # Perform the split and crossover for the selected equations
        split_idx = random.randint(0, min(len(eq1[1]), len(eq2[1])))
        child[i][1] = list(set(eq1[1][:split_idx] + eq2[1][split_idx:]))
        if config.DEBUG:print(len(eq1[1]), len(eq2[1]), split_idx, len(eq1[:split_idx]), len(eq2[1][split_idx:]))

        # new_terms = eq1[1][:split_idx] + eq2[1][split_idx:]
        # child.append([eq1[0], list(set(new_terms))])

    # same order of equation
    # for eq1, eq2 in zip(parent1, parent2):
    #     # Randomly split equation from parent1 or parent2
    #     split_idx = random.randint(0, min(len(eq1[1]), len(eq2[1])))
    #     if config.DEBUG: print(len(eq1[1]), len(eq2[1]), split_idx, len(eq1[:split_idx]), len(eq2[1][split_idx:]))
    #     eq = eq1[1][:split_idx] + eq2[1][split_idx:]
    #     child.append([eq1[0], list(set(eq))])

    if config.DEBUG:
        print(f'# Crossover Result:')
        print(parent1, parent2, "->", child)

    return child


def mutate(system, config, funcs):
    """Perform mutation on a system by altering one term"""
    mutated_system = copy.deepcopy(system)

    for i, eq in enumerate(mutated_system):
        # Choose a random term to mutate
        term_idx = random.randint(0, len(eq[1]) - 1)
        new_term = eq[1][term_idx]

        variables = [create_variable(j) for j in range(1, config.M + 1)]
        while new_term in eq[1]:  # check redundancy
            new_term = generate_term(variables, config, len(eq[1]) <= 1, funcs)

        # Replace
        if new_term is None:
            del eq[1][term_idx]
        else:
            eq[1][term_idx] = new_term

        mutated_system[i] = eq

    if config.DEBUG:
        print("mutated: ", system, "->", mutated_system)

    return mutated_system


if __name__ == "__main__":
    config = Config_Test()

    t = np.linspace(0, 10, 100)
    X0 = [1.8, 1.3]
    y_raw = solve_ivp(config.target.func, (t[0], t[-1]), X0, args=config.target.betas, t_eval=t,
                      method=config.ivp_method).y.T
    y_target = y_raw + np.random.normal(0.0, 0.0, y_raw.shape)

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for use_hash in [False, True]:
            for func in ["5",  "1,4,5", "1,2,4,5,9"]: # "4,5","1,4,5", "1,2,4,5", "1,2,4,5,9", "1,2,4,5,6,9", "1,2,3,4,5,6,9"
                reset_solved()
                print(f"reset: solved_map:{len(get_solved_map())}")
                funcs = get_functions(func)

                population = generate_population(config, use_hash, funcs)
                systems = load_systems(config.system_load_dir)

                history = []
                time_records = []
                for i in range(config.G):
                    print("#### SYSTEM EVALUATION ####")
                    start_time = time.time()
                    history.append([])

                    for j, system in enumerate(systems):
                        solved = None
                        if use_hash:
                            system_hash = convert_system_to_hash(population[j])
                            solved = get_individual_solved(system_hash)
                        if not solved:
                            ode_func = create_ode_function(system)
                            initial_guess = np.zeros(config.I * config.M)
                            solved = estimate_parameters(ode_func, X0, t, y_target, initial_guess,
                                                         config.minimize_method,
                                                         config.ivp_method, config.DEBUG)
                            if use_hash: add_individual_solved(system_hash, solved)

                        history[i].append((solved, population[j], ode_func))
                        print(
                            f"### Generation {i} #{j} | Error: {round(solved.fun, 4)} | System: {beautify_system(population[j])} ")

                    print(f"Completed Generation {i}: {time.time()-start_time} | term_map:{len(get_term_map())}, solved_map:{len(get_solved_map())}")
                    results.append({'use_hash': use_hash, 'func': funcs_to_str(funcs), 'generation':i, 'ts': time.time() - start_time})

                    if i < config.G - 1:
                        population = generate_new_population(history[i], population, config, funcs, use_hash)
                        systems = load_systems(config.system_load_dir)


    print(results)

    df = pd.DataFrame(results)
    print(df)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for method in df['use_hash'].unique():
        subset = df[df['use_hash'] == method]
        for func in subset['func'].unique():
            color = 'green' if 'cos' in func else 'red' if 'sin' in func else 'blue'
            subset_subset = subset[subset['func'] == func]
            axs[0].plot(subset_subset['generation']+1, subset_subset['ts'], '--' if method else '-'
                        , color = color, label=func +(" (Hash)" if method else ""))
    axs[0].set_title('Execution Time per Generation by Using Hash')
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Computation Time (seconds)')
    axs[0].legend()

    plt.tight_layout()
    plt.show()








