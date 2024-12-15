import random
import copy
import numpy as np
import math

from population.initial_generation import generate_term, beautify_system, generate_systems
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.symbolic_utils import create_variable
from utils.mapping import convert_system_to_hash, get_individual_solved


def generate_new_population(history, population, config):
    new_population = []
    new_population_hash_list = []

    sorted_population = []
    scores = []
    valid_entries = [(solved[0], ind) for solved, ind in zip(history, population) if not math.isinf(solved[0].fun)]
    for solved, ind in sorted(valid_entries, key=lambda x: x[0].fun):
        sorted_population.append(ind)
        scores.append(solved.fun)
        if config.DEBUG:
            print(f'score: {solved.fun} | system: {ind}')

    ranked_indices = np.argsort(scores)  # Indices of sorted scores
    ranks = np.argsort(ranked_indices) + 1  # Assign ranks (1 = best)

    gamma = 0.9 # todo - tune as hyperparmeter
    weights = [gamma ** (rank - 1) for rank in ranks]
    probabilities = np.array(weights) / sum(weights)

    # elites
    num_parents = max(2, int(len(population) * config.elite_rate))
    for system in sorted_population[:num_parents]:
        new_population.append(system)
        s_hash = convert_system_to_hash(system)
        new_population_hash_list.append(s_hash)

    # crossover
    n = 0
    n_crossover = int(len(population) * config.crossover_rate / 2)
    while n < n_crossover:
        parent1, parent2 = random.choices(sorted_population, weights=probabilities, k=2)
        child = crossover(parent1, parent2, config)
        s_hash = convert_system_to_hash(child)
        if not s_hash in new_population_hash_list:
            solved = get_individual_solved(s_hash)
            if not solved or not math.isinf(solved.fun):
                new_population.append(child)
                new_population_hash_list.append(s_hash)
                n += 1

    # mutation
    n = 0
    n_mutation = int(len(population) * config.mutation_rate)
    while n < n_mutation:
        parent = random.choices(sorted_population, weights=probabilities, k=1)
        child = mutate(parent[0], config)
        s_hash = convert_system_to_hash(child)
        if not s_hash in new_population_hash_list:
            solved = get_individual_solved(s_hash)
            if not solved or not math.isinf(solved.fun):
                new_population.append(child)
                new_population_hash_list.append(s_hash)
                n += 1

    # new
    n = 0
    n_new = int(len(population) - len(new_population))
    while n < n_new:
        child = generate_systems(1, config)[0]
        s_hash = convert_system_to_hash(child)
        if not s_hash in new_population_hash_list:
            solved = get_individual_solved(s_hash)
            if not solved or not math.isinf(solved.fun):
                new_population.append(child)
                new_population_hash_list.append(s_hash)
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


def mutate(system, config):
    """Perform mutation on a system by altering one term"""
    mutated_system = copy.deepcopy(system)

    for i, eq in enumerate(mutated_system):
        # Choose a random term to mutate
        term_idx = random.randint(0, len(eq[1]) - 1)
        new_term = eq[1][term_idx]

        variables = [create_variable(j) for j in range(1, config.M + 1)]
        while new_term in eq[1]:  # check redundancy
            new_term = generate_term(variables, config, len(eq[1]) <= 1)

        # Replace
        if new_term is None:
            del eq[1][term_idx]
        else:
            eq[1][term_idx] = new_term

        mutated_system[i] = eq

    if config.DEBUG:
        print("mutated: ", system, "->", mutated_system)

    return mutated_system
