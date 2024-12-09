import random
import copy
import numpy as np
import math

from population.initial_generation import generate_term, beautify_system, generate_systems
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.symbolic_utils import create_variable, is_redundant


def generate_new_population(history, population, config):
    valid_entries = [(solved[0], ind) for solved, ind in zip(history, population) if not math.isinf(solved[0].fun)]
    new_population = []

    sorted_population = []
    scores = []
    for solved, ind in sorted(valid_entries, key=lambda x: x[0].fun):
        sorted_population.append(ind)
        scores.append(solved.fun)
        if config.DEBUG:
            print(f'score: {solved.fun} | system: {ind}')

    # selection probabilities
    weights = [(max(scores) - score) / (max(scores) - min(scores) + 1e-6) for score in scores]
    probabilities = np.array(weights) / sum(weights)

    # elites
    num_parents = max(2, int(len(population) * config.elite_rate))
    elites = sorted_population[:num_parents]
    if config.DEBUG:
        for x in elites: print(f'elite: {x}')
    new_population.extend(elites)
    # for system in sorted_population[:num_parents]:
    #     new_population.append(system)
    #     if config.DEBUG: print(f'elite: {system}')

    # crossover
    n = 0
    n_crossover = int(len(population) * config.crossover_rate / 2)
    while n < n_crossover:
        parent1, parent2 = random.choices(sorted_population, weights=probabilities, k=2)
        child = crossover(parent1, parent2, config)
        if not is_redundant(child, new_population):
            n += 1
            new_population.append(child)

    # mutation
    n = 0
    n_mutation = int(len(population) * config.mutation_rate)
    while n < n_mutation:
        parent = random.choices(sorted_population, weights=probabilities, k=1)
        child = mutate(parent[0], config)
        if not is_redundant(child, new_population):
            n += 1
            new_population.append(child)

    # new
    n = 0
    n_new = int(len(population) - len(new_population))
    while n < n_new:
        child = generate_systems(1, config)[0]
        if not is_redundant(child, new_population):
            n += 1
            new_population.append(child)

    print("\n#### GENERATE NEW POPULATION ####")
    beautified_systems = [beautify_system(p) for p in new_population]
    for i, system in enumerate(beautified_systems):
        print(f"new_systems {i}: {system}")

    save_systems_as_numpy_funcs(new_population, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return new_population


def crossover(parent1, parent2, config):
    child = []
    for eq1, eq2 in zip(parent1, parent2):
        # Randomly split equation from parent1 or parent2
        split_idx =  random.randint(0, min(len(eq1[1]),len(eq2[1])))
        if config.DEBUG: print(len(eq1[1]), len(eq2[1]), split_idx, len(eq1[:split_idx]), len(eq2[1][split_idx:]))
        eq = eq1[1][:split_idx] + eq2[1][split_idx:]
        child.append([eq1[0], list(set(eq))])

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
            new_term = generate_term(variables, config, len(eq[1])<=1)

        # Replace
        if new_term is None:
            del eq[1][term_idx]
        else:
            eq[1][term_idx] = new_term

        mutated_system[i] = eq

    if config.DEBUG:
        print("mutated: ", system, "->", mutated_system)

    return mutated_system
