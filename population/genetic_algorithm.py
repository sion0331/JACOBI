import random
import copy
import numpy as np
import sympy as sp

from population.initial_generation import generate_term, beautify_system, generate_systems
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.symbolic_utils import create_variable, is_redundant


def generate_new_population(history, population, config):
    valid_entries = [(solved[0], ind) for solved, ind in zip(history, population) if solved[0].fun < 100]
    new_population = []
    symbolic_set = []

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
    # elites = sorted_population[:num_parents]
    # if config.DEBUG:
    #     for x in elites: print(f'elite: {x}')
    # new_population.extend(elites)
    for system in sorted_population[:num_parents]:
        new_population.append(system)
        system_symbolic = [sum(sp.sympify(terms)) for _, terms in system]
        symbolic_set.append(system_symbolic)
        if config.DEBUG: print(f'elite: {system}')

    # crossover
    n = 0
    n_crossover = int(len(population) * config.crossover_rate / 2)
    while n < n_crossover:
        parent1, parent2 = random.choices(sorted_population, weights=probabilities, k=2)
        child = crossover(parent1, parent2, config)
        system_symbolic = [sum(sp.sympify(terms)) for _, terms in child]
        if not is_redundant(system_symbolic, symbolic_set):
            n += 1
            new_population.append(child)
            symbolic_set.append(system_symbolic)

    # mutation
    n = 0
    n_mutation = int(len(population) * config.mutation_rate)
    while n < n_mutation:
        parent = random.choices(sorted_population, weights=probabilities, k=1)
        child = mutate(parent[0], config)
        system_symbolic = [sum(sp.sympify(terms)) for _, terms in child]
        if not is_redundant(system_symbolic, symbolic_set):
            n += 1
            new_population.append(child)
            symbolic_set.append(system_symbolic)

    # new
    n = 0
    n_new = int(len(population) - len(new_population))
    while n < n_new:
        child = generate_systems(1, config)[0]
        system_symbolic = [sum(sp.sympify(terms)) for _, terms in child]
        if not is_redundant(system_symbolic, symbolic_set):
            n += 1
            new_population.append(child)
            symbolic_set.append(system_symbolic)

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
        # Randomly select equation from parent1 or parent2
        selected_eq = copy.deepcopy(eq1 if random.random() < 0.5 else eq2)
        child.append(selected_eq)

    if config.DEBUG:
        print(f'# Crossover Result:')
        print(f'Parent1: {parent1}')
        print(f'Parent2: {parent2}')
        print(f'Child: {child}')

    return child


def mutate(system, config):
    """Perform mutation on a system by altering one term"""
    if config.DEBUG:
        print(f'#### MUTATION ###')
        print(f'original_system: {system}')
    mutated_system = copy.deepcopy(system)

    # Choose a random equation to mutate
    eq_idx = random.randint(0, len(mutated_system) - 1)
    mutated_equation = mutated_system[eq_idx]

    # Choose a random term to mutate
    term_idx = random.randint(0, len(mutated_equation[1]) - 1)
    mutated_term = mutated_equation[1][term_idx]

    # Create a random term
    new_term = mutated_term
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    while new_term in mutated_equation[1]:  # check redundancy
        new_term = generate_term(variables, config, True)

    # Replace
    mutated_equation[1][term_idx] = new_term
    mutated_system[eq_idx] = mutated_equation

    if config.DEBUG:
        print(f'new_term: {new_term}')
        print(f'mutated_term: {mutated_term}')
        print(f'mutated_equation: {mutated_equation}')
        print(f'mutated_system: {mutated_system}')

    return mutated_system