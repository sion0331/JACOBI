import random
import copy
import numpy as np

from population.initial_generation import generate_term, beautify_system, generate_systems
from utils.numpy_conversion import save_systems_as_numpy_funcs
from utils.symbolic_utils import create_variable


def generate_new_population(history, population, config):
    valid_entries = [(solved[0], ind) for solved, ind in zip(history, population) if solved[0].fun < 100]
    new_population = []

    sorted_population = []
    scores = []
    for solved, ind in sorted(valid_entries, key=lambda x: x[0].fun):
        sorted_population.append(ind)
        scores.append(solved.fun)
        if config.DEBUG:
            print(f'score: {solved.fun} | system: {ind}')

    # selection probabilities
    weights = [(max(scores) - score) / (max(scores) - min(scores) + 1e-6) for score in scores]  # Avoid division by zero
    probabilities = np.array(weights) / sum(weights)

    # elites
    num_parents = max(2, int(len(population) * config.elite_rate))
    elites = sorted_population[:num_parents]
    if config.DEBUG:
        for x in elites: print(f'elite: {x}')
    new_population.extend(elites)

    # new
    new_systems = generate_systems(int(len(population) * config.new_rate), config)
    new_population.extend(new_systems)

    while len(new_population) < len(population):
        parent1, parent2 = random.choices(sorted_population, weights=probabilities, k=2)

        # Crossover
        if random.random() < config.crossover_rate:
            child1, child2 = crossover(parent1, parent2, config)
        else:
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # Mutation
        if random.random() < config.mutation_rate:
            child1 = mutate(child1, config)
        if random.random() < config.mutation_rate:
            child2 = mutate(child2, config)

        # TODO - generate new population to add

        # Add children to the new population
        new_population.append(child1)
        if len(new_population) < len(population):
            new_population.append(child2)

    print("\n#### GENERATE NEW POPULATION ####")
    beautified_systems = [beautify_system(p) for p in new_population]
    for i, system in enumerate(beautified_systems):
        print(f"new_systems {i}: {system}")

    save_systems_as_numpy_funcs(new_population, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return new_population


def crossover(parent1, parent2, config):
    # Swap the i-th equation between the two parents
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    if config.DEBUG:
        print(f'# crossover | Before')
        print(f'child1:{child1}')
        print(f'child2:{child2}')

    i_1 = random.randint(0, config.M - 1)
    i_2 = random.randint(0, config.M - 1)
    child1[i_1][1], child2[i_2][1] = child2[i_2][1], child1[i_1][1]

    if config.DEBUG:
        print(f'# crossover | After')
        print(f'child1:{child1}')
        print(f'child2:{child2}')
    return child1, child2


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

# def mutate(system, DEBUG):
#     """Perform mutation on a system by altering one or more terms."""
#     if not DEBUG:
#         print(f'#### MUTATION ###')
#         print(f'{system}')
#     mutated_system = copy.deepcopy(system)
#
#     # Choose a random equation to mutate
#     eq_idx = random.randint(0, len(mutated_system) - 1)
#
#     # Mutate a single term in the chosen equation
#     mutated_equation = mutated_system[eq_idx]
#
#     if not DEBUG:
#         print(f'{eq_idx} | {system}')
#         print(f'SELECTED: {mutated_equation}')
#     # Randomly decide whether to change a variable, operator, or function
#
#     if random.random() < 0.33:
#         # Change a variable
#         print(f'variable')
#         mutated_equation = mutate_variable(mutated_equation)
#     elif random.random() < 0.66:
#         print(f'operator')
#         # Change an operator
#         mutated_equation = mutate_operator(mutated_equation)
#     else:
#         print(f'function')
#         # Change a function
#         mutated_equation = mutate_function(mutated_equation)
#
#     if not DEBUG:
#         print(f'MUTATED: {mutated_equation}')
#
#     mutated_equation = mutate_function(mutated_equation)
#     if DEBUG:
#         print(f'MUTATED: {mutated_equation}')
#
#     mutated_system[eq_idx] = mutated_equation
#     if DEBUG:
#         print(f'MUTATED: {mutated_system}')
#     return mutated_system
#
#
# def mutate_variable(equation):
#     """Randomly change a variable in the equation."""
#     pattern = r'x_\d+\(t\)'
#     variables = set(re.findall(pattern, str(equation)))
#     print(f'variables: {variables}')
#     variables = list(variables)
#     print(f'variables: {variables}')
#     if not variables:
#         return equation
#
#     var_to_change = random.choice(variables)
#     print(f'var_to_change: {var_to_change}')
#
#     new_var = random.choice([var for var in variables if var != var_to_change])
#     print(f'new_var: {new_var}')
#
#     mutated_equation = []
#     for term in equation:
#         if isinstance(term, str) and var_to_change in term:
#             print(f'term: {term} | {term.replace(var_to_change, new_var)}')
#             mutated_equation.append(term.replace(var_to_change, new_var))
#         else:
#             print(f'term: {term}')
#             mutated_equation.append(term)
#
#     return mutated_equation
#
#
# def mutate_operator(equation):
#     """Randomly change an operator in the equation."""
#     # Replace one of the operators (+, *, -, /) with another operator
#     operators = ['+', '-', '*', '/']
#     op_to_change = random.choice(operators)
#
#     print(f'op_to_change: {op_to_change}')
#     new_op = random.choice([op for op in operators if op != op_to_change])
#     print(f'new_op: {new_op}')
#
#     mutated_equation = []
#     for term in equation:
#         if isinstance(term, str) and op_to_change in term:
#             print(f'term op_to_change: {term.replace(op_to_change, new_op)}')
#             mutated_equation.append(term.replace(op_to_change, new_op))
#         else:
#             print(f'term: {term}')
#             mutated_equation.append(term)
#
#     print(f'mutated_equation: {mutated_equation}')
#     return mutated_equation
#
#
# def mutate_function(equation):
#     """Randomly change a function (sin, cos, exp) in the equation."""
#     # Replace a function with another function
#     functions = ['sin', 'cos', 'exp', 'log']
#     func_to_change = random.choice(functions)
#     print(f'func_to_change: {func_to_change}')
#
#     new_func = random.choice([func for func in functions if func != func_to_change])
#     print(f'new_func: {new_func}')
#
#     mutated_equation = []
#     for term in equation:
#         if isinstance(term, str) and func_to_change in term:
#             print(f'term: {term} | {term.replace(func_to_change, new_func)}')
#             mutated_equation.append(term.replace(func_to_change, new_func))
#         else:
#             print(f'term: {term}')
#             mutated_equation.append(term)
#
#     return mutated_equation
