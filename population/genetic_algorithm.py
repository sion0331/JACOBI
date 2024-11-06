import random
import copy


def generate_new_population(scores, population, DEBUG):
    print("\n#### GENERATE NEW POPULATION | BEFORE ####")
    for s, x in sorted(zip(scores, population)):
        print(f'score: {s} | system: {x}')

    elitism_rate = 0.1
    crossover_rate = 0.9
    mutation_rate = 0.1

    new_population = []
    sorted_population = [x for _, x in sorted(zip(scores, population))]
    num_parents = max(2, int(len(population) * elitism_rate))
    parents = sorted_population[:num_parents]
    if DEBUG:
        for x in parents: print(f'parents: {x}')
    new_population.extend(sorted_population[:num_parents])

    while len(new_population) < len(population):
        parent1, parent2 = random.sample(parents, 2)

        # Crossover
        if random.random() < crossover_rate:
            child1, child2 = crossover(parent1, parent2, DEBUG)
        else:
            child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

        # TODO - Mutation
        # if random.random() < mutation_rate:
        #     child1 = mutate(child1)
        # if random.random() < mutation_rate:
        #     child2 = mutate(child2)

        # Add children to the new population
        new_population.append(child1)
        if len(new_population) < len(population):
            new_population.append(child2)

    print("#### GENERATE NEW POPULATION | AFTER ####")
    for x in new_population:
        print(x)
    print()
    return new_population


def crossover(parent1, parent2, DEBUG):
    # Swap the i-th equation between the two parents
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    if DEBUG:
        print(f'# crossover | Before')
        print(f'child1:{child1}')
        print(f'child2:{child2}')
    for i in range(len(parent1)):
        if random.random() < 0.5:
            # Swap the i-th equation between the two parents
            child1[i], child2[i] = child2[i], child1[i]
    if DEBUG:
        print(f'# crossover | After')
        print(f'child1:{child1}')
        print(f'child2:{child2}')
    return child1, child2


def mutate(system):
    """Perform mutation on a system by altering one or more terms."""
    mutated_system = copy.deepcopy(system)

    # Choose a random equation to mutate
    eq_idx = random.randint(0, len(mutated_system) - 1)

    # Mutate a single term in the chosen equation
    mutated_equation = mutated_system[eq_idx]

    # Randomly decide whether to change a variable, operator, or function
    if random.random() < 0.33:
        # Change a variable
        mutated_equation = mutate_variable(mutated_equation)
    elif random.random() < 0.66:
        # Change an operator
        mutated_equation = mutate_operator(mutated_equation)
    else:
        # Change a function
        mutated_equation = mutate_function(mutated_equation)

    mutated_system[eq_idx] = mutated_equation
    return mutated_system


def mutate_variable(equation):
    """Randomly change a variable in the equation."""
    # For simplicity, we assume the equation contains variables like x_1, x_2, etc.
    # Replace a variable with another randomly chosen variable
    variables = ['x_1', 'x_2', 'x_3', 'x_4', 'x_5']
    var_to_change = random.choice(variables)
    new_var = random.choice(variables)

    return equation.replace(var_to_change, new_var)


def mutate_operator(equation):
    """Randomly change an operator in the equation."""
    # Replace one of the operators (+, *, -, /) with another operator
    operators = ['+', '-', '*', '/']
    op_to_change = random.choice(operators)
    new_op = random.choice(operators)

    return equation.replace(op_to_change, new_op)


def mutate_function(equation):
    """Randomly change a function (sin, cos, exp) in the equation."""
    # Replace a function with another function
    functions = ['sin', 'cos', 'exp', 'log']
    func_to_change = random.choice(functions)
    new_func = random.choice(functions)

    return equation.replace(func_to_change, new_func)
