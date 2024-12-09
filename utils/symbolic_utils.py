import sympy as sp

t = sp.Symbol('t')

def create_variable(i):
    return sp.Function(f'x_{i}')(t)

def diff(expr, var=t):
    return sp.diff(expr, var)

def diff2(expr, var=t):
    return sp.diff(expr, var, 2)

def o(j):
    switcher_o = {
        0: lambda a, b: a * b,  # Multiplication
        1: lambda a, b: a / b,  # Division
        2: lambda a, b: a + b,  # Addition
        3: lambda a, b: a - b   # Subtraction
    }
    return switcher_o.get(j, "Invalid")

def is_redundant(system, population):
    system_symbolic = [sum(sp.sympify(terms)) for _, terms in system]
    # print("system_symbolic: ", system_symbolic)

    for population_system in population:
        population_symbolic = [sum(sp.sympify(terms)) for _, terms in population_system]

        check = True
        for (rhs1, rhs2) in zip(system_symbolic, population_symbolic):
            if sp.simplify(rhs1 - rhs2) != 0:
                check = False
                continue

        if check:
            print("####### REDUNDANT ###### ")
            print(system, population_system)
            return True

    return False
    #
    # print("is_redundant")
    # print(system)
    # # Convert each system in the population to symbolic expressions
    # # mutated_symbolic = [sp.sympify(equation[1]) for equation in system]
    # # print(mutated_symbolic)
    # mutated_symbolic = [sum(sp.sympify(terms)) for _, terms in system]
    # print("Mutated symbolic:", mutated_symbolic)
    #
    #
    # for s in population:
    #     system_symbolic = [sum(sp.sympify(terms)) for _, terms in s]
    #     print("System symbolic:", system_symbolic)
    #
    #     # system_symbolic = [sp.sympify(equation[1]) for equation in system]
    #     # print(system_symbolic)
    #
    #     # Compare equations term by term
    #     if all(sp.simplify(mut_eq - sys_eq) == 0 for mut_eq, sys_eq in zip(mutated_symbolic, system_symbolic)):
    #         print("true)")
    #         return True  # Redundant system found
    #
    # return False  # No redundancy found