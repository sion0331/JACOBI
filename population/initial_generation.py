import copy
import random as rd
import sympy as sp
from utils.symbolic_utils import t, create_variable, o, is_redundant
from utils.functions import f
from utils.numpy_conversion import save_systems_as_numpy_funcs


def generate_population(config):
    print("\n#### GENERATE INITIAL POPULATION ####")
    systems = generate_systems(config.N, config)
    for i, system in enumerate(systems):
        if config.DEBUG: print(f"generate_system {i}: {system}")

    beautified_systems = [beautify_system(p) for p in systems]
    for i, system in enumerate(beautified_systems):
        print(f"beautified_systems {i}: {system}")

    # Save the system as NumPy functions
    save_systems_as_numpy_funcs(systems, config.system_save_dir)
    print(f"System saved as NumPy functions in {config.system_save_dir}\n")

    return systems


def generate_systems(N, config):
    variables = [create_variable(i) for i in range(1, config.M + 1)]
    v = copy.deepcopy(variables)

    systems = []
    symbolic_set = []
    n = 0
    while n<N:
        system = []
        for m in range(config.M):
            terms = []
            for i in range(config.I):
                term = generate_term(v, config, i == 0)
                if term is not None and not term in terms:
                    terms.append(term)
            system.append([sp.diff(variables[m], t), terms])

        # system_symbolic = [sum(sp.sympify(terms)) for _, terms in system]
        system_symbolic = [sum(terms) for _, terms in system]
        if not is_redundant(system, systems):
            systems.append(system)
            symbolic_set.append(system_symbolic)
            n+=1

    return systems


def generate_term(variables, config, non_empty):
    rd.shuffle(variables)  # O(logN)
    term = None
    var_list = []
    j = 0
    for var in variables:
        # equation to have at least one term
        if (non_empty and j == 0) or rd.randint(0, 1) == 1:
            func = f(rd.choice(config.f0ps))
            var = func(var)
            j += 1
            if config.allow_composite and j < config.J:  # limit applying composite to only once
                if rd.randint(0, 1) == 1:
                    func = f(rd.choice(config.fOps))
                    var = func(var)
                    j += 1
            var_list.append(var)
            if j == config.J: break

    if var_list:
        term = var_list[0]
        for var in var_list[1:]:
            operator = o(rd.randint(0, 1))
            term = operator(term, var)

    return term


def beautify_equation(eq, beta_start):
    lhs = eq[0]
    lhs_str = f"d{str(lhs.args[0])}/dt"

    replacements = {
        "sin": "sin", "cos": "cos", "tan": "tan", "exp": "exp",
        "**2": "²", "**3": "³", "**4": "⁴",
        "Derivative": "d",
        "(t)": "",
    }

    for i in range(1, 100):  # Adjust range as needed
        replacements[f"x_{i}(t)"] = f"x_{i}"
        replacements[f"Derivative(x_{i}(t), t)"] = f"dx_{i}/dt"
        replacements[f"Derivative(x_{i}(t), (t, 2))"] = f"d²x_{i}/dt²"

    # Split the right-hand side into individual terms
    beautified_terms = []
    if not eq[1]:
        beautified_terms.append('0')
    else:
        for i, term in enumerate(eq[1]):
            term_str = str(term)
            for old, new in replacements.items():
                term_str = term_str.replace(old, new)
            beautified_terms.append(f"beta_{beta_start + i}*({term_str})")

    rhs_str = " + ".join(beautified_terms)

    return f"{lhs_str} = {rhs_str}"


def beautify_system(system):
    beta_count = 0
    beautified_equations = []
    for eq in system:
        beautified_eq = beautify_equation(eq, beta_count)
        beautified_equations.append(beautified_eq)
        beta_count += len(eq[1])

    return beautified_equations


def manual_lotka_systems():
    variables = [create_variable(i) for i in range(1, 3)]
    linear = f(5)
    mult = o(0)
    div = o(1)

    system0 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[1]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
    ]

    system1 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             linear(variables[1])
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ]
    ]

    system2 = [
        [sp.diff(variables[0], t),
         [
             div(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             linear(variables[1])
         ]
         ]
    ]

    system3 = [
        [sp.diff(variables[0], t),
         [
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             div(linear(variables[0]), linear(variables[1])),
         ]
         ]
    ]

    system4 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             div(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             mult(linear(variables[0]), linear(variables[1]))
         ]
         ]
    ]
    return [system0, system1, system2, system3, system4]
