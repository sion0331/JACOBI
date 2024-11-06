import random as rd
import sympy as sp
from utils.symbolic_utils import t, create_variable, o
from utils.functions import f
from utils.numpy_conversion import save_systems_as_numpy_funcs


def generate_population(N, M, I, J, fOps, allow_composite, save_dir, DEBUG):
    print("\n#### GENERATE INITIAL POPULATION ####")
    systems = generate_systems(N, M, I, J, fOps, allow_composite)
    for i, system in enumerate(systems):
        if DEBUG: print(f"generate_system {i}: {system}")

    beautified_systems = beautify_system(systems)
    for i, system in enumerate(beautified_systems):
        print(f"beautified_systems {i}: {system}")

    # Save the system as NumPy functions
    save_systems_as_numpy_funcs(systems, save_dir)
    print(f"System saved as NumPy functions in {save_dir}\n")

    return systems


def generate_systems(N, M, I, J, fOps, allow_composite=True):
    variables = [create_variable(i) for i in range(1, M + 1)]

    systems = []
    for n in range(N):
        equations = []
        for m in range(M):
            terms = []
            for i in range(I):
                var_list = []
                rd.shuffle(variables)  # O(logN)
                j = 0
                for var in variables:
                    if rd.randint(0, 1) == 1:
                        func = f(rd.choice(fOps))
                        var = func(var)
                        j += 1
                        if allow_composite and j < J:  # limit applying composite to only once
                            if rd.randint(0, 1) == 1:
                                func = f(rd.choice(fOps))
                                var = func(var)
                                j += 1
                        var_list.append(var)
                        if j == J: break

                if var_list:
                    term = var_list[0]
                    for var in var_list[1:]:
                        operator = o(rd.randint(0, 3))
                        term = operator(term, var)
                    terms.append(term)

            equations.append([sp.diff(variables[m], t), terms])
        systems.append(equations)
    return systems


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


def beautify_system(systems):
    beautified_systems = []
    for system in systems:
        beta_count = 0
        beautified_equations = []
        for eq in system:
            beautified_eq = beautify_equation(eq, beta_count)
            beautified_equations.append(beautified_eq)
            beta_count += len(eq[1])
        beautified_systems.append(beautified_equations)
    return beautified_systems


def manual_lotka_systems():
    variables = [create_variable(i) for i in range(1, 3)]
    linear = f(5)
    mult = o(0)
    add = o(2)

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
             linear(variables[1])
         ]
         ]
    ]

    system2 = [
        [sp.diff(variables[0], t),
         [
             add(linear(variables[0]), linear(variables[1]))
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
             mult(linear(variables[0]), linear(variables[1])),
             linear(variables[1])
         ]
         ]
    ]

    system4 = [
        [sp.diff(variables[0], t),
         [
             linear(variables[0]),
             add(linear(variables[0]), linear(variables[1]))
         ]
         ],
        [sp.diff(variables[1], t),
         [
             linear(variables[0]),
             add(linear(variables[0]), linear(variables[1]))
         ]
         ]
    ]
    return [system0, system1, system2, system4]
