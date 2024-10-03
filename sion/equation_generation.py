import random as rd
import sympy as sp
from symbolic_utils import t, create_variable, o
from functions import f, one


def generate_systems(N, M, I, J, fOps, allow_composite=True):
    variables = [create_variable(i) for i in range(1, M + 1)]

    systems = []
    for n in range(N):
        equations = []
        for m in range(M):
            terms = []
            for i in range(I):
                term = one(1)
                for var in variables:
                    if rd.randint(0, 1) == 1:  # todo - better logic
                        func = f(rd.choice(fOps))
                        temp = func(var)
                        if allow_composite:
                            for j in range(J - 1):
                                func = f(rd.choice(fOps))
                                temp = func(temp)

                        operator = o(rd.randint(0, 3)) # multiplication / division
                        term = operator(term, temp)
                terms.append(term)


            # equation = terms[0]
            # for term in terms[1:]:
            #     operator = o(rd.randint(2, 3))  # addition/subtraction
            #     equation = operator(equation, term)
            # equations.append(sp.Eq(sp.diff(variables[m], t), equation))

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
    for i, term in enumerate(eq[1]):
        term_str = str(term)
        for old, new in replacements.items():
            term_str = term_str.replace(old, new)
        beautified_terms.append(f"beta_{beta_start + i}*({term_str})")

    rhs_str = " + ".join(beautified_terms)
    return f"{lhs_str} = {rhs_str}"


def beautify_system(systems):
    beautified_systems=[]
    for system in systems:
        beta_count = 0
        beautified_equations = []
        for eq in system:
            beautified_eq = beautify_equation(eq, beta_count)
            beautified_equations.append(beautified_eq)
            beta_count += len(eq[1])
        beautified_systems.append(beautified_equations)
    return beautified_systems