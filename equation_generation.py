import random as rd
import sympy as sp
from symbolic_utils import t, create_variable, o
from functions import f


def generate_system(n, m, k, fOps, allow_composite=True):
    equations = []
    variables = [create_variable(i) for i in range(1, n + 1)]

    for i in range(n):
        terms = []
        for _ in range(m):
            func = f(rd.choice(fOps))
            var = rd.choice(variables)  # Choose from all available variables
            term = func(var)
            if allow_composite:
                for _ in range(k - 1):
                    func = f(rd.choice(fOps))
                    term = func(term)
            terms.append(term)

        equation = terms[0]
        for term in terms[1:]:
            operator = o(rd.randint(0, 1))  # Only use multiplication and division
            equation = operator(equation, term)

        equations.append(sp.Eq(sp.diff(variables[i], t), equation))

    return equations


def beautify_equation(eq, beta_start):
    lhs, rhs = eq.args
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
    terms = sp.Mul.make_args(rhs) if rhs.func == sp.Mul else sp.Add.make_args(rhs)

    beautified_terms = []
    for i, term in enumerate(terms):
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
        beta_count += len(sp.Mul.make_args(eq.rhs)) if eq.rhs.func == sp.Mul else len(sp.Add.make_args(eq.rhs))
    return beautified_equations
