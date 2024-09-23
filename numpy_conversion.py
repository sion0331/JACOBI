import sympy as sp
import numpy as np


def eq_to_numpy_func(eq, betas):
    rhs = eq.rhs
    variables = list(rhs.free_symbols)
    variables.sort(key=lambda x: x.name)
    lambda_func = sp.lambdify(variables + betas, rhs, modules=['numpy'])

    def numpy_func(X, betas):
        return lambda_func(*X, *betas)

    return numpy_func


def save_system_as_numpy_funcs(system, filename):
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")

        f.write("def diff(x, t):\n")
        f.write("    return np.gradient(x, t)\n\n")

        f.write("def diff2(x, t):\n")
        f.write("    return np.gradient(np.gradient(x, t), t)\n\n")

        for i, eq in enumerate(system):
            lhs, rhs = eq.args
            var = str(lhs.args[0].func)

            f.write(f"def eq_{i}(X, betas, t):\n")
            f.write(f"    # d{var}/dt = {eq.rhs}\n")
            f.write("    x_1, x_2, x_3, x_4, x_5 = X\n")

            terms = []
            for j, term in enumerate(rhs.args):
                term_str = str(term)
                term_str = term_str.replace('x_1(t)', 'x_1').replace('x_2(t)', 'x_2').replace('x_3(t)', 'x_3').replace(
                    'x_4(t)', 'x_4').replace('x_5(t)', 'x_5')
                term_str = term_str.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan').replace(
                    'exp', 'np.exp')
                terms.append(f"betas[{j}] * ({term_str})")

            rhs_str = " + ".join(terms)
            f.write(f"    return {rhs_str}\n\n")

        f.write("def system(X, betas, t):\n")
        f.write("    return np.array([" + ", ".join(f"eq_{i}(X, betas, t)" for i in range(len(system))) + "])\n")
