import sympy as sp


def eq_to_numpy_func(eq, betas):
    rhs = eq.rhs
    variables = list(rhs.free_symbols)
    variables.sort(key=lambda x: x.name)
    lambda_func = sp.lambdify(variables + betas, rhs, modules=['numpy'])

    def numpy_func(X, betas):
        return lambda_func(*X, *betas)

    return numpy_func


def save_systems_as_numpy_funcs(systems, filename):
    with open(filename, 'w') as f:
        f.write("import numpy as np\n\n")
        #
        # f.write("def diff(x, t):\n")
        # f.write("    return np.gradient(x, t)\n\n")
        #
        # f.write("def diff2(x, t):\n")
        # f.write("    return np.gradient(np.gradient(x, t), t)\n\n")

        for n, system in enumerate(systems):
            M = len(system)
            j = 0
            for i, eq in enumerate(system):
                lhs = eq[0]
                var = str(lhs.args[0].func)

                f.write(f"def eq_{n}_{i}(X, betas, t):\n")
                f.write(f"    # d{var}/dt = {eq[1]}\n")
                f.write("    ")
                f.write(", ".join([f"x_{j + 1}" for j in range(M)]))  # Using j+1 to start from x_1
                f.write(" = X\n")

                terms = []
                if not eq[1]:
                    terms.append('0')
                else:
                    for term in eq[1]:
                        # todo - extend to more than 5 variables
                        term_str = str(term)
                        term_str = term_str.replace('x_1(t)', 'x_1').replace('x_2(t)', 'x_2').replace('x_3(t)',
                                                                                                      'x_3').replace(
                            'x_4(t)', 'x_4').replace('x_5(t)', 'x_5')
                        term_str = term_str.replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan',
                                                                                                      'np.tan').replace(
                            'exp', 'np.exp')
                        terms.append(f"betas[{j}] * ({term_str})")
                        j += 1

                rhs_str = " + ".join(terms)
                f.write(f"    return {rhs_str}\n\n")

            f.write(f"def system_{n}(X, betas, t):\n")
            f.write(
                "    return np.array([" + ", ".join(f"eq_{n}_{i}(X, betas, t)" for i in range(len(system))) + "])\n\n")
