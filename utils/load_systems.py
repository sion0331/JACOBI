import re


def load_systems(filename):
    """Load the system of differential equations from the generated file."""
    with open(filename, 'r') as f:
        code = f.read()
    exec(code, globals())

    systems = []
    for key in globals():
        if re.match(r'system_\d+', key):  # Look for system_0, system_1, etc.
            systems.append(globals()[key])

    return systems


def count_betas_and_equations(filename):
    """Count the number of beta parameters and equations in the generated file."""
    with open(filename, 'r') as f:
        code = f.read()

    # Split the code into individual equation functions
    eq_functions = re.findall(r'def eq_\d+\(.*?\):.*?return.*?(?=def|\Z)', code, re.DOTALL)

    total_betas = 0
    for func in eq_functions:
        # Count the number of 'betas[n]' in each equation function
        beta_count = len(re.findall(r'betas\[\d+\]', func))
        total_betas += beta_count

    num_equations = len(eq_functions)
    return total_betas, num_equations


def create_ode_function(system):
    """Create an ODE function that can be used with scipy's odeint."""

    def ode_func(X, t, *betas):
        return system(X, betas, t)

    return ode_func
