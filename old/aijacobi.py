import sympy as sp
import random as rd
import numpy as np
import inspect


# Define basic mathematical functions
def one(x):
    return np.ones_like(x)


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def exp(x):
    return np.exp(x)


def linear(x):
    return x


def square(x):
    return x ** 2


def cube(x):
    return x ** 3


def quart(x):
    return x ** 4


# First, let's define a dictionary of all available functions:
all_functions = {
    0: ("one", one),
    1: ("sin", sin),
    2: ("cos", cos),
    3: ("tan", tan),
    4: ("exp", exp),
    5: ("linear", linear),
    6: ("square", square),
    7: ("cube", cube),
    8: ("quart", quart)
}

print("Available functions:")
for key, (name, _) in all_functions.items():
    print(f"{key}: {name}")

allowed_functions = input("Enter the numbers of the functions you want to allow (comma-separated): ")
allowed_functions = [int(x.strip()) for x in allowed_functions.split(',')]

# Create fOps based on user input
fOps = allowed_functions

# Define the symbolic variable 't'
t = sp.Symbol('t')

# Define the symbolic functions 'x(t)', 'y(t)', etc.
x = sp.Function('x')(t)
y = sp.Function('y')(t)
z = sp.Function('z')(t)
w = sp.Function('w')(t)


# Define a function to compute the first derivative of an expression with respect to 'var' (default is 't')
def diff(expr, var=t):
    return sp.diff(expr, var)


# Define a function to compute the second derivative of an expression with respect to 'var' (default is 't')
def diff2(expr, var=t):
    return sp.diff(expr, var, 2)


# Define a function to map an index to a corresponding mathematical function
def f(i):
    return all_functions.get(i, ("Invalid", lambda x: x))[1]


# Define a function to map an index to a corresponding operator
def o(j):
    switcher_o = {
        0: lambda a, b: a * b,  # Multiplication
        1: lambda a, b: a / b,  # Division
        2: lambda a, b: a + b,  # Addition
        3: lambda a, b: a - b  # Subtraction
    }
    return switcher_o.get(j, "Invalid")  # Return the corresponding operator or "Invalid" if index is out of range


def beautify_equation(eq, beta_start):
    lhs, rhs = eq.args
    lhs_str = f"d{str(lhs.args[0])}/dt"
    rhs_str = str(rhs)

    # Replace function names with more readable versions
    replacements = {
        "sin": "sin",
        "cos": "cos",
        "tan": "tan",
        "exp": "exp",
        "x(t)": "x",
        "y(t)": "y",
        "z(t)": "z",
        "w(t)": "w",
        "**2": "²",
        "**3": "³",
        "**4": "⁴",
        "Derivative(x(t), t)": "dx/dt",
        "Derivative(y(t), t)": "dy/dt",
        "Derivative(z(t), t)": "dz/dt",
        "Derivative(w(t), t)": "dw/dt",
        "Derivative(x(t), (t, 2))": "d²x/dt²",
        "Derivative(y(t), (t, 2))": "d²y/dt²",
        "Derivative(z(t), (t, 2))": "d²z/dt²",
        "Derivative(w(t), (t, 2))": "d²w/dt²"
    }

    for old, new in replacements.items():
        rhs_str = rhs_str.replace(old, new)

    # Insert beta parameters into the equation
    terms = rhs_str.split('+')
    rhs_str_with_betas = ' + '.join([f'beta_{i + beta_start}*({term.strip()})' for i, term in enumerate(terms)])

    return f"{lhs_str} = {rhs_str_with_betas}"


def generate_system(n, m, k, allow_composite=True):
    equations = []
    variables = [x, y, z, w]  # Extend this list if more variables are needed

    for i in range(n):
        terms = []
        for _ in range(m):
            func = f(rd.choice(fOps))  # Randomly select a function from allowed functions
            var = variables[rd.randint(0, len(variables) - 1)]  # Randomly select a variable
            term = func(var)
            if allow_composite:
                for _ in range(k - 1):
                    func = f(rd.choice(fOps))  # Randomly select another function from allowed functions
                    term = func(term)
            terms.append(term)

        # Combine terms using random operators
        equation = terms[0]
        for term in terms[1:]:
            operator = o(rd.randint(0, 3))  # Randomly select an operator
            equation = operator(equation, term)

        equations.append(sp.Eq(sp.diff(variables[i], t), equation))

    return equations


def eq_to_numpy_func(eq, betas):
    # Extract the right-hand side of the equation
    rhs = eq.rhs

    # Get all the variables in the equation
    variables = list(rhs.free_symbols)
    variables.sort(key=lambda x: x.name)  # Sort variables to ensure consistent order

    # Create a lambda function
    lambda_func = sp.lambdify(variables + betas, rhs, modules=['numpy'])

    # Create a wrapper function that takes a single array argument
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

        betas = [sp.Symbol(f'beta_{i}') for i in range(len(system))]

        for i, eq in enumerate(system):
            lhs, rhs = eq.args
            var = str(lhs.args[0].func)
            rhs_str = sp.printing.pycode(rhs)

            f.write(f"def eq_{i}(X, betas, t):\n")
            f.write(f"    # d{var}/dt = {rhs}\n")
            f.write("    x, y, z, w = X\n")
            f.write(f"    return {rhs_str}\n\n")

        f.write("def system(X, betas, t):\n")
        f.write("    return np.array([" + ", ".join(f"eq_{i}(X, betas, t)" for i in range(len(system))) + "])\n")


# Example usage
n = 2  # Number of equations
m = 3  # Maximum number of terms in a particular differential equation
k = 2  # Maximum number of functions per feature in a differential equation
allow_composite = True  # Set to False if you don't want composite functions

system = generate_system(n, m, k, allow_composite)

# Print the beautified equations
beta_count = 0
for eq in system:
    print(beautify_equation(eq, beta_count))
    beta_count += len(eq.rhs.args)  # Count the number of terms in each equation

# Save the system as NumPy functions
save_system_as_numpy_funcs(system, "differential_equations.txt")
print("\nSystem saved as NumPy functions in 'differential_equations.txt'")
