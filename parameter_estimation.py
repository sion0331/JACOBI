import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import re


def load_system(filename):
    """Load the system of differential equations from the generated file."""
    with open(filename, 'r') as f:
        code = f.read()
    exec(code, globals())
    return globals()['system']


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
    def ode_func(X, t, betas):
        # Pad X with zeros to match the expected input size
        X_padded = np.pad(X, (0, 5 - len(X)), 'constant')
        return system(X_padded, betas, t)
    return ode_func


def simulate_system(ode_func, X0, t, betas):
    """Simulate the system using the given parameters."""
    return odeint(ode_func, X0, t, args=(betas,))


def calculate_error(simulated, observed):
    """Calculate the mean squared error between simulated and observed data."""
    return np.mean((simulated - observed) ** 2)


def objective_function(betas, ode_func, X0, t, observed_data):
    """Objective function to minimize: the error between simulated and observed data."""
    simulated = simulate_system(ode_func, X0, t, betas)
    return calculate_error(simulated, observed_data)


def estimate_parameters(system, X0, t, observed_data, initial_guess):
    """Estimate the parameters using scipy's minimize function."""
    ode_func = create_ode_function(system)

    result = minimize(
        objective_function,
        initial_guess,
        args=(ode_func, X0, t, observed_data),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    return result.x


def main():
    # Load the system
    system = load_system('differential_equations.txt')

    # Count the number of beta parameters and equations
    num_betas, num_equations = count_betas_and_equations('differential_equations.txt')
    print(f"Number of beta parameters: {num_betas}")
    print(f"Number of equations: {num_equations}")

    # Define your observed data here
    # For demonstration, we'll generate synthetic data
    t = np.linspace(0, 10, 100)
    X0 = [1.0] * num_equations  # Initial conditions for only the necessary equations
    true_betas = np.random.rand(num_betas)  # True parameter values
    ode_func = create_ode_function(system)
    observed_data = simulate_system(ode_func, X0, t, true_betas)

    # Add some noise to the observed data
    observed_data += np.random.normal(0, 0.1, observed_data.shape)

    # Initial guess for parameters (random values between 0 and 1)
    initial_guess = np.random.rand(num_betas)

    # Estimate parameters
    estimated_betas = estimate_parameters(system, X0, t, observed_data, initial_guess)

    # Print results
    print("True parameters:", true_betas)
    print("Estimated parameters:", estimated_betas)

    # Simulate the system with estimated parameters
    simulated_data = simulate_system(ode_func, X0, t, estimated_betas)

    # Calculate and print the error
    error = calculate_error(simulated_data, observed_data)
    print("Mean squared error:", error)

if __name__ == "__main__":
    main()