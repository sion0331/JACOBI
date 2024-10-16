import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize


def simulate_system(ode_func, X0, t, betas):
    """Simulate the system using the given parameters."""
    return odeint(ode_func, X0, t, args=tuple(betas))


def calculate_error(simulated, observed):
    """Calculate the mean squared error between simulated and observed data."""
    return np.mean((simulated - observed) ** 2)


def objective_function(betas, ode_func, X0, t, observed_data):
    """Objective function to minimize: the error between simulated and observed data."""
    simulated = simulate_system(ode_func, X0, t, tuple(betas))
    return calculate_error(simulated, observed_data)


def estimate_parameters(ode_func, X0, t, observed_data, initial_guess):
    """Estimate the parameters using scipy's minimize function."""
    result = minimize(
        objective_function,
        initial_guess,
        args=(ode_func, X0, t, observed_data),
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )
    return result.x
