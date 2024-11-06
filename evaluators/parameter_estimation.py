import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def simulate_system(ode_func, X0, t, betas, method):
    """Simulate the system using the given parameters."""
    return solve_ivp(ode_func, (t[0], t[-1]), X0, t_eval=t, args=betas, method=method)


def calculate_error(simulated, observed, DEBUG):
    """Calculate the mean squared error between simulated and observed data."""
    if simulated.status == -1:
        # Return a large error if shapes do not match
        if DEBUG: print(f"Shape mismatch: simulated {simulated.y.T.shape}, observed {observed.shape}. Skipping...")
        return float('inf')

    error = np.mean((simulated.y.T - observed) ** 2)
    if DEBUG: print("error: ", error)
    return error


def objective_function(betas, ode_func, X0, t, observed_data, method, DEBUG):
    """Objective function to minimize: the error between simulated and observed data."""
    simulated = simulate_system(ode_func, X0, t, tuple(betas), method)
    return calculate_error(simulated, observed_data, DEBUG)


def estimate_parameters(ode_func, X0, t, observed_data, initial_guess, method, DEBUG):
    """Estimate the parameters using scipy's minimize function."""
    result = minimize(
        objective_function,
        initial_guess,
        args=(ode_func, X0, t, observed_data, method, DEBUG),
        # method='L-BFGS-B',
        options={'maxiter': 100}
    )
    if DEBUG: print("estimated parameters: ", result)
    return result.x
