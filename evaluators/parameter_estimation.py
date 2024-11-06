import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


def simulate_system(ode_func, X0, t, betas, method):
    """Simulate the system using the given parameters."""
    return solve_ivp(ode_func, (t[0], t[-1]), X0, t_eval=t, args=betas, method=method)


def calculate_error(simulated, observed, DEBUG):
    """Calculate the mean squared error between simulated and observed data."""
    if simulated.status == -1:
        if DEBUG: print(f"Shape mismatch: simulated {simulated.y.T.shape}, observed {observed.shape}. Skipping...")
        return float('inf')
    return np.mean((simulated.y.T - observed) ** 2)


def objective_function(betas, ode_func, X0, t, observed_data, method, DEBUG):
    """Objective function to minimize: the error between simulated and observed data."""
    w_reg = 0.01
    simulated = simulate_system(ode_func, X0, t, tuple(betas), method)
    reg = w_reg * np.sum(betas ** 2)
    error = calculate_error(simulated, observed_data, DEBUG)
    if DEBUG: print(f"betas: {betas} | error: {error} | regularization: {reg}")
    return error + reg


def estimate_parameters(ode_func, X0, t, observed_data, initial_guess, method, DEBUG):
    """Estimate the parameters using scipy's minimize function."""
    result = minimize(
        objective_function,
        initial_guess,
        args=(ode_func, X0, t, observed_data, method, DEBUG),
        method='Nelder-Mead',
        tol=1e-6,
        options={'maxiter': 100}  # 'disp': True, 'gtol': 1e-6, 'eps': 1e-10}
    )
    if DEBUG: print("estimated parameters: ", result)
    return result
