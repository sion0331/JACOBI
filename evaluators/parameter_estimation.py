import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, OptimizeResult


def simulate_system(ode_func, X0, t, betas, method, DEBUG):
    """Simulate the system using the given parameters."""
    start_time = time.time()
    timeout = 1

    def stop_event(*args):
        ts = (time.time() - start_time)
        if ts > timeout:
            if not DEBUG: print(f'solve_vip exceeded timeout: {ts} > {timeout}')
            raise TookTooLong()
        return timeout - ts

    stop_event.terminal = True

    try:
        return solve_ivp(ode_func, (t[0], t[-1]), X0, t_eval=t, args=betas, method=method, events=stop_event)
    except:
        return None


def calculate_error(simulated, observed, DEBUG):
    """Calculate the mean squared error between simulated and observed data."""
    if simulated.status == -1:
        if DEBUG: print(f"Shape mismatch: simulated {simulated.y.T.shape}, observed {observed.shape}. Skipping...")
        return float('inf')
    return np.mean((simulated.y.T - observed) ** 2)


def objective_function(betas, ode_func, X0, t, observed_data, config):
    """Objective function to minimize: the error between simulated and observed data."""
    w_reg = 0.01
    simulated = simulate_system(ode_func, X0, t, tuple(betas), config.ivp_method, config.DEBUG)
    if simulated is None:
        if config.DEBUG: print("SOLVE_IVP FAILED")
        return float('inf')
    reg = w_reg * np.sum(betas ** 2)
    error = calculate_error(simulated, observed_data, config.DEBUG)
    if config.DEBUG: print(f"betas: {betas} | error: {error} | regularization: {reg}")
    return error + reg


def estimate_parameters(ode_func, X0, t, observed_data, initial_guess, config):
    """Estimate the parameters using scipy's minimize function."""
    try:
        result = minimize(
            objective_function,
            initial_guess,
            args=(ode_func, X0, t, observed_data, config),
            method=config.minimize_method,
            tol=1e-6,
            options={'maxiter': 1000},  # 'disp': True, 'gtol': 1e-6, 'eps': 1e-10}
            callback=OptimizeStopper(config.DEBUG, 15)
        )
    except TookTooLong as e:
        return OptimizeResult(fun=float('inf'), x=None, success=False, message=str(e))

    if config.DEBUG: print("estimated parameters: ", result)
    return result


class TookTooLong(Warning):
    pass


class OptimizeStopper(object):
    def __init__(self, DEBUG, max_sec=60):
        self.max_sec = max_sec
        self.start = time.time()
        self.DEBUG = DEBUG

    def __call__(self, xk=None):
        elapsed = time.time() - self.start
        if elapsed > self.max_sec:
            if not self.DEBUG: print(f"Terminating optimization: exceeded {self.max_sec} seconds")
            raise TookTooLong()
