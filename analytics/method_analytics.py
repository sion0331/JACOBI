import warnings
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

from evaluators.parameter_estimation import estimate_parameters
from population.genetic_algorithm import generate_new_population
from population.initial_generation import generate_population, beautify_system
from utils.functions import get_functions
from utils.load_systems import load_systems, create_ode_function
from utils.mapping import get_term_map, get_solved_map
from utils.models import SIR


class Config_Test:
    def __init__(self):
        self.target = SIR()

        self.G = 1  # Number of generations
        self.N = 100  # Maximum number of population
        self.M = 3  # Maximum number of equations
        self.I = 2  # Maximum number of terms per equation
        self.J = 2  # Maximum number of functions per feature
        self.allow_composite = False  # Composite Functions
        self.f0ps = get_functions("4,5")
        self.ivp_method = 'Radau'
        self.minimize_method = 'L-BFGS-B' #'Nelder-Mead' / L-BFGS-B / CG, COBYLA, COBYQA, TNC - fast

        self.elite_rate = 0.1
        self.crossover_rate = 0.2
        self.mutation_rate = 0.5
        self.new_rate = 0.2

        self.system_load_dir = '../data/analysis/computation_equations.txt'  #data/sir_equations.txt'
        self.system_save_dir = '../data/analysis/computation_equations.txt'

        self.DEBUG = False


if __name__ == "__main__":
    config = Config_Test()
    population = generate_population(config)
    systems = load_systems(config.system_load_dir)

    t = np.linspace(0, 100, 300)
    X0 = [997, 3, 0]
    y_raw = solve_ivp(config.target.func, (t[0], t[-1]), X0, args=config.target.betas, t_eval=t,
                      method=config.ivp_method).y.T
    y_target = y_raw + np.random.normal(0.0, 0.0, y_raw.shape)

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for size in [10**0,10**0.5,10**1,10**1.5,10**2]:

            for method in ['Radau', 'BDF', 'LSODA']: # 'DOP853', "RK23"
            # for method in ["BFGS", "L-BFGS-B",'COBYLA', 'COBYQA', 'COBYQA','TNC','SLSQP','Nelder-Mead', 'Powell', 'CG']:
                start_time = time.time()
                best_score = float('inf')
                error_count = 0

                history = []
                time_records = []
                for i in range(config.G):
                    print("#### SYSTEM EVALUATION ####")
                    history.append([])
                    for j, system in enumerate(systems):
                        if j>=size: break
                        ode_func = create_ode_function(system)
                        initial_guess = np.zeros(config.I * config.M)
                        solved = estimate_parameters(ode_func, X0, t, y_target, initial_guess , config.minimize_method, method, config.DEBUG)
                        if solved.fun < best_score:
                            best_score = solved.fun
                        if math.isinf(solved.fun):
                            error_count += 1


                    print(f"Completed Generation {i}: {time.time()-start_time} | term_map:{len(get_term_map())}, solved_map:{len(get_solved_map())}")

                    if i < config.G - 1:
                        population = generate_new_population(history[i], population, config)
                        systems = load_systems(config.system_load_dir)
                results.append({'method': method, 'size': size, 'error_count': error_count, 'best': best_score, 'ts': time.time() - start_time})
    print(results)


    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    df = pd.DataFrame.from_dict(results)
    # Time plot
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axs[0].plot(subset['size'], subset['ts'], marker='o', label=method)
    axs[0].set_title('Execution Time by Method and Population Size')
    axs[0].set_xlabel('Population Size')
    axs[0].set_ylabel('Time (s)')
    axs[0].set_xscale('log')
    axs[0].legend()

    # Error count plot
    for method in df['method'].unique():
        subset = df[df['method'] == method]
        axs[1].plot(subset['size'], subset['error_count'], marker='o', label=method)
    axs[1].set_title('Error Count by Method and Population Size')
    axs[1].set_xlabel('Population Size')
    axs[1].set_ylabel('Error Count')
    axs[1].set_xscale('log')
    axs[1].legend()

    plt.tight_layout()
    plt.show()








