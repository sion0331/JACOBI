import pickle
import os
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from population.initial_generation import beautify_system
from utils.functions import get_functions
from utils.load_systems import create_ode_function
from utils.models import lotka
import matplotlib.pyplot as plt

from utils.plots import plot_2d_by_func, plot_2d_by_y, plot_invalid_by_iteration, plot_loss_by_iteration


def save_history(config, history):
    results = [{'G': config.G, 'N': config.N, 'M': config.M, 'I': config.I, 'J': config.J,
                'allow_composite': config.allow_composite,
                'f0ps': str(config.f0ps), 'ivp_method': config.ivp_method, 'minimize_method': config.minimize_method,
                'elite_rate': config.elite_rate, 'crossover_rate': config.crossover_rate,
                'mutation_rate': config.mutation_rate, 'new_rate': config.new_rate}]

    for i, generation in enumerate(history):
        for j, individual in enumerate(generation):
            result = {'Generation': i, 'Population': j, 'Score': individual[0].fun,
                      'System': beautify_system(individual[1]), 'param': individual[0].x}
            results.append(result)

    filename = (f"./data/results/{str(config.target.__class__.__name__)}_{str(config.f0ps)}"
                f"_G{config.G}_N{config.N}_M{config.M}_I{config.I}_J{config.I}_{config.ivp_method}"
                f"_{int(config.elite_rate*100)}_{int(config.crossover_rate*100)}"
                f"_{int(config.mutation_rate*100)}_{int(config.new_rate*100)}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"History saved to {filename}.")


def load_history():
    directory = "../data/results"
    histories = []
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'rb') as file:
                    history = pickle.load(file)
                    histories.append((filename, history))
                    print(f"Loaded {filename} successfully.")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return histories


def convert_to_ode_func(system_strings):
    """Convert a string representation of ODEs to a callable function."""
    # Define symbolic variables
    t = sp.Symbol('t')
    variables = [sp.Function(f'x_{i+1}')(t) for i in range(len(system_strings))]
    betas = sp.symbols(f'beta_0:{len(system_strings) * 2}')  # Adjust the range based on your beta count

    # Parse the equations into symbolic expressions
    equations = []
    for eq in system_strings:
        eq_rhs = eq.split('=')[1].strip()  # Extract the right-hand side of the equation
        equations.append(sp.sympify(eq_rhs))

    def ode_func(t, X, *params):
        # Substitute variables and parameters into the symbolic equations
        substitutions = {var: val for var, val in zip(variables, X)}
        substitutions.update({beta: param for beta, param in zip(betas, params)})
        rhs = [float(eq.subs(substitutions)) for eq in equations]
        return np.array(rhs)

    return ode_func


if __name__ == "__main__":
    results = []
    histories = load_history()
    for history in histories:
        params = history[1][0]
        result = {'func': history[0].split('_')[0], 'param': params}

        loss = []
        min_loss = []
        avg_loss = []
        invalid = []
        last = []

        G = 0
        for i, h in enumerate(history[1][1:]):
            if G < h['Generation']:
                G = h['Generation']
                min_loss.append(min(loss))
                avg_loss.append(np.mean([l for l in loss if l < 1000]))  # Exclude loss > 100
                invalid.append(sum(l >= 1000 for l in loss))
                loss = []

            # print(f'generation {h['Generation']} {h['Population']} | Score:{round(h['Score'], 4)} func: {h['System']} param:{h['param']}')
            loss.append(h['Score'])

            if G == params['G'] -1:
                last.append(h)

        result['min_loss'] = min_loss
        result['avg_loss'] = avg_loss
        result['invalid'] = invalid
        result['last'] = last
        results.append(result)

    print(results)



    for result in results:
        valid_entries = [(h['Score'], h) for h in result['last']]
        best = None
        for score, h in sorted(valid_entries, key=lambda x: x[0]):
            #print(score, h['Score'])
            if best is None or score < best['Score']:
                best = h
        print(f'\nBest | Loss:{best['Score']} func: {best['System']}')

    ### lotka
    if results[0]['func'] == 'lotka':
        target = lotka()
        t = np.linspace(0, 10, 1000)
        X0 = np.random.rand(target.N) + 1.0  # 1.0~2.0
        y_raw = solve_ivp(target.func, (t[0], t[-1]), X0, args=target.betas, t_eval=t, method='Radau').y.T
        y_target = y_raw + np.random.normal(0.0, 0.02, y_raw.shape)

        print("initial   ", X0)
        # y_best = solve_ivp(convert_to_ode_func(best['System']), (t[0], t[-1]), X0, args=tuple(best['param']), t_eval=t, method='Radau').y.T
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        plot_2d_by_func(axs[0, 0], target.func, target.betas)
        plot_2d_by_y(axs[0, 1], X0, [y_raw, y_target], ["Original Data", "Noisy Data", "BEST"])
        #plot_2d_by_y(axs[0, 1], [y_raw, y_target, y_best], ["TARGET_RAW", "TARGET_NOISED", "BEST"])
        plot_loss_by_iteration(axs[1, 0], results[0]['min_loss'], results[0]['avg_loss'])
        plot_invalid_by_iteration(axs[1, 1], results[0]['invalid'])
    # #
    # # note = f""" Target:{type(target).__name__} | G:{history[0][G} N:{config.N} M:{config.M} I:{config.I} J:{config.J} f0ps:{funcs_to_str(config.f0ps)} Composite:{config.allow_composite} | elite:{config.elite_rate} new:{config.new_rate} cross:{config.crossover_rate} mutate:{config.mutation_rate}| ivp:{config.ivp_method} min:{config.minimize_method}
    # #     Best Function: {beautify_system(best[1])}
    # #     Best Loss: {best[0]['fun']} Best Parameters: {best[0]['x']}"""
    # # fig.text(0.03, 0.08, note, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()



