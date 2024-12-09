import pickle
import os
import re
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp

from evaluators.parameter_estimation import calculate_error
from population.initial_generation import beautify_system, generate_population
from utils.functions import get_functions, funcs_to_str
from utils.load_systems import create_ode_function, load_systems
from utils.models import lotka
import matplotlib.pyplot as plt

from utils.numpy_conversion import save_systems_as_numpy_funcs, save_str_systems_as_numpy_funcs
from utils.plots import plot_2d_by_func, plot_2d_by_y, plot_invalid_by_iteration, plot_loss_by_iteration, \
    plot_min_loss_by_iteration, plot_avg_loss_by_iteration, plot_invalid_counts_by_iteration, plot_2d_estimates_by_y


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

            loss.append(h['Score'])
            if G == params['G'] -1:
                last.append(h)

        result['min_loss'] = min_loss
        result['avg_loss'] = avg_loss
        result['invalid'] = invalid
        result['last'] = last
        results.append(result)

    for result in results:
        valid_entries = [(h['Score'], h) for h in result['last']]
        result['last'] = [h for score, h in sorted(valid_entries, key=lambda x: x[0])]
        print(result['last'][0])
        # result['best']=[]
        # for r in result['last'][:5]:
        #     result['best'].append(convert_to_ode_func(r['System']))
        # # print(save_str_systems_as_numpy_funcs(result['last'][:5], 'best_equations.txt'))
        # print(convert_to_ode_func(result['last'][0]['System']))

        # best = None
        # for score, h in sorted(valid_entries, key=lambda x: x[0]):
        #     #print(score, h['Score'])
        #     if best is None or score < best['Score']:
        #         best = h
        # print(f'\nBest | Loss:{best['Score']} func: {best['System']}')

    ### lotka
    if results[0]['func'] == 'lotka':
        target = lotka()
        t = np.linspace(0, 100, 500)
        X0 = np.random.rand(target.N) + 1.0  # 1.0~2.0
        y_raw = solve_ivp(target.func, (t[0], t[-1]), X0, args=target.betas, t_eval=t, method='Radau').y.T
        y_target = y_raw + np.random.normal(0.0, 0.01, y_raw.shape)

        # y_best = solve_ivp(convert_to_ode_func(best['System']), (t[0], t[-1]), X0, args=tuple(best['param']), t_eval=t, method='Radau').y.T
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_2d_by_func(axs[0, 0], target.func, target.betas)
        plot_2d_by_y(axs[0, 1], X0, [y_raw, y_target], ["Original Data", "Noisy Data", "BEST"])
        #plot_2d_by_y(axs[0, 1], [y_raw, y_target, y_best], ["TARGET_RAW", "TARGET_NOISED", "BEST"])
        plot_min_loss_by_iteration(axs[1, 0], results)
        plot_avg_loss_by_iteration(axs[1, 1], results)
        # plot_invalid_by_iteration(axs[1, 1], results[0]['invalid'])
    # #
    # # note = f""" Target:{type(target).__name__} | G:{history[0][G} N:{config.N} M:{config.M} I:{config.I} J:{config.J} f0ps:{funcs_to_str(config.f0ps)} Composite:{config.allow_composite} | elite:{config.elite_rate} new:{config.new_rate} cross:{config.crossover_rate} mutate:{config.mutation_rate}| ivp:{config.ivp_method} min:{config.minimize_method}
    # #     Best Function: {beautify_system(best[1])}
    # #     Best Loss: {best[0]['fun']} Best Parameters: {best[0]['x']}"""
    # # fig.text(0.03, 0.08, note, va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

        #####
        best_systems = load_systems('../data/lotka_best.txt')
        y_best = []
        labels = [funcs_to_str(get_functions(f)) for f in ["5", "4,5", "1,4,5"]]
        for system in best_systems:
            system = create_ode_function(system)
            y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
            y_best.append(y)
            print("best error: ", print(np.mean((y - y_raw) ** 2)))

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_invalid_counts_by_iteration(axs[0, 0], results)
        plot_2d_estimates_by_y(axs[0, 1], y_target, y_best, labels)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()



