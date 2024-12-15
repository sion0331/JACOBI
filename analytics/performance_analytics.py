import pickle
import os
import numpy as np
from scipy.integrate import solve_ivp

from population.initial_generation import beautify_system
from utils.functions import get_functions, funcs_to_str
from utils.history import load_history
from utils.load_systems import create_ode_function, load_systems
from utils.models import lotka, SIR
import matplotlib.pyplot as plt
from utils.plots import plot_2d_by_func, plot_2d_by_y, \
    plot_min_loss_by_iteration, plot_avg_loss_by_iteration, plot_invalid_counts_by_iteration, plot_2d_estimates_by_y, \
    plot_time_per_generation, plot_3d_by_y, plot_3d_estimates

if __name__ == "__main__":
    target = "SIR"  # "SIR" #lotka
    histories = load_history(target)

    results = []
    for history in histories:
        if history[0].split('_')[0] != target: continue

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
                avg_loss.append(np.mean([l for l in loss if l < 100000]))  # Exclude loss > 100
                invalid.append(sum(l >= 100000 for l in loss))
                loss = []

            loss.append(h['Score'])
            if G == params['G'] - 1:
                last.append(h)

        result['min_loss'] = min_loss
        result['avg_loss'] = avg_loss
        result['invalid'] = invalid
        result['last'] = last
        results.append(result)

    for result in results:
        valid_entries = [(h['Score'], h) for h in result['last']]
        result['last'] = [h for score, h in sorted(valid_entries, key=lambda x: x[0])]
        print("Best:", target, funcs_to_str(get_functions(result['param']['f0ps'])), result['last'][0])

    if target == "SIR":
        target = SIR()
        t = np.linspace(0, 100, 300)
        X0 = [997, 3, 0]
        y_raw = solve_ivp(target.func, (t[0], t[-1]), X0, args=target.betas, t_eval=t, method='Radau').y.T
        y_target = y_raw + np.random.normal(0.0, 10, y_raw.shape)

        # best_systems = load_systems('../data/results/SIR/SIR_best.txt')
        # y_best = []
        # labels = [funcs_to_str(get_functions(f)) for f in ["5", "4,5"]]
        # for system in best_systems:
        #     system = create_ode_function(system)
        #     y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
        #     y_best.append(y)
        #     print("best error: ", labels, print(np.mean((y - y_raw) ** 2)))

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_3d_by_y(axs[0, 0], t, y_target, [y_raw], ['Noisy Data'])
        plot_min_loss_by_iteration(axs[1, 0], results)
        plot_avg_loss_by_iteration(axs[1, 1], results)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_invalid_counts_by_iteration(axs[0, 0], results)
        # plot_3d_by_y(axs[0, 1], t, y_target, y_best, labels)
        plot_time_per_generation(axs[1, 0], histories)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

        ###### PLOT #2 ######
        # best_systems = load_systems('../data/results/SIR/SIR_best.txt')
        # y_best = []
        # labels = [funcs_to_str(get_functions(f)) for f in ["5", "4,5"]]
        # for system in best_systems:
        #     system = create_ode_function(system)
        #     y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
        #     y_best.append(y)
        #     print("best error: ", labels, print(np.mean((y - y_raw) ** 2)))

        best_systems1 = load_systems('../data/results/SIR/sir_best_10.txt')
        y_best1 = []
        labels = [funcs_to_str(get_functions(f)) for f in
                  ["5", "4,5", "1,4,5", "1,2,4,5", "1,2,4,5,9", "1,2,4,5,6,9", "1,2,3,4,5,6,9"]]
        for system in best_systems1:
            system = create_ode_function(system)
            y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
            y_best1.append(y)
            print("best error: ", labels, print(np.mean((y - y_target) ** 2)))

        best_systems2 = load_systems('../data/results/SIR/sir_best_20.txt')
        y_best2 = []
        labels = [funcs_to_str(get_functions(f)) for f in
                  ["5", "4,5", "1,4,5", "1,2,4,5", "1,2,4,5,9", "1,2,4,5,6,9", "1,2,3,4,5,6,9"]]
        for system in best_systems2:
            system = create_ode_function(system)
            y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
            y_best2.append(y)
            print("best error: ", print(np.mean((y - y_target) ** 2)))

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        # plot_3d_by_y(axs[0, 0], t, y_target, [y_raw], ['Noisy Data'])
        plot_3d_estimates(axs[0, 0], t, X0, y_target, y_best1, labels, "Generation #10")
        plot_3d_estimates(axs[0, 1], t, X0, y_target, y_best2, labels, "Generation #20")

        # plot_2d_estimates_by_y(axs[0, 0], X0, y_target, y_best1, labels, "Generation #5")
        # plot_3d_estimates(axs[0, 1], t, X0, y_target, y_best2, labels, "Generation #20")
        # plot_invalid_counts_by_iteration(axs[1, 0], results)
        # plot_time_per_generation(axs[1, 1], histories)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

    ### lotka
    if target == 'lotka':
        target = lotka()
        t = np.linspace(0, 10, 100)
        X0 = [1.8, 1.3]
        y_raw = solve_ivp(target.func, (t[0], t[-1]), X0, args=target.betas, t_eval=t, method='Radau').y.T
        y_target = y_raw + np.random.normal(0.0, 0.01, y_raw.shape)

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_2d_by_func(axs[0, 0], target.func, target.betas)
        plot_2d_by_y(axs[0, 1], X0, [y_raw, y_target], ["Original Data", "Noisy Data", "BEST"])
        plot_min_loss_by_iteration(axs[1, 0], results)
        plot_avg_loss_by_iteration(axs[1, 1], results)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

        ###### PLOT #2 ######
        best_systems1 = load_systems('../data/results/lotka/lotka_best_5.txt')
        y_best1 = []
        labels = [funcs_to_str(get_functions(f)) for f in
                  ["5", "4,5", "1,4,5", "1,2,4,5", "1,2,4,5,9", "1,2,4,5,6,9", "1,2,3,4,5,6,9"]]
        for system in best_systems1:
            system = create_ode_function(system)
            y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
            y_best1.append(y)
            print("best error: ", labels, print(np.mean((y - y_raw) ** 2)))

        best_systems2 = load_systems('../data/results/lotka/lotka_best_10.txt')
        y_best2 = []
        labels = [funcs_to_str(get_functions(f)) for f in
                  ["5", "4,5", "1,4,5", "1,2,4,5", "1,2,4,5,9", "1,2,4,5,6,9", "1,2,3,4,5,6,9"]]
        for system in best_systems2:
            system = create_ode_function(system)
            y = solve_ivp(system, (t[0], t[-1]), X0, args=tuple([]), t_eval=t, method="Radau").y.T
            y_best2.append(y)
            print("best error: ", labels, print(np.mean((y - y_raw) ** 2)))

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        plot_2d_estimates_by_y(axs[0, 0], X0, y_target, y_best1, labels, "Generation #5")
        plot_2d_estimates_by_y(axs[0, 1], X0, y_target, y_best2, labels, "Generation #10")
        plot_invalid_counts_by_iteration(axs[1, 0], results)
        plot_time_per_generation(axs[1, 1], histories)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()
