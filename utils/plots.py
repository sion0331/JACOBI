import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def plot_2d_by_func(axs, ode_func, betas):
    t = np.linspace(0, 10, 100)
    IC = np.linspace(0.9, 1.8, 10)

    for x in IC:
        X0 = [x, x]
        Xs = solve_ivp(ode_func, (t[0], t[-1]), X0, args=betas, t_eval=t, method='BDF').y.T
        axs.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{str(round(X0[0], 2))},{str(round(X0[1], 2))}")
    axs.set_title("Target Phase-space by IC")
    axs.set_xlabel("X_0")
    axs.set_ylabel("X_1")
    axs.legend()


def plot_2d_by_y(axs, ys, labels):
    for y, label in zip(ys, labels):
        axs.plot(y[:, 0], y[:, 1], "-", label=label)

    x_min, x_max = ys[0][:, 0].min(), ys[0][:, 0].max()
    y_min, y_max = ys[0][:, 1].min(), ys[0][:, 1].max()
    axs.set_xlim(x_min - x_max * 0.1, x_max * 1.1)
    axs.set_ylim(y_min - y_max * 0.1, y_max * 1.1)

    axs.set_title("Best Estimate")
    axs.set_xlabel("X_0")
    axs.set_ylabel("X_1")
    axs.legend()


def plot_loss_by_iteration(axs, min_loss, avg_loss):
    axs.plot(np.arange(len(min_loss)), min_loss, marker='o', label='Minimum Loss', linestyle='-', alpha=0.8)
    axs.set_xlabel("Generation")
    axs.set_ylabel("Min Loss", color='blue')
    axs.tick_params(axis='y', labelcolor='blue')

    ax2 = axs.twinx()
    ax2.plot(np.arange(len(avg_loss)), avg_loss, marker='s', label='Average Loss', linestyle='--', alpha=0.8,
             color='red')
    ax2.set_ylabel("Average Loss", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    axs.set_title("Loss Over Generations")
    lines, labels = axs.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend(lines + lines2, labels + labels2, loc='upper right')


def plot_invalid_by_iteration(axs, invalid):
    axs.plot(range(len(invalid)), invalid, marker='o', linestyle='-', color='r', alpha=0.8, label="Invalid Systems")
    axs.set_title("Invalid Systems Over Generations")
    axs.set_xlabel("Generation")
    axs.set_ylabel("Number of Invalid Entries")
    axs.xaxis.set_major_locator(MaxNLocator(integer=True))
    axs.legend()
