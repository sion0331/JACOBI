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
        axs.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{str(round(X0[0], 2))},{str(round(X0[1], 2))}]")
    axs.set_title("Target Phase-space by IC")
    axs.set_xlabel("Preys")
    axs.set_ylabel("Predators")
    axs.legend()


def plot_2d_by_y(axs, x0, ys, labels):
    for y, label in zip(ys, labels):
        axs.plot(y[:, 0], y[:, 1], "-", label=label)

    axs.plot(x0[0], x0[1], 'ro')
    axs.annotate(f'IC:[{round(x0[0],1)},{round(x0[1],1)}]', xy=(x0[0], x0[1]), xytext=(x0[0] + 0.5, x0[1] + 0.5),
                 arrowprops=dict(facecolor='black', shrink=1))

    x_min, x_max = ys[0][:, 0].min(), ys[0][:, 0].max()
    y_min, y_max = ys[0][:, 1].min(), 2#ys[0][:, 1].max()
    axs.set_xlim(x_min - x_max * 0.1, x_max * 1.1)
    axs.set_ylim(y_min - y_max * 0.1, y_max * 1.1)

    axs.set_title("Best Estimate")
    axs.set_xlabel("Preys")
    axs.set_ylabel("Predators")
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


def plot_time_series(t, y_raw, y_target, y_best):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    # plot for the first variable
    axs[2, 0].plot(t, y_raw[:, 0], label='Raw Data (Variable 1)')
    axs[2, 0].plot(t, y_target[:, 0], label='Target Data (Variable 1)')
    axs[2, 0].plot(t, y_best[:, 0], label='Best Fit (Variable 1)')
    axs[2, 0].set_xlabel('Time')
    axs[2, 0].set_ylabel('Value')
    axs[2, 0].legend()
    axs[2, 0].set_title('Time Series Plot (Variable 1)')
    # plot for the second variable
    axs[2, 1].plot(t, y_raw[:, 1], label='Raw Data (Variable 2)')
    axs[2, 1].plot(t, y_target[:, 1], label='Target Data (Variable 2)')
    axs[2, 1].plot(t, y_best[:, 1], label='Best Fit (Variable 2)')
    axs[2, 1].set_xlabel('Time')
    axs[2, 1].set_ylabel('Value')
    axs[2, 1].legend()
    axs[2, 1].set_title('Time Series Plot (Variable 2)')


def new_plot(t, y_raw, y_target, y_best):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    # plot for all data combined (Raw + Target + Best Fit for both variables)
    axs[1, 1].plot(t, y_raw[:, 0], label='Raw Data (Variable 1)', color='blue', linestyle='-')
    axs[1, 1].plot(t, y_target[:, 0], label='Target Data (Variable 1)', color='green', linestyle='--')
    axs[1, 1].plot(t, y_best[:, 0], label='Best Fit (Variable 1)', color='red', linestyle='-.')
    axs[1, 1].plot(t, y_raw[:, 1], label='Raw Data (Variable 2)', color='orange', linestyle='-')
    axs[1, 1].plot(t, y_target[:, 1], label='Target Data (Variable 2)', color='purple', linestyle='--')
    axs[1, 1].plot(t, y_best[:, 1], label='Best Fit (Variable 2)', color='brown', linestyle='-.')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].legend()
    axs[1, 1].set_title('Combined Data (Raw + Target + Best Fit for Var 1 & Var 2)')
    plt.tight_layout()
    plt.show()