import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def plot_2d_by_func(ode_func, betas):
    t = np.linspace(0, 10, 100)
    IC = np.linspace(0.9, 1.8, 10)

    plt.figure()
    for x in IC:
        X0 = [x, x]
        Xs = solve_ivp(ode_func, (t[0], t[-1]), X0, args=betas, t_eval=t, method='BDF').y.T
        plt.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{str(round(X0[0], 2))},{str(round(X0[1], 2))}")
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.legend()
    plt.show()


def plot_2d_by_y(y1, label1, y2, label2):
    plt.figure()
    plt.plot(y1[:, 0], y1[:, 1], "-", label=label1)
    plt.plot(y2[:, 0], y2[:, 1], "-", label=label2)
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.legend()
    plt.show()
