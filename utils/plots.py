import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def plot_2D(ode_func, betas):
    t = np.linspace(0, 10, 100)
    IC = np.linspace(0.9, 1.8, 10)

    plt.figure()
    for x in IC:
        X0 = [x, x]
        Xs = odeint(ode_func, X0, t, args=tuple(betas))
        plt.plot(Xs[:, 0], Xs[:, 1], "-", label=f"IC:[{str(round(X0[0], 2))},{str(round(X0[1], 2))}")
    plt.xlabel("X_0")
    plt.ylabel("X_1")
    plt.legend()
    plt.show()
