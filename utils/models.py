import numpy as np


def lotka():
    n = 2
    betas = [2 / 3, -4 / 3, -1, 1]

    def lotka_func(t, X, alpha, beta, delta, gamma):
        x, y = X
        dotx = (alpha * x) + (beta * x * y)
        doty = (delta * y) + (gamma * x * y)
        return np.array([dotx, doty])

    return lotka_func, n, betas
