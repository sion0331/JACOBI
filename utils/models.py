import numpy as np

lotka_defaults = 2, [2 / 3, -4 / 3, -1, 1]


def lotka():
    def lotka_func(X, t, alpha, beta, delta, gamma):
        x, y = X
        dotx = (alpha * x) + (beta * x * y)
        doty = (delta * y) + (gamma * x * y)
        return np.array([dotx, doty])

    return lotka_func
