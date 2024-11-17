import numpy as np


def lotka_func(t, X, alpha, beta, delta, gamma):
    x, y = X
    dotx = (alpha * x) + (beta * x * y)
    doty = (delta * y) + (gamma * x * y)
    return np.array([dotx, doty])


class lotka():
    def __init__(self):
        self.func = lotka_func
        self.N = 2
        self.betas = [2 / 3, -4 / 3, -1, 1]
