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


def sir_func(t, X, b1, b2, b3, b4):
    S, I, R = X
    dotS = b1 * S * I
    dotI = b2 * S * I + b3 * I
    dotR = b4 * I
    return np.array([dotS, dotI, dotR])


class SIR():
    def __init__(self):
        self.func = sir_func
        self.N = 3
        self.betas = [-0.0002, 0.0002, -0.04, 0.04]
