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


def sir_func(t, X, beta, gamma):
    S, I, R = X
    dotS = (-1) * beta * S * I
    dotI = beta * S * I - gamma * I
    dotR = gamma * I
    return np.array([dotS, dotI, dotR])


class SIR():
    def __init__(self, beta, gamma):
        self.func = sir_func
        self.N = 3
        self.beta = beta
        self.gamma = gamma
        self.betas = [self.beta, self.gamma]

    def get_params(self):
        return self.beta, self.gamma
