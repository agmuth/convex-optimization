import numpy as np


def grad(f: callable, h: float=1e-4) -> callable:
    h_inv = 1/h
    def grad_f(x: np.array) -> np.array:
        return h_inv * (f(x+h) - f(x-h))

    return grad_f