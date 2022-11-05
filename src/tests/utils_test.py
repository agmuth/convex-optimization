from utils import grad
import numpy as np


def sphere_func(x: np.array) -> float:
    return x.dot(x)

def test_grad_1():
    tol = 1e-3
    grad_sphere_func = grad(sphere_func)
    x = np.array([0, 0])
    assert np.sum((grad_sphere_func(x) - np.array([0, 0])**2)) < tol