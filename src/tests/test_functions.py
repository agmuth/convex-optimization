import numpy as np


class TestFunction(object):
    def __init__(self, func: callable, dim: int, minimum: np.array):
        self.func, self.dim, self.minimum = func, dim, minimum


axis_parallel_hyper_ellipsoid_function_2_d = TestFunction(
    func=lambda x: np.dot(np.arange(1, x.shape[0]+1), x**2),
    dim=2,
    minimum=np.array([0., 0.])
)