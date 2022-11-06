import numpy as np


class TestFunction(object):
    def __init__(self, func: callable, x_0: np.array, x_minimum: np.array, func_minimum: float):
        self.func, self.x_0, self.x_minimum, self.func_minimum = func, x_0, x_minimum, func_minimum


sphere_function_2d = TestFunction(
    func=lambda x: (x**2).sum(),
    x_0=np.array([-5.12, 5.12]),
    x_minimum=np.array([0., 0.]),
    func_minimum=0.
)


axis_parallel_hyper_ellipsoid_function_3d = TestFunction(
    func=lambda x: np.dot(np.arange(1, x.shape[0]+1), x**2),
    x_0=np.array([5.12, 5.12, 5.12]),
    x_minimum=np.array([0., 0., 0.]),
    func_minimum=0.
)


rotated_hyper_ellipsoid_function_4d = TestFunction(
    func=lambda x: np.cumsum(x**2).sum(),
    x_0=np.array([-65.536, 65.536, -65.536, 65.536]),
    x_minimum=np.array([0., 0., 0., 0]),
    func_minimum=0.
)

