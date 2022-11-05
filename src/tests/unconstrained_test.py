from tests.test_functions import * 
from convexoptimization.unconstrained import * 
import numpy as np


def test_newtons_method_axis_parallel_hyper_ellipsoid_function_2_d():
    tol = 1e-4
    x_0 = np.array([5.12, 5.12])
    res = newtons_method(
        axis_parallel_hyper_ellipsoid_function_2_d.func, 
        x_0
    )
    assert res["convergence"] and np.linalg.norm(res["x"] - axis_parallel_hyper_ellipsoid_function_2_d.minimum, 2) < tol



