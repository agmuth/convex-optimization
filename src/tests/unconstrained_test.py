from tests.test_functions import * 
from convexoptimization.unconstrained import * 
import numpy as np


def wrapper(solver: callable, test_func: TestFunction, tol: float=1e-4) -> None:
    res = solver(
        test_func.func, 
        test_func.x_0
    )
    assert all(
        [
            res["convergence"],
             np.linalg.norm(res["x"] - test_func.x_minimum, 2) < tol,
             abs(res["func"] - test_func.func_minimum) < tol

        ]
    )


def test_newtons_method_sphere_function_2d():
    wrapper(newtons_method, sphere_function_2d)

def test_newtons_method_axis_parallel_hyper_ellipsoid_function_3d():
    wrapper(newtons_method, axis_parallel_hyper_ellipsoid_function_3d)

def test_newtons_method_rotated_hyper_ellipsoid_function_4d():
    wrapper(newtons_method, rotated_hyper_ellipsoid_function_4d)
