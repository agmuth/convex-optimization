import numpy as np
import numdifftools as nd


def newtons_method(func: callable, x_0: np.array, tol: float=1e-4, maxiters: int=100) -> dict:
    res = {"x": None, "func": None, "grad": None, "hess": None, "iters": None, "convergence": None}
    grad = nd.Gradient(func)
    hess = nd.Hessian(func)
    x_n = x_0

    for i in range(maxiters):
        inv_hess_at_x_n = np.linalg.inv(hess(x_n))
        grad_at_x_n = grad(x_n)
        delta_x = inv_hess_at_x_n @ grad_at_x_n
        if np.linalg.norm(delta_x, 2) < tol:
            res["convergence"] = True
            res["iters"] = i
            break
        else:
            x_n -= delta_x
    else:
        res["convergence"] = False
        res["iters"] = maxiters
    
    res["x"] = x_n
    res["func"] = func(x_n)
    res["grad"] = grad(x_n)
    res["hess"] = hess(x_n)
    
    return res

# if __name__ == "__main__":
#     def boyd_equ_9_20_func(x: np.array):
#         return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

#     def sphere_func(x: np.array):
#         return np.sum(x**2)

#     x_0 = np.array([50., 50.])
#     res_sphere = newtons_method(sphere_func, x_0)
#     res_boyd_equ_9_20 = newtons_method(boyd_equ_9_20_func, x_0)
#     print()