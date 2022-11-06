import numpy as np
import numdifftools as nd


def back_tracking_line_search(func: callable, grad: callable, x: np.array, delta_x: np.array, alpha: float=0.3, beta: float=0.8) -> float:
    t = 1
    while func(x + t * delta_x[:, 0]) > func(x) + alpha * t * delta_x @ grad(x).T:
        t *= beta
    return t


def gradient_descent(func: callable, x_0: np.array, tol: float=1e-4, maxiters: int=100) -> dict:
    res = {"x": None, "func": None, "grad": None, "hess": None, "iters": None, "convergence": None}
    grad = nd.Jacobian(func)
    x_n = x_0

    for i in range(maxiters):
        delta_x = -grad(x_n)
        t = back_tracking_line_search(func, grad, x_n, delta_x)
        if np.linalg.norm(t*delta_x, 2) < tol:
            res["convergence"] = True
            res["iters"] = i
            break
        else:
            x_n += t*delta_x[:, 0]
    else:
        res["convergence"] = False
        res["iters"] = maxiters
    
    res["x"] = x_n.flatten()
    res["func"] = func(x_n)
    res["grad"] = grad(x_n)
    
    return res


def newtons_method(func: callable, x_0: np.array, tol: float=1e-4, maxiters: int=100) -> dict:
    res = {"x": None, "func": None, "grad": None, "hess": None, "iters": None, "convergence": None}
    grad = nd.Jacobian(func)
    hess = nd.Hessian(func)
    x_n = x_0

    for i in range(maxiters):
        inv_hess_at_x_n = np.linalg.inv(hess(x_n))
        grad_at_x_n = grad(x_n).T
        delta_x = -inv_hess_at_x_n @ grad_at_x_n
        if np.linalg.norm(delta_x, 2) < tol:
            res["convergence"] = True
            res["iters"] = i
            break
        else:
            x_n += delta_x[:, 0]
    else:
        res["convergence"] = False
        res["iters"] = maxiters
    
    res["x"] = x_n.flatten()
    res["func"] = func(x_n)
    res["grad"] = grad(x_n)
    res["hess"] = hess(x_n)
    
    return res


# if __name__ == "__main__":
#     def func(x: np.array):
#         return -1 * np.log(x)/(1+x)

#     x_star = np.array([3.59112])
#     x_0 = np.array([1.])

#     res1 = gradient_descent(func, x_0)
#     res2 = gradient_descent(lambda x: (x**2).sum(), np.array([5., 5.]))
#     res3 = gradient_descent(lambda x: np.dot(np.arange(1, x.shape[0]+1), x**2), np.array([2., 2., 2.]))
#     print()