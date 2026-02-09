"""
An example of using optimizers using the function "example_func"
"""

import optimizers as opt
import numpy as np

def example_func(x):
    """
    :param x: vector of weights
    :return: the function for which you need to find the minimum
    """
    return x[0]**2 + 4*x[1]**2

def get_gradient(x):
    """
    :param x: vector of weights
    :return: vector of derivatives (gradient)
    """
    return np.array([2*x[0], 8*x[1]], np.float32)

max_iter = 1024 #max iterations
tol = 1e-5 #sufficient accuracy of convergence

x = np.array([1.0, 2.0], np.float32) #initial weights
x_old = x + 10*tol #x_old is for stopping the algorithm
opt = opt.SGD([x], lr = 0.01) #initialized optimizer

#algorithm
while (np.linalg.norm(x - x_old) > tol and opt.step_count < max_iter):
    x_old = x.copy()
    opt.params[0].grad = get_gradient(x)
    opt.step() #updating weights
    opt.zero_grad()

true_x = np.array([0.0, 0.0], np.float32) #true weights which realize the minimum

print(opt.step_count) #count of iterations before stopping the algorithm
print(x)
print(true_x)
print(np.linalg.norm(x-true_x))



