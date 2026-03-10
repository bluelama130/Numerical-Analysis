from scipy.special import factorial
import numpy as np
#finite differentiation
def diff_forward(f, x, h):
    return (f(x+h) - f(x))/h
def diff_backward(f, x, h):
    return (f(x)-f(x-h))/h
def diff_center(f, x, h):
    return (f(x+h) - f(x-h)) / (2*h)

#determing coefficients for finite differentiation
def fdcoeff(k, xbar, x):
    n = len(x)
    A = np.ones((n, n))
    b = np.zeros(n)
    b[k] = 1
    for j in range(1,n):
        A[j,:] = (x - xbar)**j / factorial(j)
    c = np.linalg.solve(A,b)
    return c
