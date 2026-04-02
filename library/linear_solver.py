import numpy as np
def steepest_descent(A, b, x0, tol=1e-5, max_it=1000):
    x = x0.copy()
    r = b - A@x0
    for _ in range(max_it):
        w = A@r
        alpha = (r@r)/(r@w)
        x = x + alpha*r
        r = r - alpha*w
        if (np.linalg.norm(r) < tol):
            break
    return x
    
def conjuage_gradient(A, b, x0, tol=1e-10, max_it=1000):
    x = x0
    r = b - A@x0 
    p = r
    for _ in range(max_it):
        w = A@p
        alpha = (r@r)/(p@w)
        x = x + alpha*p
        r_next = r - alpha*w
        if (np.linalg.norm(r_next) < tol):
            return x
        beta = (r_next@r_next)/(r@r)
        p = r_next + beta*p
        r = r_next
    return x
        



