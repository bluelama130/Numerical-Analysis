import numpy as np
from scipy import optimize

def forward_euler(f, tspan, y0, k):
    n = int(round((tspan[1]-tspan[0]))/k)
    t = np.linspace(tspan[0], tspan[1], n+1)
    y = t.copy()
    y[0] = y0
    for i in range(n):
        y[i+1] = y[i] + k*f(t[i], y[i])
    return t, y

def midpoint(f, tspan, y0, k):
    n = int(round((tspan[1]-tspan[0]))/k)
    t = np.linspace(tspan[0], tspan[1], n+1)
    y = t.copy()
    y[0] = y0
    yp = y0 + k/2*f(t[0], y[0])
    y[1] = y[0] + k*f(t[0]+k/2, yp)
    for i in range(1,n):
        y[i+1] = y[i-1] + 2*k*f(t[i], y[i])
    return t, y

def trapezoidal(f, tspan, y0, k, fprime=None):
    n = int(round((tspan[1]-tspan[0]))/k)
    t = np.linspace(tspan[0], tspan[1], n+1)
    y = t.copy()
    y[0] = y0
    eqprime = None
    for i in range(0,n):
        eq = lambda u: u-y[i] - k/2*(f(t[i], y[i]) + f(t[i+1],u))
        if fprime != None:
            eqprime = lambda u: 1 - k/2*fprime(t[i+1], u)
        sol = optimize.root_scalar(eq, x0=y[i],fprime=eqprime)
        y[i+1] = sol.root
    return t, y

def rk4(f, tspan, y0, k):
    n = int(round((tspan[1]-tspan[0]))/k)
    t = np.linspace(tspan[0], tspan[1], n+1)
    y = t.copy()
    y[0] = y0

    for i in range(n):
        t0 = t[i]
        y0 = y[i]
        h1 = f(t0,y0)
        h2 = f(t0+k/2, y0 + k/2*h1)
        h3 = f(t0+k/2, y0 + k/2*h2)
        h4 = f(t0+k, y0+k*h3)
        y[i+1] = y[i] + k/6*(h1 + 2*h2 + 2*h3 + h4)
    return t, y


    

