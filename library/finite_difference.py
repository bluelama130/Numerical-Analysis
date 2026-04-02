import numpy as np
from scipy.linalg import solve_banded, norm
class solver:
    def __init__(self, start=0, end=1, n=10,
                 boundary_start=('dirichlet',0),
                 boundary_end = ('dirichlet',0)):
        self.set_boundary(boundary_start, boundary_end)
        self.set_interval(start, end, n)
        

    def set_n(self, n):
        self.n = n
        self.x, self.h = np.linspace(self.start, self.end, n+1, endpoint=True, retstep=True)
        self.y = np.zeros(n+1, dtype=float)
    
    def set_h(self, h):
        self.n = int(round((self.end - self.start)/h))
        self.set_n(self.n)
        
    def set_interval(self, start, end, n):
        self.start = start
        self.end = end
        self.set_n(n)
        
    def set_boundary(self, boundary_start, boundary_end):
        self.boundary_start = boundary_start
        self.boundary_end= boundary_end

    def solve(self):
        pass

class fdsolver(solver):
    def __init__(self, start=0, end=1, n=10,
                 boundary_start=('dirichlet', 0),
                 boundary_end=('dirichlet', 0), 
                 f=None):
        super().__init__(start, end, n, boundary_start, boundary_end)
        if f != None:
            self.f = f
    
    def _build_matrix(self):
        pass

    def set_f(self, f):
        self.f = f

    

class upp_sovler_d(fdsolver):
    def __init__(self, start=0, end=1, n=10, boundary_start=('dirichlet', 0), boundary_end=('dirichlet', 0), f=None):
        super().__init__(start, end, n, boundary_start, boundary_end, f)
        
    def _build_matrix(self):
        n, h = self.n, self.h
        self.u = (1./h**2) * np.ones(n-1)
        self.d = -(2./h**2) * np.ones(n-1)
        self.l = (1./h**2) * np.ones(n-1)
        self.u[0] = 0
        self.l[-1] = 0
        self.ab = np.array([self.u, self.d, self.l])


    def solve(self):
        self._build_matrix()
        self.y[0] = self.boundary_start[1]
        self.y[-1] = self.boundary_end[1]

        vec_f = np.vectorize(self.f)
        b = vec_f(self.x[1:-1])
        b[0] -= self.boundary_start[1]/self.h**2
        b[-1] -= self.boundary_end[1]/self.h**2
        
        self.y[1:-1] = solve_banded((1, 1), self.ab, b)

        ret_x = self.x.copy() 
        ret_y = self.y.copy()
        
        return ret_x, ret_y
    
class upp_solver_n1(fdsolver):
    def __init__(
                 self, start=0, end=1, n=10,
                 boundary_start=('neumann', 0), 
                 boundary_end = ('dirichlet', 0),
                 f=None):
        super().__init__(start, end, n, boundary_start, boundary_end, f)
        
    def _build_matrix(self):
        n, h = self.n, self.h
        self.u = (1./h**2) * np.ones(n+1)
        self.d = -(2./h**2) * np.ones(n+1)
        self.l = (1./h**2) * np.ones(n+1)
        self.u[0] = 0
        self.l[-1] = 0
        if self.boundary_start[0] == 'neumann':
            self.u[1] = 1/h
            self.d[0] = - 1/h
            self.d[-1] = 1.
            self.l[-2] = 0.
        #right neumann condition needs to be tested.
        if self.boundary_end[0] == 'neumann':
            self.u[1] = 0.
            self.d[0] = 1.
            self.d[-1] = 1/h
            self.l[-2] = -1/h

        self.band = (1,1)
        self.ab = np.array([self.u, self.d, self.l])


    def solve(self):
        #self.y[0] = self.boundary_start[1]
        #self.y[-1] = self.boundary_end[1]
        
        self._build_matrix()

        vec_f = np.vectorize(self.f)
        b = vec_f(self.x)
        b[0] = self.boundary_start[1]
        b[-1] = self.boundary_end[1]
        
        self.y = solve_banded(self.band, self.ab, b)

        ret_x = self.x.copy() 
        ret_y = self.y.copy()
        
        return ret_x, ret_y
   
class upp_solver_n2(fdsolver):
    def __init__(
                 self, start=0, end=1, n=10,
                 boundary_start=('neumann', 0), 
                 boundary_end = ('dirichlet', 0),
                 f=None):
        super().__init__(start, end, n, boundary_start, boundary_end, f)
        
    def _build_matrix(self):
        n, h = self.n, self.h
        self.u = (1./h**2) * np.ones(n+1)
        self.d = -(2./h**2) * np.ones(n+1)
        self.l = (1./h**2) * np.ones(n+1)
        self.u[0] = 0
        self.l[-1] = 0
        if self.boundary_start[0] == 'neumann':
            self.u[1] = 1/h
            self.d[0] = - 1/h
            self.d[-1] = 1.
            self.l[-2] = 0.
        if self.boundary_end[0] == 'neumann':
            self.u[1] = 0.
            self.d[0] = 1.
            self.d[-1] = 1/h
            self.l[-2] = -1/h
        self.band = (1,1)
        self.ab = np.array([self.u, self.d, self.l])


    def solve(self):
        self._build_matrix()
        vec_f = np.vectorize(self.f)
        b = vec_f(self.x)
        b[0] = self.boundary_start[1]
        b[-1] = self.boundary_end[1]
        
        if self.boundary_start[0] == 'neumann':
            b[0] += self.h/2 * self.f(self.x[0]) 
        if self.boundary_end[0] == 'neumann':
            b[-1] += self.h/2 * self.f(self.x[-1])
        
        self.y = solve_banded(self.band, self.ab, b)

        ret_x = self.x.copy() 
        ret_y = self.y.copy()
        
        return ret_x, ret_y
   
class upp_solver_n3(fdsolver):
    def __init__(
                 self, start=0, end=1, n=10,
                 boundary_start=('neumann', 0), 
                 boundary_end = ('dirichlet', 0),
                 f=None):
        super().__init__(start, end, n, boundary_start, boundary_end, f)
        
    def _build_matrix(self):
        n, h = self.n, self.h
        self.p = np.zeros(n+1)
        self.u = (1./h**2) * np.ones(n+1)
        self.d = -(2./h**2) * np.ones(n+1)
        self.l = (1./h**2) * np.ones(n+1)
        self.u[0] = 0
        self.l[-1] = 0
        if self.boundary_start[0] == 'neumann':
            self.p[2] = -1/(2*h)
            self.u[1] = 2/h
            self.d[0] = -3/2 * 1/h
            self.d[-1] = 1.
            self.l[-2] = 0.
            self.band = (1,2)
            self.ab = np.array([self.p, self.u, self.d, self.l])
        if self.boundary_end[0] == 'neumann':
            self.u[1] = 0.
            self.d[0] = 1.
            self.d[-1] = -2/3 * 1/h
            self.l[-2] = 2/h 
            self.p[-3] = -1/(2*h)
            self.band = (2,1)
            self.ab = np.array([self.u, self.d, self.l, self.p])

        

    def solve(self):
        self._build_matrix()
        vec_f = np.vectorize(self.f)
        b = vec_f(self.x)
        b[0] = self.boundary_start[1]
        b[-1] = self.boundary_end[1]
        
        
        self.y = solve_banded(self.band, self.ab, b)

        ret_x = self.x.copy() 
        ret_y = self.y.copy()
        
        return ret_x, ret_y
       

#nonlinear solver, neumann condition is not implemented
class nl_solver(solver):
    def __init__(self, start=0, end=1, n=10,
                 boundary_start=('dirichlet', 0),
                 boundary_end=('dirichlet', 0), 
                 tol=1e-5, max_it=1000):
        super().__init__(start, end, n, boundary_start, boundary_end)
        self.tol = tol
        self.max_it = max_it
    
    def newton_method(self, G, J, init_y, max_it, tol):
        n = len(init_y)
        y = init_y
        for i in range(max_it):
            J_val = np.array([[self.J[i][j](y) for j in range(n)] for i in range(n)])
            G_val = np.array([self.G[i](y) for i in range(n)])
            y_next = np.linalg.solve(J_val, -G_val)
            
            if norm(y_next-y) < tol:
                break
            y = y_next
        return y

        
    def solve(self):
        y = self.newton_method(self.G, self.J, self.init_y, self.max_it, self.tol)         
        return y
 