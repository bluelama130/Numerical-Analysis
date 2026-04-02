import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
class solver2d:
    def __init__(self, ld=(0.,0.), ur=(1.,1.), n=10):
        self.set_grid(ld, ur, n)

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, _n):
        self._n = _n
        self.x, self.dx = np.linspace(self.ld[0], self.ur[0], _n+1, retstep=True)
        self.y, self.dy = np.linspace(self.ld[1], self.ur[1], _n+1, retstep=True)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing = 'xy')
        self.U = np.zeros_like(self.X)
        
    
        
    def set_grid(self, ld, ur, _n):
        self.ld = ld
        self.ur = ur
        self.n = _n

    def solve():
        pass
#finite difference solver, neumann condition is not implemented
class fdsolver2d(solver2d):
    def __init__(self, ld=(0.,0.), ur=(1.,1.), n=10, f=None, bc = None):
        super().__init__()
        if f != None:
            self.f = f
    @property
    def f(self):
        return self._f
    @f.setter
    def f(self, _f):
        self._f = _f
    @property
    def bc(self):
        return self._bc
    @bc.setter
    def bc(self, bc):
        self._bc = bc

class poisson_solver(fdsolver2d):
    def __init__(self, ld=(0.,0.), ur=(1.,1.), n=10, f=None):
        super().__init__(ld, ur, n)
        
    def build_matrix(self):
        dx, dy = self.dx, self.dy
        n = self.n
        
        e = np.ones(n-1)
        I = np.eye(n-1)
        T = dia_array(([e, -2.*e, e], [-1, 0, 1]), shape=(n-1, n-1))
        Ax = T / dx**2
        Ay = T / dy**2
        self.A = kron(I,Ax) + kron(Ay, I)
        
        self.B = self.f(self.X, self.Y)

        h,j,k,l = self.bc[0], self.bc[1], self.bc[2], self.bc[3]
        self.U[:,0] = h(self.Y[:,0])
        self.U[:,-1] = j(self.Y[:,-1])
        self.U[0,:] = k(self.X[0,:])
        self.U[-1,:] = l(self.X[-1,:])

        for i in range(1,n):
            self.B[1,i] -= self.U[0,i]/self.dx**2
            self.B[-2,i] -= self.U[-1,i]/self.dx**2
            self.B[i,1] -= self.U[i,0]/self.dy**2
            self.B[i,-2] -= self.U[i,-1]/self.dy**2


    #we assume that the boundary condition has been applied manually 
    def solve(self):
        self.build_matrix()
        n = self.n
            
        b = np.ravel(self.B[1:-1, 1:-1], order='C')

        u = spsolve(self.A,b)                
        self.U[1:-1,1:-1] = u.reshape(n-1,n-1)
        
        return self.X,self.Y,self.U


        
        
        

        

