import numpy as np
from mpi4py import MPI
import math
import os

class SharedMatrix:
    def __init__(self, rows, cols, comm=None):
        self.rows = rows
        self.cols = cols
        self.data = None
        self.comm = comm if comm else MPI.COMM_WORLD
        self.win = None
        
        # Allocate shared memory
        size = rows * cols
        if comm.Get_rank() == 0:
            self.win = MPI.Win.Allocate_shared(size * 8, 8, comm=self.comm)
        else:
            self.win = MPI.Win.Allocate_shared(0, 8, comm=self.comm)
        
        # Get the shared array
        buf, itemsize = self.win.Shared_query(0)
        self.data = np.ndarray(buffer=buf, shape=(rows, cols), dtype=np.float64)
        
        # Initialize to zero on rank 0
        if comm.Get_rank() == 0:
            self.data.fill(0.0)
        
        self.win.Fence()
    
    def free(self):
        if self.win:
            self.win.Fence()
            self.win.Free()
            self.win = None
        self.data = None

def create_shared_matrix(rows, cols, comm):
    """Create a shared matrix across MPI processes"""
    return SharedMatrix(rows, cols, comm)

def free_shared_matrix(mat):
    """Free shared matrix"""
    if mat:
        mat.free()

def generate_five_diag_mpi(xn, yn, comm):
    """Generate a five-diagonal matrix for Poisson equation"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    n = xn * yn
    A = create_shared_matrix(n, n, comm)
    
    # Distribute rows among processes
    rows_per_proc = n // size
    remainder = n % size
    
    start_row = rank * rows_per_proc + min(rank, remainder)
    end_row = start_row + rows_per_proc + (1 if rank < remainder else 0)
    
    # Each process fills its assigned rows
    for i in range(start_row, end_row):
        # Center
        A.data[i, i] = 4.0
        
        # Up (i - xn)
        if i >= xn:
            A.data[i, i - xn] = -1.0
        
        # Down (i + xn)
        if i + xn < n:
            A.data[i, i + xn] = -1.0
        
        # Left (i - 1) - not on left edge
        if i % xn != 0:
            A.data[i, i - 1] = -1.0
        
        # Right (i + 1) - not on right edge
        if i % xn != xn - 1:
            A.data[i, i + 1] = -1.0
    
    A.win.Fence()
    print(f"Process with rank: {rank} ended matrix creation")
    return A

def cholesky_mpi(A, n, comm):
    """Parallel Cholesky decomposition using MPI shared memory"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Verify matrix dimensions on rank 0
    if rank == 0:
        assert A.cols == A.rows == n
    
    # Create shared matrix for L
    L = create_shared_matrix(n, n, comm)
    
    # Ensure A is synchronized if it's shared
    if A.win:
        A.win.Fence()
    
    for j in range(n):
        L.win.Fence()
        
        # Process 0 computes the diagonal element
        if rank == 0:
            s = 0.0
            for k in range(j):
                s += L.data[j, k] * L.data[j, k]
            L.data[j, j] = math.sqrt(A.data[j, j] - s)
        
        L.win.Fence()
        
        # All processes need the diagonal value
        diag_val = L.data[j, j]
        
        # Parallelize the i-loop across processes
        remaining_rows = n - j - 1
        if remaining_rows > 0:
            chunk_size = remaining_rows // size
            remainder = remaining_rows % size
            
            local_count = chunk_size + (1 if rank < remainder else 0)
            
            # Calculate global start for each process
            offset = 0
            for p in range(rank):
                offset += chunk_size + (1 if p < remainder else 0)
            local_start = j + 1 + offset
            
            # Each process computes its assigned rows
            for i in range(local_start, local_start + local_count):
                s = 0.0
                for k in range(j):
                    s += L.data[i, k] * L.data[j, k]
                L.data[i, j] = (A.data[i, j] - s) / diag_val
        
        L.win.Fence()
    
    # Final synchronization
    L.win.Fence()
    return L

def solve_gauss_forward(L, b):
    """Forward substitution (Ly = b)"""
    n = L.shape[0]
    y = np.zeros(n)
    
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        
        if abs(L[i, i]) < 1e-12:
            print(f"Error: zero diagonal element in row {i}!")
            return None
        
        y[i] = s / L[i, i]
    
    return y

def solve_gauss_reverse(U, y):
    """Backward substitution (Ux = y)"""
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= U[i, j] * x[j]
        
        if abs(U[i, i]) < 1e-12:
            print(f"Error: zero diagonal element in row {i}!")
            return None
        
        x[i] = s / U[i, i]
    
    return x

def pcg_preconditioned(A, b, x0, tol, max_iter=1000, L=None, Lt=None):
    """Preconditioned Conjugate Gradient method with Cholesky preconditioner"""
    n = len(b)
    x = x0.copy()
    r = b - A @ x
    
    # If preconditioner is provided
    if L is not None and Lt is not None:
        z = solve_gauss_forward(L, r)
        z = solve_gauss_reverse(Lt, z)
    else:
        z = r.copy()
    
    p = z.copy()
    
    r0_norm = np.linalg.norm(r)
    
    for k in range(max_iter):
        q = A @ p
        pq = np.dot(p, q)
        
        if abs(pq) < 1e-15:
            break
            
        alpha = np.dot(z, r) / pq
        
        x += alpha * p
        r_new = r - alpha * q
        
        # Check convergence
        rel_res = np.linalg.norm(r_new) / r0_norm
        if rel_res < tol:
            r = r_new
            break
        
        # Preconditioning
        if L is not None and Lt is not None:
            z_new = solve_gauss_forward(L, r_new)
            z_new = solve_gauss_reverse(Lt, z_new)
        else:
            z_new = r_new.copy()
        
        beta = np.dot(z_new, r_new) / np.dot(z, r)
        
        p = z_new + beta * p
        r = r_new
        z = z_new
    
    return x, k+1, np.linalg.norm(r) / r0_norm

def f(x, y):
    """Right-hand side function"""
    return -2 * np.sin(x) * np.cos(y)

def u(x, y):
    """Exact solution function"""
    return np.sin(x) * np.cos(y)

def F_func(x, y):
    """Create right-hand side vector for Poisson equation"""
    h = x[1] - x[0]
    n_x = len(x) - 2
    n_y = len(y) - 2
    n = n_x * n_y
    
    R = np.zeros(n)
    k = 0
    
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            R[k] = f(x[i], y[j]) * h * h
            
            # Boundary conditions
            if i == 1:
                R[k] -= u(x[0], y[j])
            if i == len(x)-2:
                R[k] -= u(x[-1], y[j])
            if j == 1:
                R[k] -= u(x[i], y[0])
            if j == len(y)-2:
                R[k] -= u(x[i], y[-1])
            
            k += 1
    
    return R

def u_for_xy(x, y):
    """Create vector of exact solution values at interior points"""
    n_x = len(x) - 2
    n_y = len(y) - 2
    n = n_x * n_y
    
    R = np.zeros(n)
    k = 0
    
    for i in range(1, len(x)-1):
        for j in range(1, len(y)-1):
            R[k] = u(x[i], y[j])
            k += 1
    
    return R

def err_by_eps_pcg_chol(a, b, c, d, h, comm):
    """Main function to test PCG with Cholesky preconditioner"""
    rank = comm.Get_rank()
    
    # Create grid
    x = np.linspace(a, b, int((b - a) / h) + 1)
    y = np.linspace(c, d, int((d - c) / h) + 1)
    
    # Generate matrices in parallel
    A_mpi = generate_five_diag_mpi(len(x) - 2, len(y) - 2, comm)
    L_mpi = cholesky_mpi(A_mpi, A_mpi.cols, comm)
    
    if rank == 0:
        print("Starting PCG tests")
        
        # Get data as numpy arrays
        A = A_mpi.data
        L = L_mpi.data
        
        # Create exact solution and right-hand side
        u_exact = u_for_xy(x, y)
        b_vec = F_func(x, y)
        
        n = len(b_vec)
        
        # Test different tolerances
        with open("pcgCholErr.txt", "w") as file:
            for i in range(1, 11):
                eps = 10 ** (-i)
                x0 = np.zeros(n)
                
                # Run PCG
                sol, iterations, relres = pcg_preconditioned(
                    A, b_vec, x0, eps, 
                    L=L, Lt=L.T
                )
                
                # Calculate error
                max_err = np.max(np.abs(u_exact - sol))
                
                file.write(f"{eps:.15f} {max_err:.15f}\n")
                print(f"eps={eps:.1e}, iterations={iterations}, max_err={max_err:.3e}, relres={relres:.3e}")
                
                # Print solution for debugging at i=10
                if i == 10:
                    for j in range(min(10, len(u_exact))):
                        print(f"u[{j}]={u_exact[j]:.5f}, sol[{j}]={sol[j]:.5f}")
    
    # Free shared memory
    free_shared_matrix(A_mpi)
    free_shared_matrix(L_mpi)
    
    if rank == 0:
        print("---\n")

def main():
    """Main MPI function"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Starting MPI program with {size} processes")
    
    # Parameters
    a, b = 0, 1.625
    c, d = 0, 1.625
    h = 0.025
    
    comm.Barrier()
    
    # Run the main test
    err_by_eps_pcg_chol(a, b, c, d, h, comm)
    
    if rank == 0:
        print("Program completed successfully")

if __name__ == "__main__":
    main()