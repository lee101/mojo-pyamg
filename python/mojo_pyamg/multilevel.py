"""Generic AMG solver.  Mirrors `pyamg.multilevel`."""

from warnings import warn

import numpy as np
import scipy as sp
import scipy.sparse.linalg as sla
from scipy.linalg import pinv
from scipy.sparse.linalg import LinearOperator

from . import krylov
from .util.params import set_tol
from .util.utils import to_type
from .util import upcast
from .relaxation import smoothing


class MultilevelSolver:
    """Store a multigrid hierarchy and implement the multigrid cycle.

    Parameters
    ----------
    levels : list of Level
        Array of level objects that contain A, R, and P.
    coarse_solver : str, callable, tuple
        The solver method is either a string such as 'splu' or 'pinv', a
        callable object which receives only (A, b) and returns an
        (approximate or exact) solution to the linear system Ax = b, or a tuple
        of the form (str|callable, args) where args is a dictionary of
        arguments to be passed to the function denoted by string or callable.

        Sparse direct methods:  splu.
        Sparse iterative methods: any method in scipy.sparse.linalg or
        mojo_pyamg.krylov (e.g. 'cg'); methods in mojo_pyamg.krylov take
        precedence; relaxation methods such as 'gauss_seidel' or 'jacobi'.
        Dense methods: pinv, lu, cholesky.

    Attributes
    ----------
    levels : level array
        Array of level objects that contain A, R, and P.
    coarse_solver : str
        String passed to coarse_grid_solver indicating the solve type.

    Methods
    -------
    aspreconditioner(), cycle_complexity(), grid_complexity(),
    operator_complexity(), solve(), change_solve_matrix()

    Notes
    -----
    If not defined, the R attribute on each level is set to the conjugate
    transpose of P.

    Examples
    --------
    >>> ml = smoothed_aggregation_solver(A)
    >>> print(ml)
    MultilevelSolver
    Number of Levels:     2
    Operator Complexity:   1.891
    Grid Complexity:       1.500
    Coarse Solver:        'pinv'

    """

    class Level:
        """A single level of a multigrid hierarchy (empty)."""

        def __init__(self):
            """Level construct (empty)."""
            self.A = None

    class level(Level):  # noqa: N801
        """Deprecated level class."""

        def __init__(self):
            """Raise deprecation warning on use, not import."""
            super().__init__()
            warn('level() is deprecated.  use Level()',
                 category=DeprecationWarning, stacklevel=2)

    def __init__(self, levels, coarse_solver='pinv'):
        """Initialize the cycle and ensure complete list of levels."""
        self.symmetric_smoothing = False  # force change_smoothers to set to True
        self.levels = levels
        self.coarse_solver = coarse_grid_solver(coarse_solver)

        for level in levels[:-1]:
            if not hasattr(level, 'R'):
                level.R = level.P.T.conjugate()

    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy."""
        output = 'MultilevelSolver\n'
        output += f'Number of Levels:     {len(self.levels)}\n'
        output += f'Operator Complexity:  {self.operator_complexity():6.3f}\n'
        output += f'Grid Complexity:      {self.grid_complexity():6.3f}\n'
        output += f'Coarse Solver:        {self.coarse_solver.name()}\n'

        total_nnz = sum(level.A.nnz for level in self.levels)

        output += '  level   unknowns     nonzeros\n'
        for n, level in enumerate(self.levels):
            A = level.A
            ratio = 100 * A.nnz / total_nnz
            output += f'{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{ratio:2.2f}%]\n'

        return output

    def cycle_complexity(self, cycle='V'):
        """Cycle complexity of V, W, AMLI, and F(1,1) cycles.

        Cycle complexity is an approximate measure of the number of floating
        point operations (FLOPs) required to perform a single multigrid cycle
        relative to the cost of a single smoothing operation.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        float
            F_sum / F_0, where F_sum is the total number of nonzeros in the
            matrix on all levels encountered during a cycle and F_0 the
            nonzeros on the finest level.

        Notes
        -----
        The estimate assumes that the cost of pre- and post-smoothing are
        (each) equal to the number of nonzeros in the matrix on that level.

        """
        cycle = str(cycle).upper()

        nnz = [level.A.nnz for level in self.levels]

        def V(level):
            if len(self.levels) == 1:
                return nnz[0]

            if level == len(self.levels) - 2:
                return 2 * nnz[level] + nnz[level + 1]

            return 2 * nnz[level] + V(level + 1)

        def W(level):
            if len(self.levels) == 1:
                return nnz[0]

            if level == len(self.levels) - 2:
                return 2 * nnz[level] + nnz[level + 1]

            return 2 * nnz[level] + 2 * W(level + 1)

        def F(level):
            if len(self.levels) == 1:
                return nnz[0]

            if level == len(self.levels) - 2:
                return 2 * nnz[level] + nnz[level + 1]

            return 2 * nnz[level] + F(level + 1) + V(level + 1)

        if cycle == 'V':
            flops = V(0)
        elif cycle in ('W', 'AMLI'):
            flops = W(0)
        elif cycle == 'F':
            flops = F(0)
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

        return float(flops) / float(nnz[0])

    def operator_complexity(self):
        """Nonzeros on all levels / nonzeros on the finest level."""
        return sum(level.A.nnz for level in self.levels) / \
            float(self.levels[0].A.nnz)

    def grid_complexity(self):
        """Unknowns on all levels / unknowns on the finest level."""
        return sum(level.A.shape[0] for level in self.levels) / \
            float(self.levels[0].A.shape[0])

    def change_solve_matrix(self, A):
        """Change the matrix that is solved (or preconditioned)."""
        self.levels[0].A = A

        smoothing.rebuild_smoother(self.levels[0])

    def psolve(self, b):
        """Legacy solve interface: one iteration."""
        return self.solve(b, maxiter=1)

    def aspreconditioner(self, cycle='V'):
        """Create a preconditioner using this multigrid cycle.

        Parameters
        ----------
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.

        Returns
        -------
        LinearOperator
            Preconditioner suitable for the iterative solvers defined in
            scipy.sparse.linalg and mojo_pyamg.krylov.

        See Also
        --------
        MultilevelSolver.solve

        """
        shape = self.levels[0].A.shape
        dtype = self.levels[0].A.dtype

        def matvec(b):
            return self.solve(b, maxiter=1, cycle=cycle, tol=1e-12)

        return LinearOperator(shape, matvec, dtype=dtype)

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle='V', accel=None,
              callback=None, residuals=None, cycles_per_level=1, return_info=False):
        """Execute multigrid cycling.

        Parameters
        ----------
        b : array
            Right hand side of the linear system Ax = b.
        x0 : array
            Initial guess for the solution, default is a vector of zeros.
        tol : float
            Tolerance for stopping criteria, let r = r_k, x = x_k.
        maxiter : int
            Maximum number of cycles to perform.
        cycle : {'V','W','F','AMLI'}
            Type of multigrid cycle to perform in each iteration.
        accel : str, function
            Optional Krylov accelerator, e.g. 'cg' or 'fgmres'.
        callback : function
            User-supplied function is called after each iteration as
            callback(xk), where xk is the current solution vector.
        residuals : list
            Residual history in the 2-norm, including the initial residual.
        cycles_per_level : int, default 1
            Number of V-cycles on each level of an F-cycle.
        return_info : bool
            If true, will return ``(x, info)``; if false, returns ``x``.

        Returns
        -------
        array
            Approximate solution to Ax=b after k iterations.
        str
            Halting status: 0 is a successful exit, >0 returns the iteration
            count when convergence to tolerance is not achieved.

        See Also
        --------
        aspreconditioner

        """
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = np.array(x0)  # copy

        A = self.levels[0].A

        cycle = str(cycle).upper()

        # AMLI cycles require hermitian matrix
        if (cycle == 'AMLI') and hasattr(A, 'symmetry'):
            if A.symmetry != 'hermitian':
                raise ValueError('AMLI cycles require \
                    symmetry to be hermitian')

        if accel is not None:

            # Check for symmetric smoothing scheme when using CG
            if (accel == 'cg') and (not self.symmetric_smoothing):
                warn('Incompatible non-symmetric multigrid preconditioner '
                     'detected, due to presmoother/postsmoother combination. '
                     'CG requires SPD preconditioner, not just SPD matrix.')

            # Check for AMLI compatibility
            if (accel != 'fgmres') and (cycle == 'AMLI'):
                raise ValueError('AMLI cycles require acceleration (accel) '
                                 'to be fgmres, or no acceleration')

            # Acceleration is being used
            kwargs = {}
            if isinstance(accel, str):
                kwargs = {}
                if hasattr(krylov, accel):
                    accel = getattr(krylov, accel)
                else:
                    accel = getattr(sla, accel)

            M = self.aspreconditioner(cycle=cycle)

            try:  # try PyAMG style interface which has a residuals parameter
                x, info = accel(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M,
                                callback=callback, residuals=residuals, **kwargs)
                if return_info:
                    return x, info
                return x
            except TypeError:
                # try the scipy.sparse.linalg style interface,
                # which requires a callback function if a residual
                # history is desired

                if residuals is not None:
                    residuals[:] = [np.linalg.norm(b - A @ x)]

                    def callback_wrapper(x):
                        if np.isscalar(x):
                            residuals.append(x)
                        else:
                            residuals.append(np.linalg.norm(b - A @ x))
                        if callback is not None:
                            callback(x)
                else:
                    callback_wrapper = callback

                # for scipy solvers, see if rtol is available
                kwargs['rtol'] = tol
                kwargs['atol'] = 0

                x, info = accel(A, b, x0=x0, maxiter=maxiter, M=M,
                                callback=callback_wrapper, **kwargs)
                if return_info:
                    return x, info
                return x

        else:
            # Scale tol by normb
            # Don't scale tol earlier. The accel routine should also scale tol
            normb = np.linalg.norm(b)
            if normb == 0.0:
                normb = 1.0  # set so that we have an absolute tolerance

        # Start cycling (no acceleration)
        normr = np.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals[:] = [normr]  # initial residual

        # Create uniform types for A, x and b
        tp = upcast(b.dtype, x.dtype, A.dtype)
        [b, x] = to_type(tp, [b, x])
        b = np.ravel(b)
        x = np.ravel(x)

        it = 0

        while True:
            if len(self.levels) == 1:
                # hierarchy has only 1 level
                x = self.coarse_solver(A, b)
            else:
                self.__solve(0, x, b, cycle, cycles_per_level)

            it += 1

            normr = np.linalg.norm(b - A @ x)
            if residuals is not None:
                residuals.append(normr)

            if callback is not None:
                callback(x)

            if normr < tol * normb:
                if return_info:
                    return x, 0
                return x

            if it == maxiter:
                if return_info:
                    return x, it
                return x

    def __solve(self, lvl, x, b, cycle, cycles_per_level=1):
        """Multigrid cycling on level `lvl`."""
        A = self.levels[lvl].A

        self.levels[lvl].presmoother(A, x, b)

        residual = b - A @ x

        coarse_b = self.levels[lvl].R @ residual
        coarse_x = np.zeros_like(coarse_b)

        if lvl == len(self.levels) - 2:
            coarse_x[:] = self.coarse_solver(self.levels[-1].A, coarse_b)
        elif cycle == 'V':
            self.__solve(lvl + 1, coarse_x, coarse_b, 'V')
        elif cycle == 'W':
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle)
        elif cycle == 'F':
            self.__solve(lvl + 1, coarse_x, coarse_b, cycle, cycles_per_level)
            for _ in range(0, cycles_per_level):
                self.__solve(lvl + 1, coarse_x, coarse_b, 'V', 1)
        elif cycle == 'AMLI':
            # Run nAMLI AMLI cycles, which compute "optimal" corrections by
            # orthogonalizing the coarse-grid corrections in the A-norm
            nAMLI = 2
            Ac = self.levels[lvl + 1].A
            p = np.zeros((nAMLI, coarse_b.shape[0]), dtype=np.float64)
            beta = np.zeros((nAMLI, nAMLI), dtype=np.float64)
            for k in range(nAMLI):
                # New search direction --> M^{-1}@residual
                p[k, :] = 1
                self.__solve(lvl + 1, p[k, :].reshape(coarse_b.shape),
                             coarse_b, cycle)

                # Orthogonalize new search direction to old directions
                for j in range(k):  # loops from j = 0...(k-1)
                    beta[k, j] = np.inner(p[j, :].conj(), Ac @ p[k, :]) / \
                        np.inner(p[j, :].conj(), Ac @ p[j, :])
                    p[k, :] -= beta[k, j] * p[j, :]

                # Compute step size
                Ap = Ac @ p[k, :]
                alpha = np.inner(p[k, :].conj(), np.ravel(coarse_b)) / \
                    np.inner(p[k, :].conj(), Ap)

                # Update solution
                coarse_x += alpha * p[k, :].reshape(coarse_x.shape)

                # Update residual
                coarse_b -= alpha * Ap.reshape(coarse_b.shape)
        else:
            raise TypeError(f'Unrecognized cycle type ({cycle})')

        x += self.levels[lvl].P @ coarse_x   # coarse grid correction

        self.levels[lvl].postsmoother(A, x, b)


def coarse_grid_solver(solver):
    """Return a coarse grid solver suitable for MultilevelSolver.

    Parameters
    ----------
    solver : str, callable, tuple
        The solver method is either a string such as 'splu' or 'pinv', a
        callable object which receives only (A, b), or a tuple of the form
        (string|callable, args) where args is a dictionary of arguments.

        Valid strings: splu; any method in scipy.sparse.linalg or
        mojo_pyamg.krylov (e.g. 'cg'); relaxation methods such as
        'gauss_seidel' or 'jacobi'; and the dense methods pinv, lu, cholesky.

    Returns
    -------
    GenericSolver
        A class for use as a standalone or coarse grids solver.

    Examples
    --------
    >>> A = poisson((10, 10), format='csr')
    >>> b = A @ np.ones(A.shape[0])
    >>> cgs = coarse_grid_solver('lu')
    >>> x = cgs(A, b)

    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    solver, kwargs = unpack_arg(solver)

    if solver in ['pinv', 'pinv2']:
        def solve(self, A, b):
            if not hasattr(self, 'P'):
                self.P = pinv(A.toarray(), **kwargs)
            return np.dot(self.P, b)

    elif solver == 'lu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                self.LU = sp.linalg.lu_factor(A.toarray(), **kwargs)
            return sp.linalg.lu_solve(self.LU, b)

    elif solver == 'cholesky':
        def solve(self, A, b):
            if not hasattr(self, 'L'):
                self.L = sp.linalg.cho_factor(A.toarray(), **kwargs)
            return sp.linalg.cho_solve(self.L, b)

    elif solver == 'splu':
        def solve(self, A, b):
            if not hasattr(self, 'LU'):
                # for multiple candidates in B, A will often have a couple zero
                # rows/columns that must be removed
                Acsc = A.tocsc()
                Acsc.eliminate_zeros()
                diffptr = Acsc.indptr[:-1] - Acsc.indptr[1:]
                nonzero_cols = (diffptr != 0).nonzero()[0]
                Map = sp.sparse.eye_array(Acsc.shape[0], Acsc.shape[1],
                                          format='csc')
                Map = Map[:, nonzero_cols]
                Acsc = Map.T.tocsc() @ Acsc @ Map
                self.LU = sp.sparse.linalg.splu(Acsc, **kwargs)
                self.LU_Map = Map

            return self.LU_Map @ self.LU.solve(np.ravel(self.LU_Map.T @ b))

    elif solver in ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'qmr', 'minres']:
        if hasattr(krylov, solver):
            fn = getattr(krylov, solver)
        else:
            fn = getattr(sla, solver)

        def solve(_, A, b):
            if 'tol' not in kwargs:
                kwargs['tol'] = set_tol(A.dtype)

            return fn(A, b, **kwargs)[0]

    elif solver in ['gauss_seidel', 'jacobi', 'block_gauss_seidel',
                    'block_jacobi', 'richardson', 'sor', 'chebyshev',
                    'jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']:

        if 'iterations' not in kwargs:
            kwargs['iterations'] = 10

        def solve(_, A, b):

            lvl = MultilevelSolver.Level()
            lvl.A = A
            fn = getattr(smoothing, 'setup_' + str(solver))
            relax = fn(lvl, **kwargs)
            x = np.zeros_like(b)
            relax(A, x, b)

            return x

    elif solver is None:
        # No coarse grid solve
        def solve(_, __, b):
            return 0 * b  # should this return b instead?

    elif callable(solver):
        def solve(_, A, b):
            return solver(A, b, **kwargs)

    else:
        raise ValueError(f'unknown solver: {solver}')

    class GenericSolver:
        """Generic solver class."""

        def __call__(self, A, b):
            # make sure x is same dimensions and type as b
            b = np.asanyarray(b)

            if A.nnz == 0:
                # if A.nnz = 0, then we expect no correction
                x = np.zeros(b.shape)
            else:
                x = solve(self, A, b)

            if isinstance(b, np.ndarray):
                x = np.asarray(x)
            elif isinstance(b, np.matrix):
                # convert to ndarray
                b = np.asarray(b)
                x = np.asarray(x)
            else:
                raise ValueError('unrecognized type')

            return x.reshape(b.shape)

        def __repr__(self):
            return 'coarse_grid_solver(' + repr(solver) + ')'

        @classmethod
        def name(cls):
            """Return the coarse solver name."""
            return repr(solver)

    return GenericSolver()


class multilevel_solver(MultilevelSolver):  # noqa: N801
    """Deprecated alias of MultilevelSolver."""

    def __init__(self, *args, **kwargs):
        """Raise deprecation warning on use, not import."""
        super().__init__(*args, **kwargs)
        warn('multilevel_solver is deprecated.  use MultilevelSolver()',
             category=DeprecationWarning, stacklevel=2)


__all__ = ['MultilevelSolver', 'coarse_grid_solver', 'multilevel_solver']
