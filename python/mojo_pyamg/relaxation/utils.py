"""Relaxation-centric utilities.  Mirrors `pyamg.relaxation.utils`."""

import numpy as np
from scipy.sparse.linalg import LinearOperator

from ..multilevel import MultilevelSolver
from .. import relaxation


def relaxation_as_linear_operator(method, A, b):
    """A linear operator that applies a relaxation method to a fixed right-hand side.

    Parameters
    ----------
    method : {tuple or string}
        Relaxation descriptor: ('method', opts) or 'method', e.g.
        'gauss_seidel' or ('gauss_seidel', {'sweep': 'symmetric'}).
    A : sparse matrix
        System matrix in A x = b.
    b : array
        Right-hand side in A x = b.

    Returns
    -------
    LinearOperator
        Applies the relaxation to a vector for the fixed right-hand side b.

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> A = poisson((100, 100), format='csr')
    >>> B = np.ones((A.shape[0], 1))
    >>> b = np.zeros((A.shape[0]))
    >>> relax = relaxation_as_linear_operator('gauss_seidel', A, b)
    >>> B = relax @ B

    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    # setup variables
    accepted_methods = ['gauss_seidel', 'block_gauss_seidel', 'sor',
                        'gauss_seidel_ne', 'gauss_seidel_nr', 'jacobi',
                        'block_jacobi', 'jacobi_ne', 'richardson']

    b = np.array(b, dtype=np.float64)
    fn, kwargs = unpack_arg(method)
    lvl = MultilevelSolver.Level()
    lvl.A = A

    # Retrieve setup call from relaxation.smoothing for this relaxation method
    if fn not in accepted_methods:
        raise NameError(f'invalid relaxation method: {fn}')
    setup_smoother = getattr(relaxation.smoothing, 'setup_' + fn)

    # Get relaxation routine that takes only (A, x, b) as parameters
    relax = setup_smoother(lvl, **kwargs)

    # Define matvec
    def matvec(x):
        xcopy = x.copy()
        relax(A, xcopy, b)
        return xcopy

    return LinearOperator(A.shape, matvec, dtype=np.float64)


__all__ = ['relaxation_as_linear_operator']
