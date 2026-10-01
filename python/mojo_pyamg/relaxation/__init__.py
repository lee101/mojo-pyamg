"""Relaxation methods.

The multigrid cycle is formed by two complementary procedures: relaxation and
coarse-grid correction.  The role of relaxation is to rapidly damp oscillatory
(high-frequency) errors out of the approximate solution.  When the error is
smooth, it can then be accurately represented on the coarser grid, where a
solution, or approximate solution, can be computed.

The relaxation methods implemented here are:
    1. Jacobi iteration
    2. Gauss-Seidel iteration
    3. Successive Over-Relaxation
    4. Polynomial smoothing (e.g. Chebyshev)
    5. Jacobi and Gauss-Seidel on the normal equations (A.H A and A A.H)
    6. Krylov methods: gmres, cg, cgnr, cgne
    7. No pre- or postsmoother

Refer to the docstrings of the individual methods for additional information.

`utils` is not imported here (upstream does the same): it depends on
MultilevelSolver, which depends on this package.
"""

from . import chebyshev
from . import relaxation
from . import smoothing

from .relaxation import (block_gauss_seidel, block_jacobi, gauss_seidel,
                         gauss_seidel_indexed, gauss_seidel_ne, gauss_seidel_nr,
                         jacobi, jacobi_indexed, jacobi_ne, polynomial, sor)

__all__ = ['chebyshev', 'relaxation', 'smoothing', 'block_gauss_seidel',
           'block_jacobi', 'gauss_seidel', 'gauss_seidel_indexed',
           'gauss_seidel_ne', 'gauss_seidel_nr', 'jacobi', 'jacobi_indexed',
           'jacobi_ne', 'polynomial', 'sor']
