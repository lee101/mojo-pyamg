"""Krylov solvers.

The recurrences in this package are PyAMG's, kept as upstream wrote them.  The
Householder and Givens helpers they lean on (`apply_householders`,
`householder_hornerscheme`, `apply_givens`) are the ported Mojo kernels from
`amg_core/krylov.h`; see `mojo_pyamg._amg_core`.
"""

from ._gmres import gmres
from ._gmres_householder import gmres_householder
from ._gmres_mgs import gmres_mgs
from ._fgmres import fgmres
from ._cg import cg
from ._cr import cr
from ._cgnr import cgnr
from ._cgne import cgne
from ._bicgstab import bicgstab
from ._steepest_descent import steepest_descent
from ._minimal_residual import minimal_residual

__all__ = [
    'bicgstab',
    'cg',
    'cgne',
    'cgnr',
    'cr',
    'fgmres',
    'gmres',
    'gmres_householder',
    'gmres_mgs',
    'minimal_residual',
    'steepest_descent',
]
