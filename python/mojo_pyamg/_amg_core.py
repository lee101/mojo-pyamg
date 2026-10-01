"""A thin NumPy-shaped shim over the ported `amg_core` kernels.

The krylov solvers upstream call `amg_core.apply_givens(Q, v, n, nrot)` and
friends with NumPy arrays.  Keeping the same call shape here means the solver
bodies stay identical to upstream's while the work happens in Mojo.
"""

from ._lib import addr, f64, lib


def apply_householders(z, B, n, start, stop, step):
    """z = (I - 2 w w^H) z for each Householder vector B[:, range(start,stop,step)]."""
    z = f64(z)
    B = f64(B)
    lib().mpa_apply_householders(addr(z), addr(B), n, start, stop, step)
    return z


def householder_hornerscheme(z, B, y, n, start, stop, step):
    """Map a least-squares solution y back through the Householder reflectors."""
    z = f64(z)
    B = f64(B)
    y = f64(y)
    lib().mpa_householder_hornerscheme(addr(z), addr(B), addr(y), n, start, stop,
                                       step)
    return z


def apply_givens(B, x, n, nrot):
    """Apply the first nrot Givens rotations in B (each 4 entries) to x."""
    x = f64(x)
    B = f64(B)
    lib().mpa_apply_givens(addr(B), addr(x), n, nrot)
    return x


__all__ = ['apply_householders', 'householder_hornerscheme', 'apply_givens']
