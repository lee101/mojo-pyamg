"""Create pre and post-smoothers on the levels of a MultilevelSolver.

Mirrors `pyamg.relaxation.smoothing` for the relaxation methods this port
covers.  The `schwarz`, `strength_based_schwarz`, `cf_*` and `fc_*` families are
not part of this port (see the README coverage table).
"""

from functools import partial, update_wrapper

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

from ..util.utils import get_block_diag, get_diagonal, scale_rows
from ..util.linalg import approximate_spectral_radius
from ..krylov import cg, cgne, cgnr, gmres
from . import relaxation
from .chebyshev import chebyshev_polynomial_coefficients

# Default relaxation parameters
DEFAULT_SWEEP = 'forward'
DEFAULT_NITER = 1

# List of by-definition symmetric relaxation schemes, e.g. Jacobi.
SYMMETRIC_RELAXATION = ['jacobi', 'richardson', 'block_jacobi',
                        'jacobi_ne', 'chebyshev', None]

# List of supported Krylov relaxation schemes
KRYLOV_RELAXATION = ['cg', 'cgne', 'cgnr', 'gmres']


def _unpack_arg(v):
    """Split a ('method', opts) tuple."""
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def change_smoothers(ml, presmoother, postsmoother):
    """Assign presmoothers and postsmoothers to the levels of a hierarchy.

    Parameters
    ----------
    ml : MultilevelSolver
        Hierarchy whose levels are given a pre- and a postsmoother.
    presmoother, postsmoother : str, tuple, list, None
        One of: None, ('method', opts), 'method', or a list of such, one
        entry per level (the last entry repeats for the remaining levels).

    Returns
    -------
    None
        ml.symmetric_smoothing records whether the pre/post combination is
        symmetric; ml.levels[i].presmoother/postsmoother are set in place.

    """
    ml.symmetric_smoothing = True
    if isinstance(presmoother, (str, tuple)) or (presmoother is None):
        presmoother = [presmoother]
    elif not isinstance(presmoother, list):
        raise ValueError('Unrecognized presmoother -- use a string:\n '
                         '"method" or ("method", opts) or list thereof.')
    if isinstance(postsmoother, (str, tuple)) or (postsmoother is None):
        postsmoother = [postsmoother]
    elif not isinstance(postsmoother, list):
        raise ValueError('Unrecognized postsmoother -- use a string:\n '
                         '"method" or ("method", opts) or list thereof.')
    fn1 = None      # Predefine to keep scope beyond first loop
    fn2 = None
    kwargs1 = {}
    kwargs2 = {}
    min_len = min(len(presmoother), len(postsmoother), len(ml.levels[:-1]))
    for i in range(0, min_len):
        fn1, kwargs1 = _unpack_arg(presmoother[i])
        setup_presmoother = _setup_call(fn1)
        ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)
        fn2, kwargs2 = _unpack_arg(postsmoother[i])
        setup_postsmoother = _setup_call(fn2)
        ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)
        if 'iterations' in kwargs1:
            it1 = kwargs1['iterations']
        else:
            it1 = DEFAULT_NITER
        if 'iterations' in kwargs2:
            it2 = kwargs2['iterations']
        else:
            it2 = DEFAULT_NITER
        if it1 != it2:
            ml.symmetric_smoothing = False
        elif fn1 != fn2:
            ml.symmetric_smoothing = False
        elif fn1 in KRYLOV_RELAXATION or fn2 in KRYLOV_RELAXATION:
            ml.symmetric_smoothing = False
        elif fn1 not in SYMMETRIC_RELAXATION:
            sweep1 = kwargs1.get('sweep', DEFAULT_SWEEP)
            sweep2 = kwargs2.get('sweep', DEFAULT_SWEEP)
            if (sweep1, sweep2) not in [('forward', 'backward'),
                                        ('backward', 'forward'),
                                        ('symmetric', 'symmetric')]:
                ml.symmetric_smoothing = False
    if len(presmoother) < len(postsmoother):
        mid_len = min(len(postsmoother), len(ml.levels[:-1]))
        for i in range(min_len, mid_len):
            ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)
            fn2, kwargs2 = _unpack_arg(postsmoother[i])
            setup_postsmoother = _setup_call(fn2)
            ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)
            if 'iterations' in kwargs1:
                it1 = kwargs1['iterations']
            else:
                it1 = DEFAULT_NITER
            if 'iterations' in kwargs2:
                it2 = kwargs2['iterations']
            else:
                it2 = DEFAULT_NITER
            if it1 != it2:
                ml.symmetric_smoothing = False
            elif fn1 != fn2:
                ml.symmetric_smoothing = False
            elif fn1 in KRYLOV_RELAXATION or fn2 in KRYLOV_RELAXATION:
                ml.symmetric_smoothing = False
            elif fn1 not in SYMMETRIC_RELAXATION:
                sweep1 = kwargs1.get('sweep', DEFAULT_SWEEP)
                sweep2 = kwargs2.get('sweep', DEFAULT_SWEEP)
                if (sweep1, sweep2) not in (('forward', 'backward'),
                                            ('backward', 'forward'),
                                            ('symmetric', 'symmetric')):
                    ml.symmetric_smoothing = False
    elif len(presmoother) > len(postsmoother):
        mid_len = min(len(presmoother), len(ml.levels[:-1]))
        for i in range(min_len, mid_len):
            fn1, kwargs1 = _unpack_arg(presmoother[i])
            setup_presmoother = _setup_call(fn1)
            ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)
            ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)
            if 'iterations' in kwargs1:
                it1 = kwargs1['iterations']
            else:
                it1 = DEFAULT_NITER
            if 'iterations' in kwargs2:
                it2 = kwargs2['iterations']
            else:
                it2 = DEFAULT_NITER
            if it1 != it2:
                ml.symmetric_smoothing = False
            elif fn1 != fn2:
                ml.symmetric_smoothing = False
            elif fn1 in KRYLOV_RELAXATION or fn2 in KRYLOV_RELAXATION:
                ml.symmetric_smoothing = False
            elif fn1 not in SYMMETRIC_RELAXATION:
                sweep1 = kwargs1.get('sweep', DEFAULT_SWEEP)
                sweep2 = kwargs2.get('sweep', DEFAULT_SWEEP)
                if (sweep1, sweep2) not in [('forward', 'backward'),
                                            ('backward', 'forward'),
                                            ('symmetric', 'symmetric')]:
                    ml.symmetric_smoothing = False
    else:
        mid_len = min_len
    for i in range(mid_len, len(ml.levels[:-1])):
        ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)
        ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)


def rho_D_inv_A(A):
    """Spectral radius of diag(A)^-1 A."""
    if not hasattr(A, 'rho_D_inv'):
        D_inv = get_diagonal(A, inv=True)
        D_inv_A = scale_rows(A, D_inv, copy=True)
        A.rho_D_inv = approximate_spectral_radius(D_inv_A)
    return A.rho_D_inv


def rho_block_D_inv_A(A, Dinv):
    """Spectral radius of Dinv A, where Dinv holds the inverse blocks."""
    if not hasattr(A, 'rho_block_D_inv'):
        blocksize = Dinv.shape[1]
        if Dinv.shape[1] != Dinv.shape[2]:
            raise ValueError('Dinv has incorrect dimensions')
        if Dinv.shape[0] != int(A.shape[0] / blocksize):
            raise ValueError('Dinv and A have incompatible dimensions')
        Dinv = sparse.bsr_array((Dinv,
                                 np.arange(Dinv.shape[0], dtype=np.int32),
                                 np.arange(Dinv.shape[0] + 1, dtype=np.int32)),
                                shape=A.shape)

        def matvec(x):
            return Dinv @ (A @ x)
        D_inv_A = LinearOperator(A.shape, matvec, dtype=A.dtype)
        A.rho_block_D_inv = approximate_spectral_radius(D_inv_A)
    return A.rho_block_D_inv


def matrix_asformat(lvl, name, format, blocksize=None):
    """Return lvl.<name>, converted once to `format` and cached as lvl.<name><format>."""
    desired_matrix = name + format
    M = getattr(lvl, name)
    if format == 'bsr':
        desired_matrix += str(blocksize[0]) + str(blocksize[1])
    if hasattr(lvl, desired_matrix):
        pass
    elif M.format == format and format != 'bsr':
        setattr(lvl, desired_matrix, M)
    elif M.format == format and format == 'bsr':
        setattr(lvl, desired_matrix, M.tobsr(blocksize=blocksize))
    else:
        newM = getattr(M, 'to' + format)()
        setattr(lvl, desired_matrix, newM)
    return getattr(lvl, desired_matrix)


def setup_gauss_seidel(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP):
    """Gauss-Seidel relaxation smoother for level `lvl`."""
    smoother = partial(relaxation.gauss_seidel, iterations=iterations, sweep=sweep)
    update_wrapper(smoother, relaxation.gauss_seidel)  # set __name__
    return smoother


def setup_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=True):
    """Jacobi relaxation smoother for level `lvl`."""
    if withrho:
        omega = omega / rho_D_inv_A(lvl.A)
    smoother = partial(relaxation.jacobi, iterations=iterations, omega=omega)
    update_wrapper(smoother, relaxation.jacobi)  # set __name__
    return smoother


def setup_block_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, Dinv=None,
                       blocksize=None, withrho=True):
    """Block Jacobi relaxation smoother for level `lvl`."""
    if blocksize is None and Dinv is None:
        if sparse.issparse(lvl.A) and lvl.A.format == 'csr':
            blocksize = 1
        elif sparse.issparse(lvl.A) and lvl.A.format == 'bsr':
            blocksize = lvl.A.blocksize[0]
    elif blocksize is None:
        blocksize = Dinv.shape[1]
    if blocksize == 1:
        smoother = setup_jacobi(lvl, iterations=iterations, omega=omega,
                                withrho=withrho)
        update_wrapper(smoother, relaxation.block_jacobi)  # set __name__
        return smoother
    if Dinv is None:
        Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)
    if withrho:
        omega = omega / rho_block_D_inv_A(lvl.A, Dinv)
    smoother = partial(relaxation.block_jacobi, iterations=iterations, omega=omega,
                       Dinv=Dinv, blocksize=blocksize)
    update_wrapper(smoother, relaxation.block_jacobi)  # set __name__
    return smoother


def setup_block_gauss_seidel(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP,
                             Dinv=None, blocksize=None):
    """Block Gauss-Seidel relaxation smoother for level `lvl`."""
    if blocksize is None and Dinv is None:
        if sparse.issparse(lvl.A) and lvl.A.format == 'csr':
            blocksize = 1
        elif sparse.issparse(lvl.A) and lvl.A.format == 'bsr':
            blocksize = lvl.A.blocksize[0]
    elif blocksize is None:
        blocksize = Dinv.shape[1]
    if blocksize == 1:
        smoother = setup_gauss_seidel(lvl, iterations=iterations, sweep=sweep)
        update_wrapper(smoother, relaxation.block_gauss_seidel)
        return smoother
    if Dinv is None:
        Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)
    smoother = partial(relaxation.block_gauss_seidel, iterations=iterations,
                       Dinv=Dinv, blocksize=blocksize, sweep=sweep)
    update_wrapper(smoother, relaxation.block_gauss_seidel)  # set __name__
    return smoother


def setup_richardson(lvl, iterations=DEFAULT_NITER, omega=1.0):
    """Richardson polynomial smoother for level `lvl`."""
    omega = omega / approximate_spectral_radius(lvl.A)

    def richardson(A, x, b):
        relaxation.polynomial(A, x, b, coefficients=[omega], iterations=iterations)
    return richardson


def setup_sor(lvl, omega=0.5, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP):
    """SOR relaxation smoother for level `lvl`."""
    smoother = partial(relaxation.sor, iterations=iterations, omega=omega, sweep=sweep)
    update_wrapper(smoother, relaxation.sor)  # set __name__
    return smoother


def setup_chebyshev(lvl, lower_bound=1.0 / 30.0, upper_bound=1.1, degree=3,
                    iterations=DEFAULT_NITER):
    """Chebyshev polynomial smoother for level `lvl`."""
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]

    def chebyshev(A, x, b):
        relaxation.polynomial(A, x, b, coefficients=coefficients,
                              iterations=iterations)
    return chebyshev


def setup_jacobi_ne(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=True):
    """Cimmino (NE Jacobi) smoother for level `lvl`."""
    matrix_asformat(lvl, 'A', 'csr')
    if withrho:
        omega = omega / rho_D_inv_A(lvl.Acsr) ** 2

    def smoother(A, x, b):
        relaxation.jacobi_ne(lvl.Acsr, x, b, iterations=iterations, omega=omega)
    update_wrapper(smoother, relaxation.jacobi_ne)  # set __name__
    return smoother


def setup_gauss_seidel_ne(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP,
                          omega=1.0):
    """NE Gauss-Seidel smoother for level `lvl`."""
    matrix_asformat(lvl, 'A', 'csr')

    def smoother(A, x, b):
        relaxation.gauss_seidel_ne(lvl.Acsr, x, b, iterations=iterations,
                                   sweep=sweep, omega=omega)
    update_wrapper(smoother, relaxation.gauss_seidel_ne)  # set __name__
    return smoother


def setup_gauss_seidel_nr(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP,
                          omega=1.0):
    """NR Gauss-Seidel smoother for level `lvl`."""
    matrix_asformat(lvl, 'A', 'csc')

    def smoother(A, x, b):
        relaxation.gauss_seidel_nr(lvl.Acsc, x, b, iterations=iterations,
                                   sweep=sweep, omega=omega)
    update_wrapper(smoother, relaxation.gauss_seidel_nr)  # set __name__
    return smoother


def setup_gmres(lvl, tol=1e-12, maxiter=DEFAULT_NITER, restart=None, M=None,
                callback=None, residuals=None):
    """GMRES Krylov smoother for level `lvl`."""
    def smoother(A, x, b):
        x[:] = gmres(A, b, x0=x, tol=tol, maxiter=maxiter, restart=restart, M=M,
                     callback=callback, residuals=residuals)[0].reshape(x.shape)
    update_wrapper(smoother, gmres)  # set __name__
    return smoother


def setup_cg(lvl, tol=1e-12, maxiter=DEFAULT_NITER, M=None,
             callback=None, residuals=None):
    """CG Krylov smoother for level `lvl`."""
    def smoother(A, x, b):
        x[:] = cg(A, b, x0=x, tol=tol, maxiter=maxiter, M=M,
                  callback=callback, residuals=residuals)[0].reshape(x.shape)
    update_wrapper(smoother, cg)  # set __name__
    return smoother


def setup_cgne(lvl, tol=1e-12, maxiter=DEFAULT_NITER, M=None,
               callback=None, residuals=None):
    """CGNE Krylov smoother for level `lvl`."""
    def smoother(A, x, b):
        x[:] = cgne(A, b, x0=x, tol=tol, maxiter=maxiter, M=M,
                    callback=callback, residuals=residuals)[0].reshape(x.shape)
    update_wrapper(smoother, cgne)  # set __name__
    return smoother


def setup_cgnr(lvl, tol=1e-12, maxiter=DEFAULT_NITER, M=None,
               callback=None, residuals=None):
    """CGNR Krylov smoother for level `lvl`."""
    def smoother(A, x, b):
        x[:] = cgnr(A, b, x0=x, tol=tol, maxiter=maxiter, M=M,
                    callback=callback, residuals=residuals)[0].reshape(x.shape)
    update_wrapper(smoother, cgnr)  # set __name__
    return smoother


def setup_none(lvl):
    """No relaxation."""
    def none(A, x, b):
        pass
    return none  # set __name__ none


def _setup_call(fn):
    """Look up the `setup_*` function that builds smoother `fn`."""
    setup_register = {
        'gauss_seidel':           setup_gauss_seidel,
        'jacobi':                 setup_jacobi,
        'block_jacobi':           setup_block_jacobi,
        'block_gauss_seidel':     setup_block_gauss_seidel,
        'richardson':             setup_richardson,
        'sor':                    setup_sor,
        'chebyshev':              setup_chebyshev,
        'jacobi_ne':              setup_jacobi_ne,
        'gauss_seidel_ne':        setup_gauss_seidel_ne,
        'gauss_seidel_nr':        setup_gauss_seidel_nr,
        'gmres':                  setup_gmres,
        'cg':                     setup_cg,
        'cgne':                   setup_cgne,
        'cgnr':                   setup_cgnr,
        'none':                   setup_none,
    }
    if fn is None:
        fn = 'none'
    if not isinstance(fn, str):
        raise ValueError(f'Input function must be a string or None: fn={fn}')
    if fn not in setup_register:
        raise ValueError(f'Function {fn} does not have a setup')
    return setup_register[fn]


def rebuild_smoother(lvl):
    """Rebuild the pre- and postsmoother of a level after its matrix changed."""
    try:
        fn1 = lvl.presmoother.__name__
        fn2 = lvl.postsmoother.__name__
    except AttributeError as exc:
        raise AttributeError('The pre/post smoothers need to be functions.') from exc
    setup_presmoother = _setup_call(fn1)
    lvl.presmoother = setup_presmoother(lvl)
    setup_postsmoother = _setup_call(fn2)
    lvl.postsmoother = setup_postsmoother(lvl)


__all__ = ['change_smoothers', 'rho_D_inv_A', 'rho_block_D_inv_A',
           'matrix_asformat', 'setup_gauss_seidel', 'setup_jacobi',
           'setup_block_jacobi', 'setup_block_gauss_seidel', 'setup_richardson',
           'setup_sor', 'setup_chebyshev', 'setup_jacobi_ne',
           'setup_gauss_seidel_ne', 'setup_gauss_seidel_nr', 'setup_gmres',
           'setup_cg', 'setup_cgne', 'setup_cgnr', 'setup_none',
           'rebuild_smoother', 'DEFAULT_SWEEP', 'DEFAULT_NITER']
