"""Relaxation methods.  Mirrors `pyamg.relaxation.relaxation`.

Every sweep below is the ported Mojo kernel from `amg_core/relaxation.h`; the
Python around it is upstream's, with the same names, argument order and
defaults.
"""

import numpy as np
from scipy import sparse

from .._lib import addr, f64, i64, lib
from ..util.linalg import norm
from ..util.utils import get_block_diag, get_diagonal, to_type


def make_system(A, x, b, formats=None):
    """Return A in a sparse format and x, b as contiguous 1-D float64 arrays.

    Complex input is rejected rather than cast: the kernels behind these
    sweeps are the real instantiations of the upstream templates.
    """
    if np.iscomplexobj(A) or np.iscomplexobj(x) or np.iscomplexobj(b):
        raise TypeError('complex input is not part of this port')
    if sparse.issparse(A):
        A = A.astype(np.float64)
        if formats is not None and A.format not in formats:
            A = A.asformat(formats[0])
    else:
        A = np.asarray(A, dtype=np.float64)

    if x is None:
        x = np.zeros(np.shape(b), dtype=np.float64)
    # a view, not a copy: the kernels write the relaxed solution in place
    x = np.ravel(np.asarray(x, dtype=np.float64))

    b = np.ravel(np.asarray(b, dtype=np.float64))

    return A, x, b


def _type_prep(upcast_type, varlist):
    """Convert scalars to 0-d/1-element arrays of the working type."""
    return to_type(upcast_type, varlist)


def _csr(A):
    A = A.tocsr()
    A.sort_indices()
    A.data = np.asarray(A.data, dtype=np.float64)
    return A


def _bsr(A):
    A = A.tobsr(blocksize=(A.blocksize[0], A.blocksize[1]))
    A.sort_indices()
    A.data = np.asarray(A.data, dtype=np.float64)
    return A


def _csc(A):
    A = A.tocsc()
    A.sort_indices()
    A.data = np.asarray(A.data, dtype=np.float64)
    return A


def _buffers(A, fmt='csr'):
    """CSR/BSR arrays as the int64/float64 buffers the kernels take.

    The arrays come back next to their addresses and must stay referenced for
    the whole kernel call: a temporary handed straight to `addr` would be
    freed before the kernel reads it.
    """
    A = _csr(A) if fmt == 'csr' else (_bsr(A) if fmt == 'bsr' else _csc(A))
    Ap = i64(A.indptr)
    Aj = i64(A.indices)
    Ax = f64(A.data)
    return A, (Ap, Aj, Ax), (addr(Ap), addr(Aj), addr(Ax))


def sor(A, x, b, omega, iterations=1, sweep='forward'):
    """Successive over-relaxation (SOR) relaxation on Ax = b.

    Parameters
    ----------
    A : csr_array, bsr_array
        Sparse matrix in CSR or BSR format.
    x : array
        Approximate solution.
    b : array
        Right-hand side.
    omega : scalar
        Relaxation parameter.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    for _i in range(iterations):
        gauss_seidel(A, x, b, iterations=1, sweep=sweep, omega=omega)


def gauss_seidel(A, x, b, iterations=1, sweep='forward', omega=1.0):
    """Gauss-Seidel relaxation on Ax = b.

    Parameters
    ----------
    A : csr_array, bsr_array
        Sparse matrix in CSR or BSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.
    omega : scalar
        Relaxation parameter; 1.0 gives plain Gauss-Seidel, anything else SOR.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    # bind the caller's buffers once: a temporary handed straight to `addr`
    # would be freed before the kernel reads it
    bx = f64(x)
    bb = f64(b)
    if sparse.issparse(A) and A.format == 'csr':
        blocksize = 1
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        blocksize = R
    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x) / blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x) / blocksize) - 1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, sweep='backward')
        return
    else:
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')
    if sparse.issparse(A) and A.format == 'csr':
        A, _buf, (aAp, aAj, aAx) = _buffers(A)
        if omega != 1.0:
            for _iter in range(iterations):
                lib().mpa_sor_gauss_seidel(aAp, aAj, aAx, addr(bx),
                                           addr(bb), row_start, row_stop,
                                           row_step, omega)
        else:
            for _iter in range(iterations):
                lib().mpa_gauss_seidel(aAp, aAj, aAx, addr(bx), addr(bb),
                                       row_start, row_stop, row_step)
    else:
        rsum = np.empty(blocksize, dtype=np.float64)
        Axloc = np.empty(blocksize, dtype=np.float64)
        A, _buf, (aAp, aAj, aAx) = _buffers(A, 'bsr')
        for _iter in range(iterations):
            lib().mpa_bsr_gauss_seidel(aAp, aAj, aAx, addr(bx), addr(bb),
                                       row_start, row_stop, row_step, blocksize,
                                       addr(rsum), addr(Axloc))


def jacobi(A, x, b, iterations=1, omega=1.0):
    """Weighted Jacobi relaxation on Ax = b.

    Parameters
    ----------
    A : csr_array
        Sparse matrix in CSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    omega : scalar
        Damping parameter.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(A.shape[0])
    if (row_stop - row_start) * row_step <= 0:  # no work to do
        return
    temp = np.empty_like(x)
    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = [float(v) for v in _type_prep(A.dtype, [omega])]
    if A.format == 'csr':
        A, _buf, (aAp, aAj, aAx) = _buffers(A)
        for _iter in range(iterations):
            lib().mpa_jacobi(aAp, aAj, aAx, addr(bx), addr(bb),
                             addr(temp), row_start, row_stop, row_step, omega)
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        row_start = int(row_start / R)
        row_stop = int(row_stop / R)
        rsum = np.empty(R, dtype=np.float64)
        Axloc = np.empty(R, dtype=np.float64)
        A, _buf, (aAp, aAj, aAx) = _buffers(A, 'bsr')
        for _iter in range(iterations):
            lib().mpa_bsr_jacobi(aAp, aAj, aAx, addr(bx), addr(bb),
                                 addr(temp), row_start, row_stop, row_step,
                                 R, omega, addr(rsum), addr(Axloc))


def block_jacobi(A, x, b, Dinv=None, blocksize=1, iterations=1, omega=1.0):
    """Block weighted Jacobi relaxation on Ax = b.

    Parameters
    ----------
    A : bsr_array
        Sparse matrix in BSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    Dinv : array
        Inverses of the diagonal blocks of A; computed from A if None.
    blocksize : int
        Dimension of the square blocks in A.
    iterations : int
        Number of relaxation steps.
    omega : scalar
        Damping parameter.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    A = A.tobsr(blocksize=(blocksize, blocksize))
    if Dinv is None:
        Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
    elif Dinv.shape[0] != int(A.shape[0] / blocksize):
        raise ValueError('Dinv and A have incompatible dimensions')
    elif (Dinv.shape[1] != blocksize) or (Dinv.shape[2] != blocksize):
        raise ValueError('Dinv and blocksize are incompatible')
    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(int(A.shape[0] / blocksize))
    if (row_stop - row_start) * row_step <= 0:  # no work to do
        return
    temp = np.empty_like(x)
    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = [float(v) for v in _type_prep(A.dtype, [omega])]
    Dinv = np.ravel(np.asarray(Dinv, dtype=np.float64))
    rsum = np.empty(blocksize, dtype=np.float64)
    v = np.empty(blocksize, dtype=np.float64)
    A, _buf, (aAp, aAj, aAx) = _buffers(A, 'bsr')
    for _iter in range(iterations):
        lib().mpa_block_jacobi(aAp, aAj, aAx, addr(bx), addr(bb),
                               addr(Dinv), addr(temp), row_start, row_stop,
                               row_step, omega, blocksize, addr(rsum), addr(v))


def block_gauss_seidel(A, x, b, iterations=1, sweep='forward', blocksize=1, Dinv=None):
    """Block Gauss-Seidel relaxation on Ax = b.

    Parameters
    ----------
    A : bsr_array
        Sparse matrix in BSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.
    blocksize : int
        Dimension of the square blocks in A.
    Dinv : array
        Inverses of the diagonal blocks of A; computed from A if None.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    A = A.tobsr(blocksize=(blocksize, blocksize))
    if Dinv is None:
        Dinv = get_block_diag(A, blocksize=blocksize, inv_flag=True)
    elif Dinv.shape[0] != int(A.shape[0] / blocksize):
        raise ValueError('Dinv and A have incompatible dimensions')
    elif (Dinv.shape[1] != blocksize) or (Dinv.shape[2] != blocksize):
        raise ValueError('Dinv and blocksize are incompatible')
    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x) / blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x) / blocksize) - 1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            block_gauss_seidel(A, x, b, iterations=1, sweep='forward',
                               blocksize=blocksize, Dinv=Dinv)
            block_gauss_seidel(A, x, b, iterations=1, sweep='backward',
                               blocksize=blocksize, Dinv=Dinv)
        return
    else:
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')
    Dinv = np.ravel(np.asarray(Dinv, dtype=np.float64))
    rsum = np.empty(blocksize, dtype=np.float64)
    v = np.empty(blocksize, dtype=np.float64)
    A, _buf, (aAp, aAj, aAx) = _buffers(A, 'bsr')
    for _iter in range(iterations):
        lib().mpa_block_gauss_seidel(aAp, aAj, aAx, addr(bx), addr(bb),
                                     addr(Dinv), row_start, row_stop, row_step,
                                     blocksize, addr(rsum), addr(v))


def polynomial(A, x, b, coefficients, iterations=1):
    """Polynomial smoother on Ax = b.

    Parameters
    ----------
    A : csr_array, bsr_array
        Sparse matrix.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    coefficients : array
        Polynomial coefficients in descending order, p[0] is the constant.
    iterations : int
        Number of polynomial applications.

    """
    A, x, b = make_system(A, x, b, formats=None)
    bx = f64(x)
    bb = f64(b)
    for _i in range(iterations):
        if norm(x) == 0:
            residual = b
        else:
            residual = b - A @ x
        h = coefficients[0] * residual
        for c in coefficients[1:]:
            h = c * residual + A @ h
        x += h


def gauss_seidel_indexed(A, x, b, indices, iterations=1, sweep='forward'):
    """Gauss-Seidel relaxation restricted to a set of row indices.

    Parameters
    ----------
    A : csr_array
        Sparse matrix in CSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    indices : array
        Row indices to sweep.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.

    """
    A, x, b = make_system(A, x, b, formats=['csr'])
    bx = f64(x)
    bb = f64(b)
    indices = np.asarray(indices, dtype=np.int64)
    if sweep == 'forward':
        row_start, row_stop, row_step = 0, len(indices), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = len(indices) - 1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel_indexed(A, x, b, indices, iterations=1, sweep='forward')
            gauss_seidel_indexed(A, x, b, indices, iterations=1, sweep='backward')
        return
    else:
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')
    A, _buf, (aAp, aAj, aAx) = _buffers(A)
    Id = i64(indices)
    for _iter in range(iterations):
        lib().mpa_gauss_seidel_indexed(aAp, aAj, aAx, addr(bx), addr(bb),
                                       addr(Id), row_start, row_stop, row_step)


def jacobi_indexed(A, x, b, indices, iterations=1, omega=1.0):
    """Weighted Jacobi relaxation restricted to a set of row indices.

    Parameters
    ----------
    A : csr_array
        Sparse matrix in CSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    indices : array
        Row indices to relax.
    iterations : int
        Number of relaxation steps.
    omega : scalar
        Damping parameter.

    """
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])
    bx = f64(x)
    bb = f64(b)
    indices = np.asarray(indices, dtype=np.int64)
    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = [float(v) for v in _type_prep(A.dtype, [omega])]
    if sparse.issparse(A) and A.format == 'csr':
        A, _buf, (aAp, aAj, aAx) = _buffers(A)
        Id = i64(indices)
        temp = np.empty(len(x), dtype=np.float64)
        for _iter in range(iterations):
            lib().mpa_jacobi_indexed(aAp, aAj, aAx, addr(bx), addr(bb),
                                     len(x), addr(Id), len(indices),
                                     omega, addr(temp))
    else:
        R, C = A.blocksize
        if R != C:
            raise ValueError('BSR blocks must be square')
        if indices.max() > A.shape[0] / R - 1:
            raise ValueError('Indices must range from 0, ..., numrows/blocksize - 1)')
        rsum = np.empty(R, dtype=np.float64)
        Axloc = np.empty(R, dtype=np.float64)
        temp = np.empty(len(x), dtype=np.float64)
        A, _buf, (aAp, aAj, aAx) = _buffers(A, 'bsr')
        Id = i64(indices)
        for _iter in range(iterations):
            lib().mpa_bsr_jacobi_indexed(aAp, aAj, aAx, addr(bx), addr(bb),
                                         len(x), addr(Id), len(indices), R, omega,
                                         addr(rsum), addr(Axloc), addr(temp))


def jacobi_ne(A, x, b, iterations=1, omega=1.0):
    """Cimmino relaxation (weighted Jacobi on A.H A x = A.H b).

    Parameters
    ----------
    A : csr_array
        Sparse matrix in CSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    omega : scalar
        Damping parameter.

    """
    A, x, b = make_system(A, x, b, formats=['csr'])
    bx = f64(x)
    bb = f64(b)
    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(A.shape[0])
    temp = np.zeros_like(x)
    # Dinv for A*A.H
    Dinv = get_diagonal(A, norm_eq=2, inv=True)
    # Create uniform type, convert possibly complex scalars to length 1 arrays
    [omega] = [float(v) for v in _type_prep(A.dtype, [omega])]
    A, _buf, (aAp, aAj, aAx) = _buffers(A)
    for _i in range(iterations):
        delta = f64(np.ravel(b - A @ x) * np.ravel(Dinv))
        lib().mpa_jacobi_ne(aAp, aAj, aAx, addr(bx), addr(bb),
                            addr(delta), addr(temp), row_start,
                            row_stop, row_step, omega)


def gauss_seidel_ne(A, x, b, iterations=1, sweep='forward', omega=1.0, Dinv=None):
    """Gauss-Seidel relaxation on A A.H y = b with x = A.H y (additive form).

    Parameters
    ----------
    A : csr_array
        Sparse matrix in CSR format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.
    omega : scalar
        Relaxation parameter.
    Dinv : array
        Inverse of diag(A A.H); computed from A if None.

    """
    A, x, b = make_system(A, x, b, formats=['csr'])
    bx = f64(x)
    bb = f64(b)
    # Dinv for A*A.H
    if Dinv is None:
        Dinv = np.ravel(get_diagonal(A, norm_eq=2, inv=True))
    if sweep == 'forward':
        row_start, row_stop, row_step = 0, len(x), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = len(x) - 1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel_ne(A, x, b, iterations=1, sweep='forward',
                            omega=omega, Dinv=Dinv)
            gauss_seidel_ne(A, x, b, iterations=1, sweep='backward',
                            omega=omega, Dinv=Dinv)
        return
    else:
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')
    Dinv = f64(np.ravel(np.asarray(Dinv, dtype=np.float64)))
    A, _buf, (aAp, aAj, aAx) = _buffers(A)
    for _i in range(iterations):
        lib().mpa_gauss_seidel_ne(aAp, aAj, aAx, addr(bx), addr(bb),
                                  row_start, row_stop, row_step, addr(Dinv), omega)


def gauss_seidel_nr(A, x, b, iterations=1, sweep='forward', omega=1.0, Dinv=None):
    """Gauss-Seidel relaxation on A.H A x = A.H b (additive form).

    Parameters
    ----------
    A : csc_array
        Sparse matrix in CSC format.
    x : array
        Approximate solution, modified in place.
    b : array
        Right-hand side.
    iterations : int
        Number of relaxation steps.
    sweep : {'forward', 'backward', 'symmetric'}
        Sweep direction.
    omega : scalar
        Relaxation parameter.
    Dinv : array
        Inverse of diag(A.H A); computed from A if None.

    """
    A, x, b = make_system(A, x, b, formats=['csc'])
    bx = f64(x)
    bb = f64(b)
    # Dinv for A.H*A
    if Dinv is None:
        Dinv = np.ravel(get_diagonal(A, norm_eq=1, inv=True))
    if sweep == 'forward':
        col_start, col_stop, col_step = 0, len(x), 1
    elif sweep == 'backward':
        col_start, col_stop, col_step = len(x) - 1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel_nr(A, x, b, iterations=1, sweep='forward',
                            omega=omega, Dinv=Dinv)
            gauss_seidel_nr(A, x, b, iterations=1, sweep='backward',
                            omega=omega, Dinv=Dinv)
        return
    else:
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')
    # Calculate initial residual
    A, _buf, (aAp, aAj, aAx) = _buffers(A, 'csc')
    Dinv = f64(np.ravel(np.asarray(Dinv, dtype=np.float64)))
    r = f64(b - A @ x)
    for _i in range(iterations):
        lib().mpa_gauss_seidel_nr(aAp, aAj, aAx, addr(bx), addr(r),
                                  col_start, col_stop, col_step, addr(Dinv), omega)


__all__ = ['make_system', 'sor', 'gauss_seidel', 'jacobi', 'block_jacobi',
           'block_gauss_seidel', 'polynomial', 'gauss_seidel_indexed',
           'jacobi_indexed', 'jacobi_ne', 'gauss_seidel_ne', 'gauss_seidel_nr']
