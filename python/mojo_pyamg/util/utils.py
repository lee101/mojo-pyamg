"""Utilities.  Mirrors `pyamg.util.utils` for the covered subset."""

from warnings import warn

import numpy as np
from scipy.sparse import (bsr_array, csr_array, issparse, eye_array, sparray)
from scipy.sparse.linalg._isolve.utils import make_system as _make_system

from .._lib import addr, f64, i64, lib


def asfptype(A):
    """Convert to the smallest compatible floating point dtype if needed.

    Complex input is rejected: the Mojo kernels are the real instantiations
    of the upstream templates, and casting complex to float would quietly drop
    the imaginary part.
    """
    if A.dtype.kind == 'c':
        raise TypeError(
            f'complex matrices are not part of this port: got dtype {A.dtype}')
    if A.dtype.char in 'fdFD':
        return A
    for fp_type in 'fdFD':
        if A.dtype <= np.dtype(fp_type):
            return A.astype(fp_type)
    raise TypeError(f'cannot upcast [{A.dtype}] to a floating point format')


def get_blocksize(A):
    """Number of dofs per node: the BSR blocksize, or 1 for point matrices."""
    if issparse(A) and A.format == 'bsr':
        return A.blocksize[0]
    return 1


def to_type(upcast_type, varlist):
    """Cast every entry of `varlist` to `upcast_type`."""
    nv = len(varlist)
    for i in range(nv):
        # convert scalars to complex
        if np.isscalar(varlist[i]):
            varlist[i] = np.array([varlist[i]], upcast_type)[0]
        else:
            # convert sparse and dense mats to complex
            try:
                if varlist[i].dtype != upcast_type:
                    varlist[i] = varlist[i].astype(upcast_type)
            except AttributeError:
                warn('Failed to cast in to_type')
    return varlist


def make_system(A, x, b, formats=None):
    """Return A in a sparse format, x and b as contiguous 1-D float64 vectors.

    This is `pyamg.relaxation.relaxation.make_system`, not the krylov one.
    """
    if issparse(A):
        A = asfptype(A)
        if formats is not None and A.format not in formats:
            A = A.asformat(formats[0])
    else:
        A = np.asarray(A)

    if x is None:
        x = np.zeros(np.shape(b), dtype=np.float64)
    # a view, not a copy: the kernels write the relaxed solution in place
    x = np.ravel(np.asarray(x, dtype=np.float64))

    b = np.ravel(np.asarray(b, dtype=np.float64))

    return A, x, b


def _scipy_make_system(A, M, x0, b):
    """Wrap scipy's krylov `make_system` so the 5-tuple form is never returned."""
    out = _make_system(A, M, x0, b)
    return out if len(out) == 4 else (out[0], out[1], out[2], out[3])


# The krylov solvers call `make_system(A, M, x0, b)`, exactly as upstream does;
# upstream takes this from scipy.sparse.linalg._isolve.utils, so we do too.
make_system = _scipy_make_system

def scale_rows(A, v, copy=True):
    """Scale the rows of A by v: A[i,:] *= v[i]."""
    v = np.ravel(v)
    M, N = A.shape
    if not issparse(A):
        raise ValueError('scale rows needs a sparse matrix')
    if M != len(v):
        raise ValueError('scale vector has incompatible shape')
    if copy:
        A = A.copy()
        A.data = np.asarray(A.data, dtype=np.float64)
    else:
        v = np.asarray(v, dtype=np.float64)

    if A.format == 'csr':
        out = A.tocsc()
        Ap = i64(out.indptr)
        Aj = i64(out.indices)
        Ax = f64(out.data)
        Xv = f64(v)
        lib().mpa_csc_scale_rows(out.shape[1], addr(Ap), addr(Aj), addr(Ax),
                                 addr(Xv))
        out.eliminate_zeros()
        return out.tocsr()

    if A.format == 'csc':
        Ap = i64(A.indptr)
        Aj = i64(A.indices)
        Ax = f64(A.data)
        Xv = f64(v)
        lib().mpa_csc_scale_rows(N, addr(Ap), addr(Aj), addr(Ax), addr(Xv))
        return A

    return scale_rows(csr_array(A), v).asformat(A.format)


def get_diagonal(A, norm_eq=False, inv=False):
    """Diagonal of A (or of A.H A / A A.H when `norm_eq`), optionally inverted."""
    if not issparse(A) or A.format not in ('bsr', 'csc', 'csr'):
        warn('Implicit conversion to sparse matrix')
        A = csr_array(A)
    # critical to sort the indices of A
    A.sort_indices()
    if norm_eq == 1:
        # This transpose involves almost no work, use csr data structures as
        # csc, or vice versa
        At = A.T
        D = (At.multiply(At.conjugate())) @ np.ones((At.shape[0],))
    elif norm_eq == 2:
        D = (A.multiply(A.conjugate())) @ np.ones((A.shape[0],))
    else:
        D = A.diagonal()
    if inv:
        Dinv = np.zeros_like(D)
        mask = D != 0.0
        Dinv[mask] = 1.0 / D[mask]
        return Dinv
    return D


def get_block_diag(A, blocksize, inv_flag=True):
    """Diagonal blocks of a BSR matrix, optionally inverted block by block."""
    if not issparse(A):
        raise TypeError('Expected sparse matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Expected square matrix')
    if np.mod(A.shape[0], blocksize) != 0:
        raise ValueError('blocksize and A.shape must be compatible')
    # If the block diagonal of A already exists, return that
    if hasattr(A, 'block_D_inv') and inv_flag:
        if (A.block_D_inv.shape[1] == blocksize) and \
           (A.block_D_inv.shape[2] == blocksize) and \
           (A.block_D_inv.shape[0] == int(A.shape[0] / blocksize)):
            return A.block_D_inv
    elif hasattr(A, 'block_D') and (not inv_flag):
        if (A.block_D.shape[1] == blocksize) and \
           (A.block_D.shape[2] == blocksize) and \
           (A.block_D.shape[0] == int(A.shape[0] / blocksize)):
            return A.block_D
    # Convert to BSR
    if not issparse(A) or A.format != 'bsr':
        A = bsr_array(A, blocksize=(blocksize, blocksize))
    if A.blocksize != (blocksize, blocksize):
        A = A.tobsr(blocksize=(blocksize, blocksize))
    A = asfptype(A)
    block_diag = np.zeros((int(A.shape[0] / blocksize), blocksize, blocksize),
                          dtype=np.float64)
    AAIJ = (np.arange(1, A.indices.shape[0] + 1), A.indices, A.indptr)
    shape = (int(A.shape[0] / blocksize), int(A.shape[0] / blocksize))
    diag_entries = csr_array(AAIJ, shape=shape).diagonal()
    diag_entries -= 1
    nonzero_mask = diag_entries != -1
    diag_entries = diag_entries[nonzero_mask]
    if diag_entries.shape != (0,):
        block_diag[nonzero_mask, :, :] = A.data[diag_entries, :, :]
    if inv_flag:
        block_diag = _pinv_blocks(block_diag)
        A.block_D_inv = block_diag
    else:
        A.block_D = block_diag
    return block_diag


def _pinv_blocks(blocks):
    """Moore-Penrose pseudo inverse of a stack of dense square blocks."""
    out = np.zeros_like(blocks)
    for i in range(blocks.shape[0]):
        out[i] = np.linalg.pinv(blocks[i])
    return out


def amalgamate(A, blocksize):
    """Block-diagonal boolean operator with the sparsity pattern of A."""
    if blocksize == 1:
        return A
    if np.mod(A.shape[0], blocksize) != 0:
        raise ValueError('Incompatible blocksize')
    A = A.tobsr(blocksize=(blocksize, blocksize))
    A.sort_indices()
    subI = (np.ones(A.indices.shape), A.indices, A.indptr)
    shape = (int(A.shape[0] / A.blocksize[0]), int(A.shape[1] / A.blocksize[1]))
    return csr_array(subI, shape=shape)


def unamal(A, rows_per_block, cols_per_block):
    """Every entry of A expanded into an all-ones block."""
    data = np.ones((A.indices.shape[0], rows_per_block, cols_per_block))
    blockI = (data, A.indices, A.indptr)
    shape = (rows_per_block * A.shape[0], cols_per_block * A.shape[1])
    return bsr_array(blockI, shape=shape)


def levelize_strength_or_aggregation(to_levelize, max_levels, max_coarse):
    """Turn a user strength/aggregation option into a per-level list."""
    if isinstance(to_levelize, tuple):
        if to_levelize[0] == 'predefined':
            to_levelize = [to_levelize]
            max_levels = 2
            max_coarse = 0
        else:
            to_levelize = [to_levelize for i in range(max_levels - 1)]
    elif isinstance(to_levelize, str):
        if to_levelize == 'predefined':
            raise ValueError('predefined to_levelize requires a user-provided '
                             'CSR matrix representing strength or aggregation '
                             'i.e., ("predefined", {"C" : CSR_MAT}).')
        to_levelize = [to_levelize for i in range(max_levels - 1)]
    elif isinstance(to_levelize, list):
        if isinstance(to_levelize[-1], tuple) and \
           (to_levelize[-1][0] == 'predefined'):
            # to_levelize is a list that ends with a predefined operator
            max_levels = len(to_levelize) + 1
            max_coarse = 0
        elif len(to_levelize) < max_levels - 1:
            mlz = max_levels - 1 - len(to_levelize)
            toext = [to_levelize[-1] for i in range(mlz)]
            to_levelize.extend(toext)
    elif to_levelize is None:
        to_levelize = [(None, {}) for i in range(max_levels - 1)]
    else:
        raise ValueError('invalid to_levelize')
    return max_levels, max_coarse, to_levelize


def levelize_smooth_or_improve_candidates(to_levelize, max_levels):
    """Turn a user smooth/improve_candidates option into a per-level list."""
    if isinstance(to_levelize, tuple):
        if isinstance(to_levelize[0], tuple):
            to_levelize = list(to_levelize)
    if isinstance(to_levelize, (str, tuple)):
        to_levelize = [to_levelize for i in range(max_levels)]
    elif isinstance(to_levelize, list):
        if len(to_levelize) < max_levels:
            mlz = max_levels - len(to_levelize)
            toext = [to_levelize[-1] for i in range(mlz)]
            to_levelize.extend(toext)
    elif to_levelize is None:
        to_levelize = [(None, {}) for i in range(max_levels)]
    return to_levelize


def filter_matrix_rows(A, theta, diagonal=False, lump=False):
    """Drop entries below a drop tolerance: |A_ij| < theta * |A_ii|."""
    if not issparse(A):
        raise ValueError('Sparse matrix input needed')
    blocksize = 1
    if A.format == 'bsr':
        blocksize = A.blocksize
    Aformat = A.format
    A = A.tocsr()
    if (theta < 0) or (theta >= 1.0):
        raise ValueError('theta must be in [0,1)')
    # Filter by diagonal entry, A_ij = 0 if |A_ij| < theta*|A_ii|
    if diagonal:
        A = A.copy()
        Ap = i64(A.indptr)
        Aj = i64(A.indices)
        Ax = f64(A.data)
        lib().mpa_filter_matrix_rows(A.shape[0], theta, addr(Ap), addr(Aj),
                                     addr(Ax), 1 if lump else 0)
        A.eliminate_zeros()
        if Aformat == 'bsr':
            A = A.tobsr(blocksize=blocksize)
        return None  # inplace
    # Apply drop-tolerance to each row of A.  We apply the drop-tolerance with
    # classical_strength_of_connection_abs(), which ignores diagonal entries,
    # thus necessitating the trick where we add A.shape[0] to each of the row
    # indices
    A_filter = A.copy()
    A.indices += A.shape[0]
    A_filter.indices += A_filter.shape[0]
    # classical_strength_of_connection takes an absolute value internally
    Ap = i64(A.indptr)
    Aj = i64(A.indices)
    Ax = f64(A.data)
    Fp = i64(A_filter.indptr)
    Fj = i64(A_filter.indices)
    Fx = f64(A_filter.data)
    lib().mpa_classical_strength_of_connection_abs(
        A.shape[0], theta, addr(Ap), addr(Aj), addr(Ax), addr(Fp), addr(Fj),
        addr(Fx))
    nnz = int(A_filter.indptr[-1])
    A_filter.indices[:nnz] -= A_filter.shape[0]
    A_filter = csr_array((A_filter.data[:nnz], A_filter.indices[:nnz],
                          A_filter.indptr), shape=A_filter.shape)
    if Aformat == 'bsr':
        A_filter = A_filter.tobsr(blocksize)
    else:
        A_filter = A_filter.asformat(Aformat)
    A.indices -= A.shape[0]
    return A_filter


def truncate_rows(A, nz_per_row):
    """Keep only the `nz_per_row` largest-magnitude entries of each row."""
    if not issparse(A):
        raise ValueError('Sparse matrix input needed')
    blocksize = 1
    if A.format == 'bsr':
        blocksize = A.blocksize
    if A.format == 'csr':
        A = A.copy()    # don't modify A in-place
    Aformat = A.format
    A = A.tocsr()
    nz_per_row = int(nz_per_row)
    Ap = i64(A.indptr)
    Aj = i64(A.indices)
    Ax = f64(A.data)
    lib().mpa_truncate_rows_csr(A.shape[0], nz_per_row, addr(Ap), addr(Aj),
                                addr(Ax))
    A.eliminate_zeros()
    if Aformat == 'bsr':
        A = A.tobsr(blocksize)
    else:
        A = A.asformat(Aformat)
    return A


def scale_block_inverse(A, blocksize):
    """Return (Dinv @ A, Dinv) where Dinv inverts the diagonal blocks of A."""
    if not issparse(A):
        raise TypeError('Expected sparse matrix')
    if A.shape[0] != A.shape[1]:
        raise ValueError('Expected square matrix')
    if np.mod(A.shape[0], blocksize) != 0:
        raise ValueError('blocksize and A.shape must be compatible')
    # Convert to BSR
    if A.format != 'bsr':
        A = bsr_array(A, blocksize=(blocksize, blocksize))
    if A.blocksize != (blocksize, blocksize):
        A = A.tobsr(blocksize=(blocksize, blocksize))
    # Get block diagonal inverse
    N_block = int(A.shape[0] / blocksize)
    Dinv = get_block_diag(A=A, blocksize=blocksize, inv_flag=True)
    scale = bsr_array((Dinv, np.arange(0, N_block), np.arange(0, N_block + 1)),
                      blocksize=[blocksize, blocksize], shape=A.shape)
    return scale @ A, scale


def compute_BtBinv(B, C):
    """BtBinv[i] = pinv(B_i.H B_i) for the sparsity pattern of supernode i."""
    if not issparse(C) or C.format not in ('csr', 'bsr'):
        raise TypeError('Expected bsr_array or csr_array for C')
    if C.shape[1] != B.shape[0]:
        raise TypeError('Expected matching dimensions such that C @ B')
    if C.format == 'bsr':
        cols_per_block = C.blocksize[1]
        rows_per_block = C.blocksize[0]
    else:
        cols_per_block = 1
        rows_per_block = 1
    Ncoarse = C.shape[1]
    Nfine = C.shape[0]
    NullDim = B.shape[1]
    Nnodes = int(Nfine / rows_per_block)

    BtBinv = np.zeros((Nnodes, NullDim, NullDim), dtype=np.float64)
    Cs = C.tocsr()
    for i in range(Nnodes):
        cols = Cs.indices[Cs.indptr[i]:Cs.indptr[i + 1]]
        blocks = [np.asarray(B[c * rows_per_block:(c + 1) * rows_per_block, :])
                  for c in cols]
        if not blocks:
            continue
        M = np.vstack(blocks)
        BtBinv[i] = np.linalg.pinv(M.T @ M)
    _ = Ncoarse
    return BtBinv


def eliminate_diag_dom_nodes(A, C, theta=1.02):
    """Remove diagonally dominant rows/columns of the strength matrix C."""
    # Find the diagonally dominant rows in A.
    A_abs = A.copy()
    A_abs.data = np.abs(A_abs.data)
    D_abs = get_diagonal(A_abs, norm_eq=0, inv=False)
    diag_dom_rows = (D_abs > (theta * (A_abs @ np.ones((A_abs.shape[0],),
                                                        dtype=np.float64) - D_abs)))
    # Account for BSR matrices and translate diag_dom_rows from dofs to nodes
    bsize = get_blocksize(A_abs)
    if bsize > 1:
        diag_dom_rows = np.array(diag_dom_rows, dtype=int)
        diag_dom_rows = diag_dom_rows.reshape(-1, bsize)
        diag_dom_rows = np.sum(diag_dom_rows, axis=1)
        diag_dom_rows = diag_dom_rows == bsize
    # Replace these rows/cols in C with rows/cols of the identity.
    Id = eye_array(C.shape[0], C.shape[1], format='csr')
    Id.data[diag_dom_rows] = 0.0
    C = Id @ C @ Id
    Id.data[diag_dom_rows] = 1.0
    Id.data[np.where(diag_dom_rows == 0)[0]] = 0.0
    C = C + Id
    del A_abs
    return C


def remove_diagonal(S):
    """S with the diagonal entries zeroed (in place)."""
    S = S.tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    return S


def scale_rows_by_largest_entry(S):
    """S with each row scaled so its largest magnitude entry is one."""
    S = scale_rows(S, 1.0 / maximum_row_value(S))
    return S


def maximum_row_value(A):
    """Row-wise maximum magnitude of a CSR matrix."""
    A = A.tocsr()
    x = np.empty(A.shape[0], dtype=np.float64)
    Ap = i64(A.indptr)
    Aj = i64(A.indices)
    Ax = f64(A.data)
    lib().mpa_maximum_row_value(A.shape[0], addr(x), addr(Ap), addr(Aj),
                                addr(Ax))
    return x


def upcast(*types):
    """The smallest dtype that holds all the given dtypes."""
    return np.result_type(*types)


def print_table(table, title='', delim='|', centering='center', col_padding=2,
                header=True, headerchar='-'):
    """Pretty-print a 2-D list of strings; kept for API parity with upstream."""
    def pad(s, width):
        if centering == 'right':
            return s.rjust(width)
        if centering == 'left':
            return s.ljust(width)
        return s.center(width)

    if header:
        table = [list(table[0])] + [list(row) for row in table[1:]]
    ncol = max(len(row) for row in table)
    widths = [0] * ncol
    for row in table:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(str(cell)))
    widths = [w + col_padding for w in widths]

    out = []
    if title:
        out.append(title)
        out.append(headerchar * (sum(widths) + ncol * (len(delim) - 1)))
    for i, row in enumerate(table):
        out.append(delim + delim.join(pad(str(c), widths[j])
                                      for j, c in enumerate(row)) + delim)
        if header and i == 0:
            out.append(delim + delim.join(headerchar * widths[j]
                                          for j in range(ncol)) + delim)
    return '\n'.join(out)


__all__ = [
    'asfptype', 'get_blocksize', 'to_type', 'make_system', 'scale_rows',
    'get_diagonal', 'get_block_diag', 'amalgamate', 'unamal',
    'levelize_strength_or_aggregation', 'levelize_smooth_or_improve_candidates',
    'filter_matrix_rows', 'truncate_rows', 'scale_block_inverse',
    'compute_BtBinv', 'eliminate_diag_dom_nodes', 'remove_diagonal',
    'scale_rows_by_largest_entry', 'maximum_row_value', 'upcast', 'print_table',
    'sparray',
]
