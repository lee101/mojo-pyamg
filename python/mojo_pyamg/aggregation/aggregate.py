"""Aggregation methods.  Mirrors `pyamg.aggregation.aggregate`.

The three kernels below are the ported Mojo routines from
`amg_core/smoothed_aggregation.h`; the wrappers are upstream's.
"""

from warnings import warn

import numpy as np
from scipy import sparse

from .._lib import addr, i64, lib
from ..strength import classical_strength_of_connection


def _pairwise_scratch(n_row):
    """Workspace for `pairwise_aggregation`.

    One array, because the kernel's bucket lists are intrusive links into it:
    m, head, next, prev, key laid out back to back.  See `bucket_insert` in
    src/ported.mojo for why they must not be separate pointer arguments.
    """
    return np.zeros(6 * n_row + 1, dtype=np.int64)


def standard_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator, greedy MIS.

    Parameters
    ----------
    C : csr_array
        Strength-of-connection matrix.

    Returns
    -------
    (csr_array, array)
        The aggregation operator AggOp (num_rows, num_aggregates) and the list
        of Cpts, the representative node of each aggregate.

    See Also
    --------
    pyamg.aggregation.tentative.fit_candidates

    Notes
    -----
    - It is assumed that C is symmetric.
    - Unaggregated nodes are marked with -1.

    """
    if not sparse.issparse(C) or C.format != 'csr':
        raise TypeError('expected csr_array')
    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')
    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.zeros(num_rows, dtype=np.int64)  # stores the aggregate #s
    Cpts = np.zeros(num_rows, dtype=np.int64)  # stores the Cpts
    Ap = i64(C.indptr)
    Aj = i64(C.indices)

    num_aggregates = lib().mpa_standard_aggregation(num_rows, addr(Ap), addr(Aj),
                                                    addr(Tj), addr(Cpts))
    Cpts = Cpts[:num_aggregates]

    # no nodes aggregated
    if num_aggregates == 0:
        # return all zero matrix and no Cpts
        return sparse.csr_array((num_rows, 1), dtype=np.int32), \
            np.array([], dtype=index_type)

    shape = (num_rows, num_aggregates)

    # some nodes not aggregated
    if Tj.min() == -1:
        mask = Tj != -1
        row = np.arange(num_rows, dtype=np.int64)[mask]
        col = Tj[mask]
        data = np.ones(len(col), dtype=np.int32)
        return sparse.coo_array((data, (row, col)), shape=shape).tocsr(), Cpts

    # all nodes aggregated
    Tp = np.arange(num_rows + 1, dtype=np.int64)
    Tx = np.ones(len(Tj), dtype=np.int32)
    return sparse.csr_array((Tx, Tj, Tp), shape=shape), Cpts


def naive_aggregation(C):
    """Compute the sparsity pattern of the tentative prolongator, greedy walk.

    Parameters
    ----------
    C : csr_array
        Strength-of-connection matrix.

    Returns
    -------
    (csr_array, array)
        The aggregation operator AggOp and the list of Cpts.

    Notes
    -----
    Differs from standard aggregation.  Each dof is considered.  If it has been
    aggregated, skip over.  Otherwise, put dof and any unaggregated neighbors in
    an aggregate.  Results in possibly much higher complexities.

    """
    if not sparse.issparse(C) or C.format != 'csr':
        raise TypeError('expected csr_array')
    if C.shape[0] != C.shape[1]:
        raise ValueError('expected square matrix')
    index_type = C.indptr.dtype
    num_rows = C.shape[0]

    Tj = np.zeros(num_rows, dtype=np.int64)  # stores the aggregate #s
    Cpts = np.zeros(num_rows, dtype=np.int64)  # stores the Cpts
    Ap = i64(C.indptr)
    Aj = i64(C.indices)

    num_aggregates = lib().mpa_naive_aggregation(num_rows, addr(Ap), addr(Aj),
                                                  addr(Tj), addr(Cpts))
    Cpts = Cpts[:num_aggregates]
    Tj = Tj - 1

    if num_aggregates == 0:
        # all zero matrix
        return sparse.csr_array((num_rows, 1), dtype=np.int32), Cpts

    shape = (num_rows, num_aggregates)
    # all nodes aggregated
    Tp = np.arange(num_rows + 1, dtype=np.int64)
    Tx = np.ones(len(Tj), dtype=np.int32)
    return sparse.csr_array((Tx, Tj, Tp), shape=shape), Cpts


def pairwise_aggregation(A, matchings=2, theta=0.25, norm='min', compute_P=False):
    """Pairwise aggregation: match nodes with their strongest strong neighbour.

    Parameters
    ----------
    A : csr_array or bsr_array
        Square matrix in CSR or BSR format.
    matchings : int
        Number of pairwise matchings to do.
    theta : scalar
        Strength of connection threshold parameter in [0, 1].
    norm : string
        Norm type used in calculating the strength of connection ('abs', 'min').
    compute_P : bool
        Compute the prolongator (True) or only the aggregation (False).

    Returns
    -------
    (bsr_array, array)
        The tentative prolongator P and the Cpts of the original fine grid
        nodes.

    See Also
    --------
    amg_core.pairwise_aggregation

    Notes
    -----
    Greedy aggregation based on pairwise matching.

    References
    ----------
    .. [1] Notay, Y., "An aggregation-based algebraic multigrid method,"
       Electronic Transactions on Numerical Analysis, vol. 14, pp. 151--180, 2010.

    """
    # Get SOC matrix
    if not sparse.issparse(A) or A.format not in ('bsr', 'csr'):
        try:
            A = A.tocsr()
            warn('Implicit conversion of A to csr', sparse.SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Invalid matrix type, must be CSR or BSR.') from e
    index_type = A.indptr.dtype
    Ac = A      # Let Ac reference A for loop purposes
    T = None
    Cpts = None
    # Loop over the number of pairwise matchings to be done
    for i in range(0, matchings):
        # Compute SOC matrix for this matching
        if sparse.issparse(A) and A.format == 'bsr':
            C = classical_strength_of_connection(A=Ac, theta=theta, block=True,
                                                norm=norm)
        else:
            C = classical_strength_of_connection(A=Ac, theta=theta, block=False,
                                                norm=norm)
        # Form pairwise aggregation matrix
        num_rows = C.shape[0]
        Tj = np.zeros(num_rows, dtype=np.int64)  # stores the aggregate #s
        new_cpts = np.zeros(num_rows, dtype=np.int64)  # stores the new_cpts
        Sp = i64(C.indptr)
        Sj = i64(C.indices)
        Sx = np.ascontiguousarray(C.data, dtype=np.float64)
        scratch = _pairwise_scratch(num_rows)
        num_aggregates = lib().mpa_pairwise_aggregation(
            num_rows, addr(Sp), addr(Sj), addr(Sx), addr(Tj),
            addr(new_cpts), addr(scratch))
        if Cpts is None:
            Cpts = new_cpts[:num_aggregates]
        else:
            Cpts = Cpts[new_cpts[:num_aggregates]]
        Tj = Tj - 1
        # Construct sparse T
        if num_aggregates == 0:
            # all zero matrix
            T_temp = sparse.csr_array((num_rows, 1), dtype=np.int32)
            warn('No pairwise aggregates found, T = 0.')
        else:
            shape = (num_rows, num_aggregates)
            Tp = np.arange(num_rows + 1, dtype=np.int64)
            # If A is not BSR
            if not sparse.issparse(A) or A.format != 'bsr':
                Tx = np.ones(len(Tj), dtype=np.int32)
                T_temp = sparse.csr_array((Tx, Tj, Tp), shape=shape)
            else:
                shape = (shape[0] * A.blocksize[0], shape[1] * A.blocksize[1])
                Tx = np.array(len(Tj) * [np.identity(A.blocksize[0])], dtype=np.int32)
                T_temp = sparse.bsr_array((Tx, Tj, Tp), blocksize=A.blocksize,
                                          shape=shape)
        # Form aggregation matrix, need to make sure is CSR/BSR
        if i == 0:
            T = T_temp
        elif sparse.issparse(A) and A.format == 'bsr':
            T = sparse.bsr_array(T @ T_temp)
        else:
            T = sparse.csr_array(T @ T_temp)
        # Break loop if zero aggregates were found
        if num_aggregates == 0:
            break
        # Form coarse grid operator for next matching
        if i < (matchings - 1):
            if sparse.issparse(T_temp) and T_temp.format == 'csr':
                Ac = T_temp.T.tocsr() @ Ac @ T_temp
            else:
                Ac = T_temp.T @ Ac @ T_temp
    # Convert T to dtype int if only used for aggregation
    if compute_P:
        T = T.astype(np.float64, copy=False)
    return T, Cpts


__all__ = ['standard_aggregation', 'naive_aggregation', 'pairwise_aggregation']
