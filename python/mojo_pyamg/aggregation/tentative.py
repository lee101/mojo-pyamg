"""Tentative prolongator.  Mirrors `pyamg.aggregation.tentative`.

`fit_candidates` calls the ported `fit_candidates` kernel from
`amg_core/smoothed_aggregation.h`.
"""

import numpy as np
from scipy.sparse import bsr_array, csr_array, issparse

from .._lib import addr, f64, i64, lib


def fit_candidates(AggOp, B, tol=1e-10):
    """Fit near-nullspace candidates to form the tentative prolongator.

    Parameters
    ----------
    AggOp : csr_array
        Describes the sparsity pattern of the tentative prolongator.
        Has dimension (#blocks, #aggregates).
    B : array
        The near-nullspace candidates stored in column-wise fashion.
        Has dimension (#blocks * blocksize, #candidates).
    tol : scalar
        Threshold for eliminating local basis functions.  If after
        orthogonalization a local basis function Q[:, j] is small, i.e.
        ||Q[:, j]|| < tol, then Q[:, j] is set to zero.

    Returns
    -------
    (bsr_array, array)
        The tentative prolongator Q, a sparse block matrix of dimensions
        (#blocks * blocksize, #aggregates * #candidates) formed by dense blocks
        of size (blocksize, #candidates), and the coarse level candidates R of
        dimensions (#aggregates * #candidates, #candidates).

    See Also
    --------
    amg_core.fit_candidates

    Notes
    -----
    Assuming that each row of AggOp contains exactly one non-zero entry, i.e.
    all unknowns belong to an aggregate, then Q and R satisfy the relationship
    B = Q@R.

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M.,
       "Algebraic Multigrid by Smoothed Aggregation for
       Second and Fourth Order Elliptic Problems",
       Computing, vol. 56, no. 3, pp. 179--196, 1996.

    Examples
    --------
    >>> AggOp = csr_array([[1, 0],
    ...                    [1, 0],
    ...                    [0, 1],
    ...                    [0, 1]])
    >>> B = [[1], [1], [1], [1]]
    >>> Q, R = fit_candidates(AggOp, B)
    >>> Q.toarray()
    array([[0.70710678, 0.        ],
           [0.70710678, 0.        ],
           [0.        , 0.70710678],
           [0.        , 0.70710678]])

    """
    if not issparse(AggOp) or AggOp.format != 'csr':
        raise TypeError('expected csr_array for argument AggOp')

    B = np.asarray(B)
    if B.dtype not in ['float32', 'float64', 'complex64', 'complex128']:
        B = np.asarray(B, dtype='float64')
    if B.dtype != 'float64':
        B = np.asarray(B, dtype='float64')

    if len(B.shape) != 2:
        raise ValueError('expected 2d array for argument B')

    if B.shape[0] % AggOp.shape[0] != 0:
        raise ValueError(f'Dimensions of AggOp {AggOp.shape} and B {B.shape} '
                         'are incompatible')

    N_fine, N_coarse = AggOp.shape

    K1 = int(B.shape[0] / N_fine)  # dof per supernode (e.g. 3 for 3d vectors)
    K2 = B.shape[1]                # candidates

    # the first two dimensions of R and Qx are collapsed later
    R = np.empty((N_coarse, K2, K2), dtype=np.float64)    # coarse candidates
    Qx = np.empty((AggOp.nnz, K1, K2), dtype=np.float64)  # BSR data array

    AggOp_csc = AggOp.tocsc()

    Ap = i64(AggOp_csc.indptr)
    Ai = i64(AggOp_csc.indices)
    Bx = f64(B.ravel())
    Rx = f64(R.ravel())
    lib().mpa_fit_candidates(N_fine, N_coarse, K1, K2, addr(Ap), addr(Ai),
                             addr(Qx), addr(Bx), addr(Rx), tol)

    Q = bsr_array((Qx.swapaxes(1, 2).copy(), AggOp_csc.indices,
                    AggOp_csc.indptr), shape=(K2 * N_coarse, K1 * N_fine))
    Q = Q.T.tobsr()
    R = R.reshape(-1, K2)

    return Q, R


__all__ = ['fit_candidates']
