"""Prolongator smoothing.  Mirrors `pyamg.aggregation.smooth`.

`satisfy_constraints` calls the ported `satisfy_constraints_helper` kernel from
`amg_core/smoothed_aggregation.h`.  The energy-minimizing smoothers
(`cg_prolongation_smoothing`, `cgnr_prolongation_smoothing`,
`energy_prolongation_smoother`) are not part of this port; see the README.
"""

from warnings import warn

import numpy as np
from scipy import sparse

from .._lib import addr, f64, i64, lib
from ..util.linalg import approximate_spectral_radius
from ..util.utils import (compute_BtBinv, get_block_diag, get_diagonal, scale_rows,
                          unamal)


def satisfy_constraints(U, B, BtBinv):
    """Project the components of span(B) out of U, row block by row block.

    Parameters
    ----------
    U : bsr_array
        m x n sparse bsr matrix, the update to the prolongator.
    B : array
        n x k array of the coarse grid near nullspace vectors.
    BtBinv : array
        Local inv(B_i.H@B_i) matrices for each supernode i, where B_i is B
        restricted to the sparsity pattern of supernode i in U.

    Returns
    -------
    bsr_array
        Updated U, so that U@B = 0.  The update is computed by orthogonally
        (in 2-norm) projecting out the components of span(B) in U in a
        row-wise fashion.

    See Also
    --------
    amg_core.satisfy_constraints_helper

    """
    rows_per_block = U.blocksize[0]
    cols_per_block = U.blocksize[1]
    num_block_rows = int(U.shape[0] / rows_per_block)
    NullDim = B.shape[1]

    UB = np.ravel(U @ B)
    Update = np.zeros(rows_per_block * cols_per_block, dtype=np.float64)
    C = np.zeros(NullDim * cols_per_block, dtype=np.float64)

    Bt = f64(np.conjugate(np.ravel(B)))
    UBx = f64(UB)
    BtBinvx = f64(np.ravel(BtBinv))
    Sp = i64(U.indptr)
    Sj = i64(U.indices)
    Sx = f64(np.ravel(U.data))
    lib().mpa_satisfy_constraints_helper(
        rows_per_block, cols_per_block, num_block_rows, NullDim,
        addr(Bt), addr(UBx), addr(BtBinvx), addr(Sp), addr(Sj), addr(Sx),
        addr(Update), addr(C))

    return U


def jacobi_prolongation_smoother(S, T, C, B, omega=4.0 / 3.0, degree=1,
                                 filter_entries=False, weighting='diagonal'):
    """Jacobi prolongation smoother.

    Parameters
    ----------
    S : csr_array, bsr_array
        Sparse NxN matrix used for smoothing.  Typically, A.
    T : csr_array, bsr_array
        Tentative prolongator.
    C : csr_array, bsr_array
        Strength-of-connection matrix.
    B : array
        Near nullspace modes for the coarse grid such that T@B exactly
        reproduces the fine grid near nullspace modes.
    omega : scalar
        Damping parameter.
    degree : int
        Number of passes.
    filter_entries : bool
        If true, filter S before smoothing T.
    weighting : str
        'block', 'diagonal' or 'local' weighting for constructing the Jacobi D.

    Returns
    -------
    csr_array, bsr_array
        Smoothed (final) prolongator defined by P = (I - omega/rho(K) K) @ T
        where K = diag(S)^-1 @ S.

    See Also
    --------
    richardson_prolongation_smoother

    """
    if weighting == 'block':
        if sparse.issparse(S) and S.format == 'csr':
            weighting = 'diagonal'
        elif sparse.issparse(S) and S.format == 'bsr':
            if S.blocksize[0] == 1:
                weighting = 'diagonal'
        else:
            raise TypeError('S must be sparse BSR or CSR format')

    if filter_entries:
        # Implement filtered prolongation smoothing for the general case by
        # utilizing satisfy constraints
        if sparse.issparse(S) and S.format == 'bsr':
            numPDEs = S.blocksize[0]
        else:
            numPDEs = 1

        # Create a filtered S with entries dropped that aren't in C
        C = unamal(C, numPDEs, numPDEs)
        S = S.multiply(C)
        S.eliminate_zeros()

    if weighting == 'diagonal':
        # Use diagonal of S
        D_inv = get_diagonal(S, inv=True)
        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = (omega / approximate_spectral_radius(D_inv_S)) * D_inv_S
    elif weighting == 'block':
        # Use block diagonal of S
        D_inv = get_block_diag(S, blocksize=S.blocksize[0], inv_flag=True)
        D_inv = sparse.bsr_array((D_inv, np.arange(D_inv.shape[0], dtype=np.int32),
                                  np.arange(D_inv.shape[0] + 1, dtype=np.int32)),
                                 shape=S.shape)
        D_inv_S = D_inv @ S
        D_inv_S = (omega / approximate_spectral_radius(D_inv_S)) * D_inv_S
    elif weighting == 'local':
        # Use the Gershgorin estimate as each row's weight, instead of a global
        # spectral radius estimate
        D = np.abs(S) @ np.ones((S.shape[0], 1), dtype=np.float64)
        D_inv = np.zeros_like(D)
        D_inv[D != 0] = 1.0 / np.abs(D[D != 0])

        D_inv_S = scale_rows(S, D_inv, copy=True)
        D_inv_S = omega * D_inv_S
    else:
        raise ValueError('Incorrect weighting option')

    if filter_entries:
        # Carry out Jacobi, but after calculating the prolongator update, U,
        # apply satisfy constraints so that U@B = 0
        P = T
        for _ in range(degree):
            U = (D_inv_S @ P).tobsr(blocksize=P.blocksize)

            # (1) Construct array of inv(Bi'Bi), where Bi is B restricted to
            # row i's sparsity pattern.  (2) Apply satisfy constraints.
            BtBinv = compute_BtBinv(B, U)
            satisfy_constraints(U, B, BtBinv)

            # Update P
            P = P - U

    else:
        # Carry out Jacobi as normal
        P = T
        for _ in range(degree):
            P = P - (D_inv_S @ P)

    return P


def richardson_prolongation_smoother(S, T, omega=4.0 / 3.0, degree=1):
    """Richardson prolongation smoother.

    Parameters
    ----------
    S : csr_array, bsr_array
        Sparse NxN matrix used for smoothing.  Typically A.
    T : csr_array, bsr_array
        Tentative prolongator.
    omega : scalar
        Damping parameter.
    degree : int
        Number of passes.

    Returns
    -------
    csr_array, bsr_array
        Smoothed (final) prolongator defined by P = (I - omega/rho(S) S) @ T
        where rho(S) is an approximation to the spectral radius of S.

    See Also
    --------
    jacobi_prolongation_smoother

    """
    weight = omega / approximate_spectral_radius(S)

    P = T
    for _ in range(degree):
        P = P - weight * (S @ P)

    return P


def energy_prolongation_smoother(A, T, Atilde, B, Bf, Cpt_params, krylov='cg',
                                 maxiter=4, tol=1e-8, degree=1, weighting='local',
                                 prefilter=None, postfilter=None):
    """Energy minimizing prolongation smoother.

    Not part of this port: the upstream implementation builds `BtBinv` from a
    pseudo-inverse of each local B_i block and then runs a constrained Krylov
    iteration, which is a different algorithm from the Jacobi and Richardson
    smoothers above rather than a Mojo rewrite of them.  Use
    ``jacobi_prolongation_smoother`` or ``richardson_prolongation_smoother``.
    """
    raise NotImplementedError(
        'energy_prolongation_smoother is not part of this port; use '
        'jacobi_prolongation_smoother or richardson_prolongation_smoother')


__all__ = ['satisfy_constraints', 'jacobi_prolongation_smoother',
           'richardson_prolongation_smoother', 'energy_prolongation_smoother']
