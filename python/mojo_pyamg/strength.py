"""Strength of connection matrices.  Mirrors `pyamg.strength`.

The classical and symmetric measures are computed by the ported Mojo kernels
(`amg_core/ruge_stuben.h` and `amg_core/smoothed_aggregation.h`); the Python
around them is upstream's, unchanged in structure.
"""

from warnings import warn

import numpy as np
from scipy import sparse

from ._lib import addr, f64, i64, lib
from .util.linalg import approximate_spectral_radius
from .util.utils import amalgamate, get_blocksize, scale_rows_by_largest_entry


def classical_strength_of_connection(A, theta=0.1, block=True, norm='abs'):
    """Classical AMG strength of connection.

    An off-diagonal entry ``A[i,j]`` is a strong connection iff::

             |A[i,j]| >= theta * max(|A[i,k]|), where k != i     (norm='abs')
             -A[i,j]  >= theta * max(-A[i,k]),  where k != i     (norm='min')

    Parameters
    ----------
    A : csr_array or bsr_array
        Square, sparse matrix in CSR or BSR format.
    theta : float
        Threshold parameter in [0,1].
    block : bool, default True
        Compute strength of connection block-wise.
    norm : str, default 'abs'
        'abs', 'min' or 'fro'; see above.  For block-wise computations C[i,j]
        is respectively the maximum absolute value, the minimum (negative)
        value, or the Frobenius norm of block A[i, j].

    Returns
    -------
    csr_array
        Matrix graph defining strong connections.

    See Also
    --------
    symmetric_strength_of_connection

    """
    if sparse.issparse(A) and A.format == 'bsr':
        if (A.blocksize[0] != A.blocksize[1]) or (A.blocksize[0] < 1):
            raise ValueError('Matrix must have square blocks')
        blocksize = A.blocksize[0]
    else:
        blocksize = 1

    if (theta < 0 or theta > 1):
        raise ValueError('expected theta in [0,1]')

    # Block structure considered before computing SOC
    if block and sparse.issparse(A) and A.format == 'bsr':
        N = int(A.shape[0] / blocksize)
        # SOC based on maximum absolute value element in each block
        if norm == 'abs':
            data = np.max(np.max(np.abs(A.data), axis=1), axis=1)
        # SOC based on hard minimum of entry in each off-diagonal block
        elif norm == 'min':
            data = np.min(np.min(A.data, axis=1), axis=1)
        # SOC based on Frobenius norms of blocks
        elif norm == 'fro':
            data = np.conjugate(A.data) * A.data
            data = np.sum(np.sum(data, axis=1), axis=1)
        else:
            raise ValueError('Invalid choice of norm.')
        # drop small numbers
        data[np.abs(data) < 1e-16] = 0.0
    else:
        if not sparse.issparse(A) or A.format != 'csr':
            warn('Implicit conversion of A to csr', sparse.SparseEfficiencyWarning)
            A = sparse.csr_array(A)
        data = A.data
        N = A.shape[0]

    Ap = i64(A.indptr)
    Aj = i64(A.indices)
    Ax = f64(data)

    Sp = i64(np.empty_like(A.indptr))
    Sj = i64(np.empty_like(A.indices))
    Sx = f64(np.empty_like(data), copy=True)

    if norm in ('abs', 'fro'):
        lib().mpa_classical_strength_of_connection_abs(
            N, theta, addr(Ap), addr(Aj), addr(Ax),
            addr(Sp), addr(Sj), addr(Sx))
    elif norm == 'min':
        lib().mpa_classical_strength_of_connection_min(
            N, theta, addr(Ap), addr(Aj), addr(Ax),
            addr(Sp), addr(Sj), addr(Sx))
    else:
        raise ValueError('Unrecognized option for norm for strength.')

    S = sparse.csr_array((Sx[:Sp[-1]], Sj[:Sp[-1]], Sp), shape=[N, N])
    # Take magnitude and scale by largest entry
    S.data = np.abs(S.data)
    S = scale_rows_by_largest_entry(S)
    S.eliminate_zeros()

    if blocksize > 1 and not block:
        S = amalgamate(S, blocksize)

    return S


def symmetric_strength_of_connection(A, theta=0):
    """The standard symmetric measure of strength of connection.

    An off-diagonal connection A[i,j] is strong iff::

        abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )

    Parameters
    ----------
    A : csr_array
        Matrix graph defined in sparse format.
    theta : float
        Threshold parameter (positive).

    Returns
    -------
    csr_array
        Matrix graph defining strong connections.

    See Also
    --------
    classical_strength_of_connection

    """
    if theta < 0:
        raise ValueError('expected a positive theta')

    if sparse.issparse(A) and A.format == 'csr':
        Ap = i64(A.indptr)
        Aj = i64(A.indices)
        Ax = f64(A.data)
        Sp = i64(np.empty_like(A.indptr))
        Sj = i64(np.empty_like(A.indices))
        Sx = f64(np.empty_like(A.data), copy=True)
        diags = np.empty(A.shape[0], dtype=np.float64)
        lib().mpa_symmetric_strength_of_connection(
            A.shape[0], theta, addr(Ap), addr(Aj), addr(Ax),
            addr(Sp), addr(Sj), addr(Sx), addr(diags))
        S = sparse.csr_array((Sx[:Sp[-1]], Sj[:Sp[-1]], Sp), shape=A.shape)
    elif sparse.issparse(A) and A.format == 'bsr':
        M, N = A.shape
        R, C = A.blocksize
        if R != C:
            raise ValueError('matrix must have square blocks')
        if theta == 0:
            data = np.ones(len(A.indices), dtype=np.float64)
            S = sparse.csr_array((data, A.indices.copy(), A.indptr.copy()),
                                  shape=(int(M / R), int(N / C)))
        else:
            # the strength of connection matrix is based on the
            # Frobenius norms of the blocks
            data = (np.conjugate(A.data) * A.data).reshape(-1, R * C)
            data = np.sqrt(data.sum(axis=1))
            A = sparse.csr_array((data, A.indices, A.indptr),
                                  shape=(int(M / R), int(N / C)))
            return symmetric_strength_of_connection(A, theta)
    else:
        raise TypeError('expected CSR or BSR sparse format')

    # Strength represents "distance", so take the magnitude
    S.data = np.abs(S.data)
    # Scale S by the largest magnitude entry in each row
    S = scale_rows_by_largest_entry(S)
    return S


def energy_based_strength_of_connection(A, theta=0.0, k=2):
    """Strength of connection based on an approximate inverse of A.

    Parameters
    ----------
    A : csr_array, bsr_array
        Square, sparse matrix in CSR or BSR format.
    theta : float
        Threshold parameter in [0,1).
    k : int, default 2
        Number of Jacobi relaxation steps used to approximate A^{-1}.

    Returns
    -------
    csr_array
        Matrix graph defining strong connections.

    """
    if theta < 0:
        raise ValueError('expected a positive theta')
    if not sparse.issparse(A):
        raise ValueError('expected sparse matrix')
    if k < 0:
        raise ValueError('expected positive number of steps')
    if not isinstance(k, int):
        raise ValueError('expected integer')
    if A.format == 'bsr':
        bsr_flag = True
        numPDEs = A.blocksize[0]
        if A.blocksize[0] != A.blocksize[1]:
            raise ValueError('expected square blocks in BSR matrix A')
    else:
        bsr_flag = False
        numPDEs = 1
    # Convert A to csc and Atilde to csr
    if A.format == 'csr':
        Atilde = A.copy()
        A = A.tocsc()
    else:
        A = A.tocsc()
        Atilde = A.copy()
        Atilde = Atilde.tocsr()
    # Calculate the weighted-Jacobi parameter
    D = A.diagonal()
    Dinv = 1.0 / D
    Dinv[D == 0] = 0.0
    Dinv = sparse.csc_array((Dinv, (np.arange(A.shape[0], dtype=A.indptr.dtype),
                                    np.arange(A.shape[1], dtype=A.indptr.dtype))),
                            shape=A.shape)
    DinvA = Dinv @ A
    omega = 1.0 / approximate_spectral_radius(DinvA)
    del DinvA
    # Approximate A-inverse with k steps of w-Jacobi and a zero initial guess
    S = sparse.csc_array(A.shape, dtype=np.float64)  # empty matrix
    Id = sparse.eye_array(A.shape[0], A.shape[1], format='csc')
    for _i in range(k + 1):
        S = S + omega * (Dinv @ (Id - A @ S))
    # Calculate the strength entries in S column-wise, but only strength
    # values at the sparsity pattern of A
    for i in range(Atilde.shape[0]):
        v = S[:, [i]].toarray()
        v = v.ravel()
        Av = A @ v
        denom = np.sqrt(np.inner(v.conj(), Av))
        # replace entries in row i with strength values
        for j in range(Atilde.indptr[i], Atilde.indptr[i + 1]):
            col = Atilde.indices[j]
            vj = v[col].copy()
            v[col] = 0.0
            #   =  (||v_j||_A - ||v||_A) / ||v||_A
            val = np.sqrt(np.inner(v.conj(), A @ v)) / denom - 1.0
            # Negative values generally imply a weak connection
            if val > -0.01:
                Atilde.data[j] = abs(val)
            else:
                Atilde.data[j] = 0.0
            v[col] = vj
    # Apply drop tolerance
    Atilde = classical_strength_of_connection(Atilde, theta=theta)
    Atilde.eliminate_zeros()
    # Put ones on the diagonal
    Atilde = Atilde + Id.tocsr()
    Atilde.sort_indices()
    # Amalgamate Atilde for the BSR case, using ones for all strong connections
    if bsr_flag:
        Atilde = Atilde.tobsr(blocksize=(numPDEs, numPDEs))
        nblocks = Atilde.indices.shape[0]
        uone = np.ones((nblocks,))
        Atilde = sparse.csr_array((uone, Atilde.indices, Atilde.indptr),
                                   shape=(int(Atilde.shape[0] / numPDEs),
                                          int(Atilde.shape[1] / numPDEs)))
    # Scale C by the largest magnitude entry in each row
    Atilde = scale_rows_by_largest_entry(Atilde)
    return Atilde


__all__ = ['classical_strength_of_connection', 'symmetric_strength_of_connection',
           'energy_based_strength_of_connection', 'get_blocksize']
