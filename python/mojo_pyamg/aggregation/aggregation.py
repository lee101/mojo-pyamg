"""Support for aggregation-based AMG.  Mirrors `pyamg.aggregation.aggregation`."""

from warnings import warn

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_array, issparse

from ..multilevel import MultilevelSolver
from ..relaxation.smoothing import change_smoothers
from ..relaxation.utils import relaxation_as_linear_operator
from ..strength import (classical_strength_of_connection,
                        energy_based_strength_of_connection,
                        symmetric_strength_of_connection)
from ..util.utils import (asfptype, eliminate_diag_dom_nodes, get_blocksize,
                          levelize_smooth_or_improve_candidates,
                          levelize_strength_or_aggregation)
from .aggregate import naive_aggregation, pairwise_aggregation, standard_aggregation
from .smooth import (jacobi_prolongation_smoother, richardson_prolongation_smoother)
from .tentative import fit_candidates


def smoothed_aggregation_solver(A, B=None, BH=None,
                                symmetry='hermitian', strength='symmetric',
                                aggregate='standard',
                                smooth=('jacobi', {'omega': 4.0 / 3.0}),
                                presmoother=('block_gauss_seidel',
                                             {'sweep': 'symmetric'}),
                                postsmoother=('block_gauss_seidel',
                                              {'sweep': 'symmetric'}),
                                improve_candidates=(('block_gauss_seidel',
                                                     {'sweep': 'symmetric',
                                                      'iterations': 4}),
                                                    None),
                                max_levels=10, max_coarse=10,
                                diagonal_dominance=False,
                                keep=False, **kwargs):
    """Create a multilevel solver using classical-style Smoothed Aggregation (SA).

    Parameters
    ----------
    A : csr_array, bsr_array
        Square matrix in CSR or BSR format.
    B : array
        Right near-nullspace candidates, (#blocks*blocksize, #candidates).
    BH : array
        Left near-nullspace candidates; only for nonsymmetric A.
    symmetry : str
        'symmetric', 'hermitian' or 'nonsymmetric'.
    strength : ['symmetric', 'classical', 'energy_based', None, list, tuple]
        Method used to determine the strength of connection between unknowns.
    aggregate : ['standard', 'naive', 'pairwise', ('predefined', {'AggOp': Agg}), list]
        Method used to aggregate nodes.
    smooth : [('jacobi', {'omega': 4.0/3.0}), ('richardson', {'omega': 4.0/3.0}),
              None, list]
        Method used to smooth the tentative prolongator.
    presmoother, postsmoother : str, dict, tuple, list
        Pre- and postsmoother used in the multigrid cycle.
    improve_candidates : tuple, str, list
        Relaxation method used to improve the near-nullspace candidates by
        relaxing on A B = 0.
    max_levels : int
        Maximum number of levels to be used in the multilevel solver.
    max_coarse : int
        Maximum number of variables permitted on the coarse grid.
    diagonal_dominance : bool, tuple
        Avoid coarsening diagonally dominant rows.
    keep : bool
        Flag to indicate keeping extra operators in the hierarchy for
        diagnostics: C, T and AggOp.
    **kwargs : dict
        Extra keywords passed to MultilevelSolver.

    Returns
    -------
    MultilevelSolver
        Smoothed aggregation solver.

    Notes
    -----
    - This method implements classical-style SA, not root-node style SA.
    - At each level, four steps are executed in order to define the coarser
      level operator: strength of connection, aggregation, a tentative
      prolongator by injection, and prolongation smoothing.

    References
    ----------
    .. [1] Vanek, P. and Mandel, J. and Brezina, M.,
       "Algebraic Multigrid by Smoothed Aggregation for
       Second and Fourth Order Elliptic Problems",
       Computing, vol. 56, no. 3, pp. 179--196, 1996.

    Examples
    --------
    >>> from mojo_pyamg import smoothed_aggregation_solver
    >>> A = poisson((100, 100), format='csr')           # matrix
    >>> b = np.ones((A.shape[0]))                      # RHS
    >>> ml = smoothed_aggregation_solver(A)            # AMG solver
    >>> x = ml.solve(b, tol=1e-8)

    """
    if not issparse(A) or A.format not in ('bsr', 'csr'):
        try:
            A = csr_array(A)
            warn('Implicit conversion of A to CSR', SparseEfficiencyWarning)
        except Exception as e:
            raise TypeError('Argument A must have type csr_array or '
                            'bsr_array, or be convertible to csr_array') from e

    A = asfptype(A)

    if symmetry not in ('symmetric', 'hermitian', 'nonsymmetric'):
        raise ValueError('Expected "symmetric", "nonsymmetric" or "hermitian" '
                         'for the symmetry parameter ')
    A.symmetry = symmetry

    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')

    # Right near nullspace candidates use constant for each variable as default
    if B is None:
        B = np.kron(np.ones((int(A.shape[0] / get_blocksize(A)), 1), dtype=np.float64),
                    np.eye(get_blocksize(A), dtype=np.float64))
    else:
        B = np.asarray(B, dtype=np.float64)
        if len(B.shape) == 1:
            B = B.reshape(-1, 1)
        if B.shape[0] != A.shape[0]:
            raise ValueError('The shape of near null-space modes B is incorrect')
        if B.shape[1] < get_blocksize(A):
            warn('Having less target vectors, B.shape[1], than blocksize of A '
                 'can degrade convergence factors.')

    # Left near nullspace candidates
    if A.symmetry == 'nonsymmetric':
        if BH is None:
            BH = B.copy()
        else:
            BH = np.asarray(BH, dtype=np.float64)
            if len(BH.shape) == 1:
                BH = BH.reshape(-1, 1)
            if BH.shape[1] != B.shape[1]:
                raise ValueError('The number of left and right near null-space '
                                 'modes B and BH must be equal')
            if BH.shape[0] != A.shape[0]:
                raise ValueError('The shape of near null-space modes B is incorrect')

    # Levelize the user parameters, so that they become lists describing the
    # desired user option on each level.
    max_levels, max_coarse, strength = \
        levelize_strength_or_aggregation(strength, max_levels, max_coarse)
    max_levels, max_coarse, aggregate = \
        levelize_strength_or_aggregation(aggregate, max_levels, max_coarse)
    improve_candidates = \
        levelize_smooth_or_improve_candidates(improve_candidates, max_levels)
    smooth = levelize_smooth_or_improve_candidates(smooth, max_levels)

    # Construct multilevel structure
    levels = []
    levels.append(MultilevelSolver.Level())
    levels[-1].A = A          # matrix

    # Append near nullspace candidates
    levels[-1].B = B          # right candidates
    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH    # left candidates

    while len(levels) < max_levels and \
            int(levels[-1].A.shape[0] / get_blocksize(levels[-1].A)) > max_coarse:
        _extend_hierarchy(levels, strength, aggregate, smooth,
                          improve_candidates, diagonal_dominance, keep)

    ml = MultilevelSolver(levels, **kwargs)
    change_smoothers(ml, presmoother, postsmoother)
    return ml


def _extend_hierarchy(levels, strength, aggregate, smooth, improve_candidates,
                      diagonal_dominance=False, keep=True):
    """Extend the multigrid hierarchy.

    Service routine to implement the strength of connection, aggregation,
    tentative prolongation construction, and prolongation smoothing.  Called by
    `smoothed_aggregation_solver`.
    """
    def unpack_arg(v):
        if isinstance(v, tuple):
            return v[0], v[1]
        return v, {}

    A = levels[-1].A
    B = levels[-1].B
    AH = None
    BH = None
    TH = None
    if A.symmetry == 'nonsymmetric':
        AH = A.T.conjugate().asformat(A.format)
        BH = levels[-1].BH

    # Compute the strength-of-connection matrix C, where larger
    # C[i,j] denote stronger couplings between i and j.
    fn, kwargs = unpack_arg(strength[len(levels) - 1])
    if fn == 'symmetric':
        C = symmetric_strength_of_connection(A, **kwargs)
    elif fn == 'classical':
        C = classical_strength_of_connection(A, **kwargs)
    elif fn == 'energy_based':
        C = energy_based_strength_of_connection(A, **kwargs)
    elif fn == 'predefined':
        C = kwargs['C'].tocsr()
    elif fn is None:
        C = A.tocsr()
    else:
        raise ValueError(f'Unrecognized strength of connection method: {fn!s}')

    # Avoid coarsening diagonally dominant rows
    flag, kwargs = unpack_arg(diagonal_dominance)
    if flag:
        C = eliminate_diag_dom_nodes(A, C, **kwargs)

    # Compute the aggregation matrix AggOp (i.e., the nodal coarsening of A).
    fn, kwargs = unpack_arg(aggregate[len(levels) - 1])
    Cnodes = None
    if fn == 'standard':
        AggOp, Cnodes = standard_aggregation(C, **kwargs)
    elif fn == 'naive':
        AggOp, Cnodes = naive_aggregation(C, **kwargs)
    elif fn == 'pairwise':
        AggOp, Cnodes = pairwise_aggregation(A, **kwargs)
    elif fn == 'predefined':
        AggOp = kwargs['AggOp'].tocsr()
    else:
        raise ValueError(f'Unrecognized aggregation method {fn!s}')

    # Improve near nullspace candidates by relaxing on A B = 0
    fn, kwargs = unpack_arg(improve_candidates[len(levels) - 1])
    if fn is not None:
        b = np.zeros((A.shape[0], 1), dtype=np.float64)
        B = relaxation_as_linear_operator((fn, kwargs), A, b) @ B
        levels[-1].B = B
        if A.symmetry == 'nonsymmetric':
            BH = relaxation_as_linear_operator((fn, kwargs), AH, b) @ BH
            levels[-1].BH = BH

    # Compute the tentative prolongator, T, which is a tentative interpolation
    # matrix from the coarse-grid to the fine-grid.  T exactly interpolates
    # B_fine = T B_coarse.
    T, B = fit_candidates(AggOp, B)
    if A.symmetry == 'nonsymmetric':
        TH, BH = fit_candidates(AggOp, BH)

    # Smooth the tentative prolongator, so that it's accuracy is greatly
    # improved for algebraically smooth error.
    fn, kwargs = unpack_arg(smooth[len(levels) - 1])
    if fn == 'jacobi':
        P = jacobi_prolongation_smoother(A, T, C, B, **kwargs)
    elif fn == 'richardson':
        P = richardson_prolongation_smoother(A, T, **kwargs)
    elif fn is None:
        P = T
    else:
        raise ValueError(f'Unrecognized prolongation smoother method {fn!s}')

    # Compute the restriction matrix, R, which interpolates from the fine-grid
    # to the coarse-grid.  If A is nonsymmetric, then R must be constructed
    # based on A.H.  Otherwise R = P.H or P.T.
    symmetry = A.symmetry
    if symmetry == 'hermitian':
        R = P.T.conjugate()
    elif symmetry == 'symmetric':
        R = P.T
    elif symmetry == 'nonsymmetric':
        fn, kwargs = unpack_arg(smooth[len(levels) - 1])
        if fn == 'jacobi':
            R = jacobi_prolongation_smoother(AH, TH, C, BH, **kwargs).T.conjugate()
        elif fn == 'richardson':
            R = richardson_prolongation_smoother(AH, TH, **kwargs).T.conjugate()
        elif fn is None:
            R = T.T.conjugate()
        else:
            raise ValueError(f'Unrecognized prolongation smoother method {fn!s}')
    else:
        raise ValueError('Unrecognized symmetry.')

    if keep:
        levels[-1].C = C            # strength of connection matrix
        levels[-1].AggOp = AggOp    # aggregation operator
        levels[-1].Cnodes = Cnodes  # centers used to generate aggregates
        levels[-1].T = T            # tentative prolongator

    levels[-1].P = P  # smoothed prolongator
    levels[-1].R = R  # restriction operator

    levels.append(MultilevelSolver.Level())
    A = R @ A @ P              # Galerkin operator
    A.symmetry = symmetry
    levels[-1].A = A
    levels[-1].B = B           # right near nullspace candidates

    if A.symmetry == 'nonsymmetric':
        levels[-1].BH = BH     # left near nullspace candidates


__all__ = ['smoothed_aggregation_solver']
