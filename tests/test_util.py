"""Utility helpers and the FFI surface itself."""

import ctypes
import re

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_pyamg.util.utils as ours
import mojo_pyamg.strength as our_strength
import pyamg.util.utils as theirs
import pyamg.strength as theirs_strength
from mojo_pyamg._lib import _SIGNATURES, f64, lib

_F = ctypes.c_double


def test_exports_match_the_c_abi():
    """Every ctypes signature matches an `@export` in src/capi.mojo.

    A mismatch here is silent until a kernel reads a pointer from the wrong
    argument, so it is checked rather than trusted.
    """
    import os
    capi = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'src', 'capi.mojo')
    with open(capi) as f:
        source = f.read()

    exports = {}
    for m in re.finditer(r'@export\("(\w+)"\)\ndef \w+\(([^)]*)\)\s*abi\("C"\)',
                         source):
        exports[m.group(1)] = [p.strip() for p in m.group(2).split(',') if p.strip()]

    assert set(exports) == set(_SIGNATURES)
    for name, params in exports.items():
        argtypes, _ = _SIGNATURES[name]
        assert len(argtypes) == len(params), name
        for param, argtype in zip(params, argtypes):
            kind = param.split(':')[1].strip()
            assert (kind == 'Float64') == (argtype is _F), (name, param)


def test_every_symbol_resolves():
    for name in _SIGNATURES:
        assert getattr(lib(), name) is not None


def test_get_blocksize(poisson_2d, blocks_3):
    A, _ = blocks_3
    assert ours.get_blocksize(poisson_2d) == theirs.get_blocksize(poisson_2d) == 1
    assert ours.get_blocksize(A) == theirs.get_blocksize(A) == 3


def test_asfptype():
    A = sp.csr_array(np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert ours.asfptype(A).dtype == theirs.asfptype(A).dtype == np.float64


@pytest.mark.parametrize('norm_eq', [0, 1, 2])
def test_get_diagonal(poisson_2d, norm_eq):
    A = sp.csr_array(poisson_2d)
    assert np.allclose(ours.get_diagonal(A, norm_eq=norm_eq, inv=True),
                       theirs.get_diagonal(poisson_2d, norm_eq=norm_eq, inv=True))


def test_get_block_diag(blocks_3):
    A, _ = blocks_3
    assert np.allclose(ours.get_block_diag(A, blocksize=3, inv_flag=True),
                       theirs.get_block_diag(sp.bsr_array(A), blocksize=3,
                                             inv_flag=True))
    assert np.allclose(ours.get_block_diag(A, blocksize=3, inv_flag=False),
                       theirs.get_block_diag(sp.bsr_array(A), blocksize=3,
                                             inv_flag=False))


def test_scale_rows(poisson_2d, random_csr):
    A = sp.csr_array(poisson_2d)
    v = np.arange(1.0, A.shape[0] + 1)
    assert np.allclose(ours.scale_rows(A, v).toarray(),
                       theirs.scale_rows(poisson_2d, v).toarray())
    B = sp.csr_array(random_csr)
    v = np.random.default_rng(0).random(B.shape[0])
    assert np.allclose(ours.scale_rows(B, v).toarray(),
                       theirs.scale_rows(random_csr, v).toarray())


def test_scale_columns(poisson_2d):
    """The CSC kernel behind scale_rows, called directly."""
    A = sp.csc_array(poisson_2d)
    v = np.arange(1.0, A.shape[1] + 1)
    A.data = np.asarray(A.data, dtype=np.float64)
    Ap = np.ascontiguousarray(A.indptr, dtype=np.int64)
    Aj = np.ascontiguousarray(A.indices, dtype=np.int64)
    Ax = np.ascontiguousarray(A.data, dtype=np.float64)
    Vv = np.ascontiguousarray(v, dtype=np.float64)
    expected = A.toarray() * v[None, :]
    lib().mpa_csc_scale_columns(A.shape[1], int(Ap.ctypes.data),
                               int(Ax.ctypes.data), int(Vv.ctypes.data))
    scaled = sp.csc_array((Ax, Aj, Ap), shape=A.shape).toarray()
    assert np.allclose(scaled, expected)


def test_maximum_row_value(poisson_2d):
    A = sp.csr_array(poisson_2d)
    expected = np.array([np.abs(A.data[A.indptr[i]:A.indptr[i + 1]]).max()
                         for i in range(A.shape[0])])
    assert np.allclose(ours.maximum_row_value(A), expected)


def test_scale_rows_by_largest_entry(poisson_2d):
    C = our_strength.symmetric_strength_of_connection(poisson_2d)
    Ct = theirs_strength.symmetric_strength_of_connection(sp.csr_array(poisson_2d))
    assert np.allclose(ours.scale_rows_by_largest_entry(sp.csr_array(C)).toarray(),
                       theirs.scale_rows_by_largest_entry(Ct).toarray())


def test_truncate_rows(poisson_2d):
    """Same survivors per row; ties may pick a different equal-magnitude entry.

    Upstream keeps the k largest-magnitude entries of each row, found with a
    quicksort over the (data, indices) pair.  This port selects them in place,
    so which of several equal-magnitude entries survives a tie can differ; the
    number kept and their magnitudes agree.
    """
    a = ours.truncate_rows(poisson_2d, 3)
    b = theirs.truncate_rows(poisson_2d, 3)
    assert a.nnz == b.nnz
    assert np.allclose(np.sort(np.abs(a.data)), np.sort(np.abs(b.data)))
    # every row keeps exactly min(row length, k) entries
    for i in range(a.shape[0]):
        assert a.indices[a.indptr[i]:a.indptr[i + 1]].size <= 3


def test_filter_matrix_rows(poisson_2d):
    """The result keeps exactly the entries above the drop tolerance.

    Not compared against `pyamg.util.utils.filter_matrix_rows` entry by entry:
    the installed 5.3.0 returns column indices that still carry the internal
    `+n` offset, so its output is not the filtered matrix it means to be.  The
    reference below is the definition of the drop tolerance instead.
    """
    theta = 0.25
    # a copy: the function offsets A.indices in place and restores it on the
    # way out, as upstream does
    A = sp.csr_array(poisson_2d).copy()
    out = ours.filter_matrix_rows(A, theta).tocsr()

    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    off = A.indices != rows
    thresh = np.zeros(A.shape[0])
    np.maximum.at(thresh, rows[off], np.abs(A.data[off]))
    thresh *= theta
    keep = (A.indices == rows) | (np.abs(A.data) >= thresh[rows])
    indptr = np.zeros(A.shape[0] + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(rows[keep], minlength=A.shape[0]))
    expected = sp.csr_array((A.data[keep], A.indices[keep], indptr),
                            shape=A.shape)
    assert np.allclose(out.toarray(), expected.toarray())
    assert out.nnz <= A.nnz


def test_amalgamate_and_unamal(blocks_3):
    A, _ = blocks_3
    assert np.array_equal(ours.amalgamate(A, 3).toarray(),
                          theirs.amalgamate(sp.bsr_array(A), 3).toarray())
    C = sp.csr_array(our_strength.symmetric_strength_of_connection(sp.csr_array(A)))
    assert np.array_equal(ours.unamal(C, 3, 1).toarray(),
                          theirs.unamal(C, 3, 1).toarray())


def test_levelize_helpers():
    assert ours.levelize_strength_or_aggregation('classical', 10, 10) == \
        theirs.levelize_strength_or_aggregation('classical', 10, 10)
    assert ours.levelize_smooth_or_improve_candidates('jacobi', 4) == \
        theirs.levelize_smooth_or_improve_candidates('jacobi', 4)
    assert ours.levelize_smooth_or_improve_candidates(None, 3) == \
        theirs.levelize_smooth_or_improve_candidates(None, 3)


def test_relaxation_as_linear_operator(poisson_2d):
    import mojo_pyamg.relaxation.utils as our_utils
    import pyamg.relaxation.utils as their_utils
    A = poisson_2d
    b = np.zeros(A.shape[0])
    B = np.ones((A.shape[0], 1))
    op = our_utils.relaxation_as_linear_operator('gauss_seidel', A, b)
    opt = their_utils.relaxation_as_linear_operator('gauss_seidel', A, b)
    assert np.allclose(op @ B, opt @ B)


def test_linalg_helpers(poisson_2d):
    import mojo_pyamg.util.linalg as L
    import pyamg.util.linalg as PL
    A = poisson_2d
    x = np.random.default_rng(0).random(A.shape[0])
    assert L.norm(x) == pytest.approx(PL.norm(x))
    assert L.infinity_norm(A) == pytest.approx(PL.infinity_norm(A))
    # an explicit starting vector: the default is random, and the estimate is
    # not part of anyone's contract
    # each call gets its own vector: upstream normalises initial_guess in place
    # fresh matrices on each side: the estimate is cached on the matrix it is
    # given, and a shared fixture would hand one side a cached value
    guess = np.linspace(1.0, 2.0, A.shape[0])
    r1 = L.approximate_spectral_radius(sp.csr_array(A).copy(),
                                       initial_guess=guess.copy())
    r2 = PL.approximate_spectral_radius(sp.csr_array(A).copy(),
                                        initial_guess=guess.copy())
    assert np.isnan(r1) == np.isnan(r2)
    if not np.isnan(r1):
        assert r1 == pytest.approx(r2)
    assert L.ishermitian(A) == PL.ishermitian(A)


def test_chebyshev_coefficients():
    import mojo_pyamg.relaxation.chebyshev as ours_cheb
    import pyamg.relaxation.chebyshev as theirs_cheb
    assert np.allclose(ours_cheb.chebyshev_polynomial_coefficients(1.0, 2.0, 3),
                       theirs_cheb.chebyshev_polynomial_coefficients(1.0, 2.0, 3))
    a, r = ours_cheb.mls_polynomial_coefficients(2.0, 2)
    b, s = theirs_cheb.mls_polynomial_coefficients(2.0, 2)
    assert np.allclose(a, b)
    assert np.allclose(r, s)




def test_triangular_solves():
    """`lower_tri_solve` / `upper_tri_solve` against a NumPy reference."""
    rng = np.random.default_rng(4)
    n = 6
    L = np.tril(rng.random((n, n)))
    L[np.diag_indices(n)] += 1.0
    R = np.triu(rng.random((n, n)))
    R[np.diag_indices(n)] += 1.0
    b = rng.random(n)

    for M, solve, name in [(L, lib().mpa_lower_tri_solve, 'lower'),
                           (R, lib().mpa_upper_tri_solve, 'upper')]:
        for col_major in (0, 1):
            Mbuf = np.asfortranarray(M).ravel(order='F') if col_major else \
                np.ascontiguousarray(M).ravel()
            Mx = f64(Mbuf)
            bx = f64(b.copy())
            x = np.zeros(n, dtype=np.float64)
            solve(int(Mx.ctypes.data), int(bx.ctypes.data), int(x.ctypes.data),
                  n, n, col_major)
            assert np.allclose(x, np.linalg.solve(M, b), atol=1e-12), \
                f'{name}, col_major={col_major}'


def test_krylov_helpers():
    """Givens and Householder helpers, against pyamg's compiled amg_core."""
    from pyamg import amg_core
    rng = np.random.default_rng(11)
    n, nrot = 8, 3

    Q = rng.random((nrot, 4))
    v = rng.random(n)
    va, vb = v.copy(), v.copy()
    lib().mpa_apply_givens(int(f64(Q.ravel()).ctypes.data), int(f64(va).ctypes.data),
                           n, nrot)
    amg_core.apply_givens(np.ascontiguousarray(Q), vb, n, nrot)
    assert np.allclose(va, vb)

    B = rng.random((n, 3))
    z = rng.random(n)
    y = rng.random(3)
    start, stop, step = 1, 3, 1

    za, zb = z.copy(), z.copy()
    lib().mpa_apply_householders(int(f64(za).ctypes.data),
                                 int(f64(B).ctypes.data), n, start, stop, step)
    amg_core.apply_householders(zb, np.ascontiguousarray(B), n, start, stop, step)
    assert np.allclose(za, zb)

    ua, ub = z.copy(), z.copy()
    lib().mpa_householder_hornerscheme(int(f64(ua).ctypes.data),
                                       int(f64(B).ctypes.data),
                                       int(f64(y).ctypes.data), n, start, stop,
                                       step)
    amg_core.householder_hornerscheme(ub, np.ascontiguousarray(B),
                                      np.ascontiguousarray(y), n, start, stop,
                                      step)
    assert np.allclose(ua, ub)
