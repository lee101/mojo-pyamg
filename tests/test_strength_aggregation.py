"""Strength of connection, aggregation, tentative prolongation and smoothing."""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_pyamg.strength as ours
from mojo_pyamg._lib import addr, i64, lib
import pyamg.strength as theirs
import mojo_pyamg.aggregation.aggregate as our_agg
import pyamg.aggregation.aggregate as their_agg
import mojo_pyamg.aggregation.tentative as our_tent
import pyamg.aggregation.tentative as their_tent
import mojo_pyamg.aggregation.smooth as our_smooth
import pyamg.aggregation.smooth as their_smooth


# --------------------------------------------------------------- strength

@pytest.mark.parametrize('theta', [0.0, 0.25, 0.5])
@pytest.mark.parametrize('norm', ['abs', 'min', 'fro'])
def test_classical_strength(poisson_2d, theta, norm):
    A = poisson_2d
    np.random.seed(0)
    a = ours.classical_strength_of_connection(A, theta=theta, norm=norm)
    b = theirs.classical_strength_of_connection(sp.csr_array(A), theta=theta,
                                                norm=norm)
    assert a.shape == b.shape
    assert (a != 0).nnz == (b != 0).nnz
    assert np.allclose(a.toarray(), b.toarray())


def test_classical_strength_blocks(blocks_3):
    A, _ = blocks_3
    a = ours.classical_strength_of_connection(A, theta=0.3, block=True)
    b = theirs.classical_strength_of_connection(sp.bsr_array(A), theta=0.3,
                                                block=True)
    assert np.allclose(a.toarray(), b.toarray())
    a = ours.classical_strength_of_connection(A, theta=0.3, block=False)
    b = theirs.classical_strength_of_connection(sp.bsr_array(A), theta=0.3,
                                                block=False)
    assert np.allclose(a.toarray(), b.toarray())


@pytest.mark.parametrize('theta', [0.0, 0.3])
def test_symmetric_strength(poisson_2d, theta):
    A = poisson_2d
    a = ours.symmetric_strength_of_connection(A, theta=theta)
    b = theirs.symmetric_strength_of_connection(sp.csr_array(A), theta=theta)
    assert np.allclose(a.toarray(), b.toarray())


def test_symmetric_strength_bsr(blocks_3):
    A, _ = blocks_3
    a = ours.symmetric_strength_of_connection(A, theta=0.2)
    b = theirs.symmetric_strength_of_connection(sp.bsr_array(A), theta=0.2)
    assert np.allclose(a.toarray(), b.toarray())


def test_energy_based_strength(poisson_2d):
    A = poisson_2d
    np.random.seed(0)
    a = ours.energy_based_strength_of_connection(A, theta=0.05, k=2)
    np.random.seed(0)
    b = theirs.energy_based_strength_of_connection(sp.csr_array(A), theta=0.05, k=2)
    assert np.allclose(a.toarray(), b.toarray())


def test_classical_strength_rejects_bad_theta(poisson_2d):
    with pytest.raises(ValueError):
        ours.classical_strength_of_connection(poisson_2d, theta=1.5)


# ------------------------------------------------------------- aggregation

@pytest.mark.parametrize('fn', ['standard_aggregation', 'naive_aggregation'])
def test_aggregation_matches_upstream(poisson_2d, fn):
    A = poisson_2d
    np.random.seed(0)
    C = ours.symmetric_strength_of_connection(A)
    C = sp.csr_array(C)
    np.random.seed(0)
    Ct = theirs.symmetric_strength_of_connection(sp.csr_array(A))
    Ct = sp.csr_array(Ct)

    a, acpts = getattr(our_agg, fn)(C)
    b, bcpts = getattr(their_agg, fn)(Ct)
    assert a.shape == b.shape
    assert np.array_equal(acpts, bcpts)
    assert np.array_equal(a.toarray(), b.toarray())


def _reference_pairwise_aggregation(S):
    """`amg_core::pairwise_aggregation`, transcribed from the upstream header.

    The nodes live in a std::multimap keyed on (m_i, i); `insert` places an
    element at the *upper* bound of the equivalent range, so equal keys keep
    insertion order.  The list below is that container: it is kept sorted by
    key alone and appended to within a key.
    """
    n = S.shape[0]
    x = [0] * n
    y = [0] * n
    m = [0] * n
    for i in range(n):
        for j in S.indices[S.indptr[i]:S.indptr[i + 1]]:
            if j != i:
                m[j] += 1
    table = []                      # the multimap, sorted by key

    def insert(node, key):
        pos = 0
        while pos < len(table) and table[pos][0] <= key:
            pos += 1
        table.insert(pos, (key, node))

    def erase(node):
        for pos, (key, other) in enumerate(table):
            if other == node:
                del table[pos]
                return

    def key_of(node):
        for key, other in table:
            if other == node:
                return key
        raise KeyError(node)

    for i in range(n):
        insert(i, m[i])

    next_aggregate = 1
    while table:
        i = table[0][1]
        erase(i)
        row = range(S.indptr[i], S.indptr[i + 1])
        j, found, max_val = 0, False, -np.inf
        x[i] = next_aggregate
        for jj in row:
            if x[S.indices[jj]] == 0 and S.data[jj] >= max_val:
                max_val, j, found = S.data[jj], S.indices[jj], True
        if found:
            x[j] = next_aggregate
        y[next_aggregate - 1] = i
        for jj in row:
            if x[S.indices[jj]] == 0:
                erase(S.indices[jj])
                insert(S.indices[jj], key_of(S.indices[jj]) - 1
                       if False else m[S.indices[jj]] - 1)
                m[S.indices[jj]] -= 1
        if found:
            x[j] = next_aggregate
            for jj in range(S.indptr[j], S.indptr[j + 1]):
                if x[S.indices[jj]] == 0:
                    erase(S.indices[jj])
                    m[S.indices[jj]] -= 1
                    insert(S.indices[jj], m[S.indices[jj]])
            erase(j)
        next_aggregate += 1

    return np.array(x), np.array(y[:next_aggregate - 1]), next_aggregate - 1


def test_pairwise_aggregation_kernel(poisson_2d):
    """The ported kernel matches a transcription of the upstream algorithm."""
    A = poisson_2d
    C = sp.csr_array(ours.classical_strength_of_connection(A, theta=0.25,
                                                            norm='min'))
    C.sort_indices()
    n = C.shape[0]
    Tj = np.zeros(n, dtype=np.int64)
    Cpts = np.zeros(n, dtype=np.int64)
    scratch = our_agg._pairwise_scratch(n)
    Ap = i64(C.indptr)
    Aj = i64(C.indices)
    Sx = np.ascontiguousarray(C.data, dtype=np.float64)
    k = lib().mpa_pairwise_aggregation(n, addr(Ap), addr(Aj), addr(Sx), addr(Tj),
                                       addr(Cpts), addr(scratch))
    ref_Tj, ref_Cpts, ref_k = _reference_pairwise_aggregation(C)
    assert k == ref_k
    assert np.array_equal(Tj, ref_Tj)
    assert np.array_equal(Cpts[:k], ref_Cpts)


def test_pairwise_aggregation_wrapper(poisson_2d):
    """The wrapper builds a valid pairwise aggregation and a valid matrix."""
    A = poisson_2d
    T, Cpts = our_agg.pairwise_aggregation(A, matchings=1, theta=0.25)
    assert T.shape[0] == A.shape[0]
    assert T.nnz > 0
    # every node lands in exactly one aggregate
    assert np.all(np.asarray(T.sum(axis=1)).ravel() == 1)
    assert len(Cpts) == T.shape[1]


# ------------------------------------------------------- tentative prolongation

def test_fit_candidates_doc_example():
    """The example from upstream's docstring."""
    AggOp = sp.csr_array([[1, 0], [1, 0], [0, 1], [0, 1]])
    B = np.array([[1.0], [1.0], [1.0], [1.0]])
    Q, R = our_tent.fit_candidates(AggOp, B)
    Qt, Rt = their_tent.fit_candidates(sp.csr_array(AggOp), B)
    assert np.allclose(Q.toarray(), Qt.toarray())
    assert np.allclose(R, Rt)
    expected = np.array([[0.70710678, 0.0],
                         [0.70710678, 0.0],
                         [0.0, 0.70710678],
                         [0.0, 0.70710678]])
    assert np.allclose(Q.toarray(), expected)


def test_fit_candidates_two_candidates():
    """Two candidates per aggregate, the classical SA setup."""
    AggOp = sp.csr_array([[1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 0, 1]])
    B = np.array([[1.0, 0.0],
                  [1.0, 1.0],
                  [1.0, 2.0],
                  [1.0, 3.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])
    Q, R = our_tent.fit_candidates(AggOp, B)
    Qt, Rt = their_tent.fit_candidates(sp.csr_array(AggOp), B)
    assert Q.shape == Qt.shape
    assert np.allclose(Q.toarray(), Qt.toarray())
    assert np.allclose(R, Rt)
    # the fit is exact: B = Q R
    assert np.allclose(Q.toarray() @ R, B)


# --------------------------------------------------------- prolongation smoothing

def test_richardson_prolongation_smoother(poisson_2d):
    A = poisson_2d
    C = sp.csr_array(ours.symmetric_strength_of_connection(A))
    AggOp, _ = our_agg.standard_aggregation(C)
    T, B = our_tent.fit_candidates(AggOp, np.ones((A.shape[0], 1)))
    # a fresh matrix on each side: approximate_spectral_radius caches its
    # estimate on the matrix it is given
    np.random.seed(5)
    P = our_smooth.richardson_prolongation_smoother(sp.csr_array(A), T)
    np.random.seed(5)
    Pt = their_smooth.richardson_prolongation_smoother(sp.csr_array(A),
                                                      sp.csr_array(T))
    assert np.allclose(P.toarray(), Pt.toarray())


def test_jacobi_prolongation_smoother(poisson_2d):
    A = poisson_2d
    C = sp.csr_array(ours.symmetric_strength_of_connection(A))
    AggOp, _ = our_agg.standard_aggregation(C)
    T, B = our_tent.fit_candidates(AggOp, np.ones((A.shape[0], 1)))
    np.random.seed(5)
    P = our_smooth.jacobi_prolongation_smoother(sp.csr_array(A), T, C, B)
    np.random.seed(5)
    Pt = their_smooth.jacobi_prolongation_smoother(sp.csr_array(A),
                                                   sp.csr_array(T), C, B)
    assert np.allclose(P.toarray(), Pt.toarray())


def test_satisfy_constraints(poisson_2d):
    """U is projected so that U @ B is (numerically) zero, as upstream does."""
    A = poisson_2d
    C = sp.csr_array(ours.symmetric_strength_of_connection(A))
    AggOp, _ = our_agg.standard_aggregation(C)
    T, B = our_tent.fit_candidates(AggOp, np.ones((A.shape[0], 1)))
    U = sp.bsr_array(sp.csr_array(T).tobsr())
    U.data[:] = np.random.default_rng(0).random(U.data.shape)

    from mojo_pyamg.util.utils import compute_BtBinv
    BtBinv = compute_BtBinv(B, U)
    out = our_smooth.satisfy_constraints(U, B, BtBinv)
    assert np.allclose(np.asarray(out @ B), 0.0, atol=1e-10)
