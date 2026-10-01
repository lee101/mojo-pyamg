"""End-to-end: the smoothed aggregation hierarchy and the multigrid cycle."""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_pyamg as ours
import pyamg as theirs


def _solver(module, A, **kw):
    # approximate_spectral_radius draws a random starting vector, so seed
    # both packages identically before building the hierarchy
    np.random.seed(4321)
    return module.smoothed_aggregation_solver(A, **kw)


def test_hierarchy_structure(poisson_2d):
    """Same number of levels, same unknowns, nonzeros and operators per level.

    The operators agree to round-off, not bit for bit: the Mojo kernels sum
    their rows in a different order than upstream's compiled ones, so the two
    hierarchies differ in the last bits and the difference grows through the
    levels.
    """
    A = poisson_2d
    mine = _solver(ours, A, keep=True)
    theirs_ = _solver(theirs, A, keep=True)

    assert len(mine.levels) == len(theirs_.levels)
    for lvl, (lm, lt) in enumerate(zip(mine.levels, theirs_.levels)):
        assert lm.A.shape == lt.A.shape
        assert lm.A.nnz == lt.A.nnz
        assert np.allclose(lm.A.toarray(), lt.A.toarray(), atol=1e-10)
        if lvl == len(mine.levels) - 1:
            continue          # the coarsest level has no P or R
        assert np.allclose(lm.P.toarray(), lt.P.toarray(), atol=1e-10)
        assert np.allclose(lm.R.toarray(), lt.R.toarray(), atol=1e-10)
        assert np.allclose(lm.C.toarray(), lt.C.toarray(), atol=1e-10)
        assert np.array_equal(lm.AggOp.toarray(), lt.AggOp.toarray())
        assert np.array_equal(lm.Cnodes, lt.Cnodes)
    assert mine.operator_complexity() == pytest.approx(
        theirs_.operator_complexity())
    assert mine.grid_complexity() == pytest.approx(theirs_.grid_complexity())
    assert mine.cycle_complexity() == pytest.approx(theirs_.cycle_complexity())


def test_v_cycle_residuals(poisson_2d):
    """The same residual history, iteration for iteration."""
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)

    ra, rb = [], []
    xa = mine.solve(b, tol=1e-12, residuals=ra)
    xb = theirs_.solve(b, tol=1e-12, residuals=rb)

    assert len(ra) == len(rb)          # same number of cycles
    assert np.allclose(ra, rb, rtol=1e-2)
    assert np.allclose(xa, xb, rtol=1e-4, atol=1e-8)


def test_solves_the_system(stencil_3d):
    """The port converges on a 3-D problem, at a similar rate."""
    A = stencil_3d
    xtrue = np.random.default_rng(1).random(A.shape[0])
    b = A @ xtrue

    mine = _solver(ours, A, max_coarse=400)
    theirs_ = _solver(theirs, A, max_coarse=400)

    ra, rb = [], []
    xa = mine.solve(b, tol=1e-10, maxiter=50, residuals=ra)
    xb = theirs_.solve(b, tol=1e-10, maxiter=50, residuals=rb)

    assert np.linalg.norm(b - A @ xa) / np.linalg.norm(b) < 1e-8
    assert len(ra) == len(rb)
    assert np.allclose(ra, rb, rtol=1e-2)


@pytest.mark.parametrize('cycle', ['V', 'W', 'F'])
def test_cycles_agree(poisson_2d, cycle):
    A = poisson_2d
    b = np.ones(A.shape[0])
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)
    ra, rb = [], []
    mine.solve(b, tol=1e-12, maxiter=3, cycle=cycle, residuals=ra)
    theirs_.solve(b, tol=1e-12, maxiter=3, cycle=cycle, residuals=rb)
    assert np.allclose(ra, rb, rtol=1e-2)


def test_aspreconditioner(poisson_2d):
    from scipy.sparse.linalg import cg as scipy_cg
    A = poisson_2d
    b = np.ones(A.shape[0])
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)
    M = mine.aspreconditioner(cycle='V')
    Mt = theirs_.aspreconditioner(cycle='V')
    x, info = scipy_cg(A, b, rtol=1e-10, maxiter=30, M=M)
    xt, infot = scipy_cg(sp.csr_array(A), b, rtol=1e-10, maxiter=30, M=Mt)
    assert info == infot
    assert np.allclose(x, xt, rtol=1e-6, atol=1e-8)


def test_strength_and_aggregate_options(poisson_2d):
    """Non-default strength/aggregate/smooth choices still build and solve."""
    A = poisson_2d
    b = np.ones(A.shape[0])
    for kwargs in [
        {'strength': 'classical'},
        {'strength': ('symmetric', {'theta': 0.25})},
        {'aggregate': 'naive'},
        {'smooth': ('richardson', {'omega': 4.0 / 3.0})},
        {'smooth': None},
        {'presmoother': ('gauss_seidel', {'sweep': 'symmetric'}),
         'postsmoother': ('gauss_seidel', {'sweep': 'symmetric'})},
    ]:
        mine = _solver(ours, A, **kwargs)
        theirs_ = _solver(theirs, A, **kwargs)
        assert len(mine.levels) == len(theirs_.levels), kwargs
        ra, rb = [], []
        mine.solve(b, tol=1e-10, maxiter=30, residuals=ra)
        theirs_.solve(b, tol=1e-10, maxiter=30, residuals=rb)
        assert len(ra) == len(rb), kwargs
        assert np.allclose(ra, rb, rtol=1e-2), kwargs


def test_multigrid_repr_and_complexities(poisson_2d):
    mine = _solver(ours, A=poisson_2d)
    theirs_ = _solver(theirs, A=poisson_2d)
    ours_lines = [line.split() for line in repr(mine).splitlines()]
    their_lines = [line.split() for line in repr(theirs_).splitlines()]
    assert [ln[:2] for ln in ours_lines[:5]] == [ln[:2] for ln in their_lines[:5]]
    assert ours_lines[6:] == their_lines[6:]


def test_psolve_is_one_cycle(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)
    xa = mine.psolve(b)
    xb = theirs_.psolve(b)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


def test_krylov_acceleration(poisson_2d):
    """cg-accelerated AMG takes the same iterations on both sides."""
    from scipy.sparse.linalg import LinearOperator
    A = poisson_2d
    b = np.ones(A.shape[0])
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)
    ra, rb = [], []
    mine.solve(b, tol=1e-10, maxiter=20, accel='cg', residuals=ra)
    theirs_.solve(b, tol=1e-10, maxiter=20, accel='cg', residuals=rb)
    assert np.allclose(ra, rb, rtol=1e-2)
    _ = LinearOperator


def test_coarse_solvers(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    for name in ['pinv', 'lu', 'cholesky', 'splu', 'gauss_seidel', 'cg', None]:
        mine = _solver(ours, A, coarse_solver=name)
        theirs_ = _solver(theirs, A, coarse_solver=name)
        assert mine.coarse_solver.name() == theirs_.coarse_solver.name()
        xa = mine.solve(b, tol=1e-8, maxiter=50)
        xb = theirs_.solve(b, tol=1e-8, maxiter=50)
        assert np.allclose(xa, xb, rtol=1e-4, atol=1e-8), name


def test_change_solve_matrix(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    mine = _solver(ours, A)
    theirs_ = _solver(theirs, A)
    mine.change_solve_matrix(A)
    theirs_.change_solve_matrix(A)
    xa = mine.solve(b, tol=1e-10)
    xb = theirs_.solve(b, tol=1e-10)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)
