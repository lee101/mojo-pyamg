"""The Krylov solvers, compared against pyamg on the same systems."""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_pyamg.krylov as ours
import pyamg.krylov as theirs

SOLVERS = ['cg', 'cgnr', 'cgne', 'cr', 'bicgstab', 'steepest_descent',
           'minimal_residual', 'gmres', 'gmres_mgs', 'gmres_householder',
           'fgmres']


def _run(fn, A, b, **kw):
    res = []
    x, flag = fn(A, b, tol=1e-10, maxiter=kw.pop('maxiter', 50),
                 residuals=res, **kw)
    return x, flag, np.array(res)


@pytest.mark.parametrize('name', SOLVERS)
def test_symmetric_system(poisson_2d, name):
    """Same iteration count, same solution, on a symmetric system."""
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    xa, fa, ra = _run(getattr(ours, name), A, b)
    xb, fb, rb = _run(getattr(theirs, name), sp.csr_array(A), b)
    assert fa == fb
    assert len(ra) == len(rb)
    assert np.allclose(ra, rb, rtol=1e-8, atol=1e-12)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('name', ['cg', 'cgnr', 'cr', 'gmres', 'fgmres',
                                  'bicgstab', 'cgne'])
def test_nonsymmetric_system(random_csr, name):
    """A nonsymmetric system: only the general-purpose methods are compared."""
    A = random_csr
    b = np.arange(1.0, A.shape[0] + 1)
    xa, fa, ra = _run(getattr(ours, name), A, b, maxiter=200)
    xb, fb, rb = _run(getattr(theirs, name), sp.csr_array(A), b, maxiter=200)
    assert fa == fb
    assert len(ra) == len(rb)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


def test_gmres_restart(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    xa, fa, ra = _run(ours.gmres, A, b, restart=5)
    xb, fb, rb = _run(theirs.gmres, sp.csr_array(A), b, restart=5)
    assert fa == fb
    assert np.allclose(ra, rb, rtol=1e-8, atol=1e-12)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


def test_preconditioned(poisson_2d):
    """A Jacobi-preconditioned solve takes the same path on both sides."""
    A = poisson_2d
    b = np.ones(A.shape[0])
    M = sp.diags(1.0 / A.diagonal()).tocsr()
    xa, fa, _ = _run(ours.cg, A, b, M=M)
    xb, fb, _ = _run(theirs.cg, sp.csr_array(A), b, M=sp.csr_array(M))
    assert fa == fb
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


def test_callback_and_initial_guess(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    x0 = np.full(b.size, 0.1)
    seen = []
    xa, fa, _ = _run(ours.cg, A, b, x0=x0, callback=lambda x: seen.append(1))
    seen_b = []
    xb, fb, _ = _run(theirs.cg, sp.csr_array(A), b, x0=x0,
                     callback=lambda x: seen_b.append(1))
    assert fa == fb
    assert len(seen) == len(seen_b)
    assert np.allclose(xa, xb, rtol=1e-6, atol=1e-8)


def test_converges_to_the_exact_solution(stencil_3d):
    """The port's solvers actually solve the system, not just agree.

    The plain projection methods (steepest descent, minimal residual) are
    excluded: they are far too slow to converge on this system in any
    reasonable number of iterations, in either package.
    """
    A = stencil_3d
    xtrue = np.random.default_rng(0).random(A.shape[0])
    b = A @ xtrue
    for name in ['cg', 'gmres', 'cgnr', 'cr', 'cgne']:
        x, flag = getattr(ours, name)(A, b, tol=1e-10, maxiter=200)
        assert flag == 0, name
        assert np.linalg.norm(b - A @ x) / np.linalg.norm(b) < 1e-8, name
