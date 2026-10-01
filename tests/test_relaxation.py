"""Every relaxation sweep is checked against pyamg on the same data."""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_pyamg.relaxation.relaxation as ours
import pyamg.relaxation.relaxation as theirs
import pyamg.relaxation.smoothing as their_smoothing
import mojo_pyamg.relaxation.smoothing as our_smoothing


def _x(n):
    return np.zeros(n)


@pytest.mark.parametrize('sweep', ['forward', 'backward', 'symmetric'])
def test_gauss_seidel(poisson_2d, sweep):
    A, b = poisson_2d, np.arange(1.0, poisson_2d.shape[0] + 1)
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel(A, x, b, iterations=3, sweep=sweep)
    theirs.gauss_seidel(sp.csr_array(A), y, b.copy(), iterations=3, sweep=sweep)
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_gauss_seidel_omega(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel(A, x, b, iterations=2, omega=0.7)
    theirs.gauss_seidel(sp.csr_array(A), y, b.copy(), iterations=2, omega=0.7)
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_sor(random_csr):
    A = random_csr
    b = np.random.default_rng(0).random(A.shape[0])
    x, y = _x(b.size), _x(b.size)
    ours.sor(A, x, b, omega=0.8, iterations=3)
    theirs.sor(sp.csr_array(A), y, b.copy(), omega=0.8, iterations=3)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('omega', [1.0, 0.85])
def test_jacobi(poisson_2d, omega):
    A = poisson_2d
    b = np.linspace(1.0, 2.0, A.shape[0])
    x, y = _x(b.size), _x(b.size)
    ours.jacobi(A, x, b, iterations=4, omega=omega)
    theirs.jacobi(sp.csr_array(A), y, b.copy(), iterations=4, omega=omega)
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_polynomial(poisson_2d):
    A = poisson_2d
    b = np.ones(A.shape[0])
    coeffs = np.array([0.2, -0.6, 0.4])
    x, y = _x(b.size), _x(b.size)
    ours.polynomial(A, x, b, coeffs, iterations=2)
    theirs.polynomial(sp.csr_array(A), y, b.copy(), coeffs, iterations=2)
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_gauss_seidel_indexed(poisson_2d):
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    indices = np.arange(0, A.shape[0], 3)
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel_indexed(A, x, b, indices, iterations=2, sweep='symmetric')
    theirs.gauss_seidel_indexed(sp.csr_array(A), y, b.copy(), indices,
                                iterations=2, sweep='symmetric')
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_jacobi_indexed(poisson_2d):
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    indices = np.arange(0, A.shape[0], 3)
    x, y = _x(b.size), _x(b.size)
    ours.jacobi_indexed(A, x, b, indices, iterations=2, omega=0.8)
    theirs.jacobi_indexed(sp.csr_array(A), y, b.copy(), indices, iterations=2,
                          omega=0.8)
    assert np.allclose(x, y, rtol=0, atol=1e-13)


def test_jacobi_ne(poisson_2d):
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    x, y = _x(b.size), _x(b.size)
    ours.jacobi_ne(A, x, b, iterations=2)
    theirs.jacobi_ne(sp.csr_array(A), y, b.copy(), iterations=2)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('sweep', ['forward', 'symmetric'])
def test_gauss_seidel_ne(poisson_2d, sweep):
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel_ne(A, x, b, iterations=2, sweep=sweep)
    theirs.gauss_seidel_ne(sp.csr_array(A), y, b.copy(), iterations=2, sweep=sweep)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('sweep', ['forward', 'symmetric'])
def test_gauss_seidel_nr(poisson_2d, sweep):
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel_nr(sp.csc_array(A), x, b, iterations=2, sweep=sweep)
    theirs.gauss_seidel_nr(sp.csc_array(A), y, b.copy(), iterations=2, sweep=sweep)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('sweep', ['forward', 'symmetric'])
def test_bsr_gauss_seidel(blocks_3, sweep):
    A, b = blocks_3
    x, y = _x(b.size), _x(b.size)
    ours.gauss_seidel(A, x, b, iterations=2, sweep=sweep)
    theirs.gauss_seidel(sp.bsr_array(A), y, b.copy(), iterations=2, sweep=sweep)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


def test_bsr_jacobi(blocks_3):
    A, b = blocks_3
    x, y = _x(b.size), _x(b.size)
    ours.jacobi(A, x, b, iterations=2, omega=1.0)
    theirs.jacobi(sp.bsr_array(A), y, b.copy(), iterations=2, omega=1.0)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


def test_bsr_jacobi_indexed(blocks_3):
    A, b = blocks_3
    # indices are block rows: the matrix has 6 block rows
    indices = np.arange(0, 6, 2)
    x, y = _x(b.size), _x(b.size)
    ours.jacobi_indexed(A, x, b, indices, iterations=2, omega=0.9)
    theirs.jacobi_indexed(sp.bsr_array(A), y, b.copy(), indices, iterations=2,
                          omega=0.9)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('fn', ['block_gauss_seidel', 'block_jacobi'])
def test_block_relaxation(blocks_3, fn):
    A, b = blocks_3
    x, y = _x(b.size), _x(b.size)
    getattr(ours, fn)(A, x, b, iterations=2, blocksize=3)
    getattr(theirs, fn)(sp.bsr_array(A), y, b.copy(), iterations=2, blocksize=3)
    assert np.allclose(x, y, rtol=0, atol=1e-11)


def test_block_relaxation_reuses_Dinv(blocks_3):
    """A caller-supplied Dinv is honoured, not recomputed."""
    from mojo_pyamg.util.utils import get_block_diag
    A, b = blocks_3
    Dinv = get_block_diag(A, blocksize=3, inv_flag=True)
    x, y = _x(b.size), _x(b.size)
    ours.block_gauss_seidel(A, x, b, iterations=1, blocksize=3, Dinv=Dinv)
    theirs.block_gauss_seidel(sp.bsr_array(A), y, b.copy(), iterations=1,
                              blocksize=3, Dinv=Dinv)
    assert np.allclose(x, y, rtol=0, atol=1e-12)


@pytest.mark.parametrize('name', ['gauss_seidel', 'jacobi', 'block_gauss_seidel',
                                  'richardson', 'sor', 'chebyshev', 'none'])
def test_setup_smoothers(poisson_2d, name):
    """`setup_*` builds a smoother that matches pyamg's on the same level."""
    A = poisson_2d
    b = np.arange(1.0, A.shape[0] + 1)

    ours_lvl, their_lvl = _level(A)
    # the polynomial smoothers scale by approximate_spectral_radius, which
    # starts from a random guess in both packages: seed it identically
    np.random.seed(1234)
    oursmooth = our_smoothing._setup_call(name)(ours_lvl)
    np.random.seed(1234)
    theirsmooth = their_smoothing._setup_call(name)(their_lvl)
    assert oursmooth.__name__ == theirsmooth.__name__

    x, y = _x(b.size), _x(b.size)
    oursmooth(A, x, b)
    theirsmooth(sp.csr_array(A), y, b.copy())
    assert np.allclose(x, y, rtol=1e-6, atol=1e-12)


def _level(A):
    """A level object for each package, holding the same matrix."""
    from mojo_pyamg.multilevel import MultilevelSolver
    from pyamg.multilevel import MultilevelSolver as TheirMultilevel
    lvl = MultilevelSolver.Level()
    lvl.A = A
    tlvl = TheirMultilevel.Level()
    tlvl.A = sp.csr_array(A)
    return lvl, tlvl


def test_complex_input_is_rejected(poisson_2d):
    """The kernels are real-only, so complex input raises rather than casting."""
    A = sp.csr_array(poisson_2d.astype(np.complex128))
    b = np.ones(A.shape[0], dtype=np.complex128)
    with pytest.raises(TypeError):
        ours.gauss_seidel(A, np.zeros(A.shape[0], dtype=np.complex128), b)
    with pytest.raises(TypeError):
        from mojo_pyamg.util.utils import asfptype
        asfptype(A)
