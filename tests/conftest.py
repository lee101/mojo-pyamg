"""Shared fixtures: the comparison matrices and the upstream package."""

import numpy as np
import pytest
import scipy.sparse as sp

pyamg = pytest.importorskip('pyamg', reason='upstream pyamg is needed for parity')

from pyamg.gallery import poisson, stencil_grid  # noqa: E402


@pytest.fixture(scope='session')
def poisson_1d():
    """1-D 9-point-free Poisson on 10 points, 10 uncoupled copies (110 dofs)."""
    A = sp.csr_array(sp.diags([np.r_[-np.ones(9), 4],
                               np.r_[-np.ones(10), -np.ones(9)],
                               np.r_[-np.ones(9), 4]], [-1, 0, 1]))
    return sp.csr_array(sp.kron(sp.eye(10, format='csr'), A, format='csr'))


@pytest.fixture(scope='session')
def poisson_2d():
    """2-D 5-point Poisson on a 40x40 grid (1600 dofs)."""
    return poisson((40, 40), format='csr')


@pytest.fixture(scope='session')
def random_csr():
    """A random, non-symmetric, diagonally dominant CSR matrix."""
    rng = np.random.default_rng(7)
    n = 512
    A = sp.random(n, n, density=0.05, format='csr', random_state=3)
    A = sp.csr_array(A + sp.eye(n, format='csr') * 5.0)
    _ = rng
    return A


@pytest.fixture(scope='session')
def blocks_3():
    """A BSR matrix of 3x3 blocks, six block rows and three blocks per row.

    Block row r couples to block columns r, r+1 and r+2 (mod 6), so the matrix
    is a block-banded, diagonally dominant, nonsymmetric operator on 18 dofs.
    """
    rng = np.random.default_rng(2)
    nblocks = 6
    per_row = 3
    indptr = np.arange(0, per_row * (nblocks + 1), per_row, dtype=np.int32)
    indices = np.array([(r + k) % nblocks for r in range(nblocks)
                        for k in range(per_row)], dtype=np.int32)
    blocks = rng.random((indices.size, 3, 3)) + 2.0 * np.eye(3)
    A = sp.bsr_array((blocks, indices, indptr), blocksize=(3, 3),
                     shape=(3 * nblocks, 3 * nblocks))
    b = rng.random(3 * nblocks)
    return A, b


@pytest.fixture(scope='session')
def stencil_3d():
    """3-D 7-point Laplacian on a 12x12x12 grid (1728 dofs)."""
    stencil = np.zeros((3, 3, 3))
    stencil[1, 1, 1] = 6.0
    for off in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0),
                (-1, 0, 0)]:
        stencil[1 + off[0], 1 + off[1], 1 + off[2]] = -1.0
    return stencil_grid(stencil, (12, 12, 12), format='csr')
