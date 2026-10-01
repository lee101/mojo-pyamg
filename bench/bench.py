"""mojo-pyamg against upstream pyamg on the same data.

Run through `pixi run bench`, never directly: the pixi task holds a
machine-wide flock so concurrent factory jobs cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojo_pyamg as ours            # noqa: E402
import mojo_pyamg.relaxation.relaxation as our_relax   # noqa: E402
import mojo_pyamg.strength as our_strength             # noqa: E402
import mojo_pyamg.util.utils as our_utils              # noqa: E402
import mojo_pyamg.aggregation.aggregate as our_agg     # noqa: E402
import mojo_pyamg.aggregation.tentative as our_tent    # noqa: E402
import pyamg                                          # noqa: E402
import pyamg.relaxation.relaxation as their_relax     # noqa: E402
import pyamg.strength as their_strength               # noqa: E402
import pyamg.util.utils as their_utils                # noqa: E402
import pyamg.aggregation.aggregate as their_agg       # noqa: E402
import pyamg.aggregation.tentative as their_tent      # noqa: E402
from pyamg.gallery import poisson, stencil_grid        # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    """Best of `repeat` wall-clock seconds: the minimum is the least noisy."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def poisson_2d(n):
    return poisson((n, n), format='csr')


def laplacian_3d(n):
    """7-point Laplacian on an n x n x n grid."""
    stencil = np.zeros((3, 3, 3))
    stencil[1, 1, 1] = 6.0
    for off in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0),
                (-1, 0, 0)]:
        stencil[1 + off[0], 1 + off[1], 1 + off[2]] = -1.0
    return stencil_grid(stencil, (n, n, n), format='csr')


def block_bsr(nblocks, blocksize, seed=0):
    rng = np.random.default_rng(seed)
    n = nblocks * blocksize
    per_row = min(3, nblocks)
    indptr = np.arange(0, per_row * (nblocks + 1), per_row, dtype=np.int32)
    indices = np.array([(r + k) % nblocks for r in range(nblocks)
                        for k in range(per_row)], dtype=np.int32)
    blocks = rng.random((indices.size, blocksize, blocksize))
    blocks += blocksize * np.eye(blocksize)
    return sp.bsr_array((blocks, indices, indptr),
                        blocksize=(blocksize, blocksize), shape=(n, n))


CASES = []


def case(name, repeat=5):
    def deco(fn):
        CASES.append((name, fn, repeat))
        return fn
    return deco


A2 = poisson_2d(100)            # 10,000 dofs, 5-point
A3 = laplacian_3d(24)           # 13,824 dofs, 7-point
AB = block_bsr(800, 4)          # 3,200 dofs in 4x4 blocks
b2 = np.ones(A2.shape[0])
C2 = their_strength.symmetric_strength_of_connection(sp.csr_array(A2))
C2o = sp.csr_array(C2)
# built by upstream so the AggOp carries int32 indices, which is what its
# binding expects; our wrappers convert indices to int64 themselves
AGG2 = their_agg.standard_aggregation(sp.csr_array(C2o))[0]
B2 = np.ones((A2.shape[0], 1))


@case('symmetric_strength_of_connection (10k dofs)')
def _():
    return (lambda: our_strength.symmetric_strength_of_connection(A2),
            lambda: their_strength.symmetric_strength_of_connection(
                sp.csr_array(A2)))


@case('classical_strength_of_connection (10k dofs)')
def _():
    return (lambda: our_strength.classical_strength_of_connection(A2, theta=0.25),
            lambda: their_strength.classical_strength_of_connection(
                sp.csr_array(A2), theta=0.25))


@case('standard_aggregation (10k dofs)')
def _():
    return (lambda: our_agg.standard_aggregation(C2o),
            lambda: their_agg.standard_aggregation(
                sp.csr_array(C2o)))


@case('naive_aggregation (10k dofs)')
def _():
    return (lambda: our_agg.naive_aggregation(C2o),
            lambda: their_agg.naive_aggregation(sp.csr_array(C2o)))


@case('fit_candidates (10k dofs, 1 candidate)')
def _():
    return (lambda: our_tent.fit_candidates(AGG2, B2),
            lambda: their_tent.fit_candidates(AGG2, B2))


@case('gauss_seidel x 10 (10k dofs)')
def _():
    def ours_fn():
        x = np.zeros(A2.shape[0])
        our_relax.gauss_seidel(A2, x, b2, iterations=10)
    def theirs_fn():
        x = np.zeros(A2.shape[0])
        their_relax.gauss_seidel(sp.csr_array(A2), x, b2.copy(), iterations=10)
    return ours_fn, theirs_fn


@case('gauss_seidel x 10, 7pt (13.8k dofs)')
def _():
    def ours_fn():
        x = np.zeros(A3.shape[0])
        our_relax.gauss_seidel(A3, x, np.ones(A3.shape[0]), iterations=10)
    def theirs_fn():
        x = np.zeros(A3.shape[0])
        their_relax.gauss_seidel(sp.csr_array(A3), x, np.ones(A3.shape[0]),
                                 iterations=10)
    return ours_fn, theirs_fn


@case('gauss_seidel x 10, symmetric sweep (10k dofs)')
def _():
    def ours_fn():
        x = np.zeros(A2.shape[0])
        our_relax.gauss_seidel(A2, x, b2, iterations=10, sweep='symmetric')
    def theirs_fn():
        x = np.zeros(A2.shape[0])
        their_relax.gauss_seidel(sp.csr_array(A2), x, b2.copy(), iterations=10,
                                 sweep='symmetric')
    return ours_fn, theirs_fn


@case('jacobi x 10 (10k dofs)')
def _():
    def ours_fn():
        x = np.zeros(A2.shape[0])
        our_relax.jacobi(A2, x, b2, iterations=10)
    def theirs_fn():
        x = np.zeros(A2.shape[0])
        their_relax.jacobi(sp.csr_array(A2), x, b2.copy(), iterations=10)
    return ours_fn, theirs_fn


@case('block_gauss_seidel x 4 (3.2k dofs, 4x4 blocks)')
def _():
    bb = np.ones(AB.shape[0])
    def ours_fn():
        x = np.zeros(AB.shape[0])
        our_relax.block_gauss_seidel(AB, x, bb, iterations=4, blocksize=4)
    def theirs_fn():
        x = np.zeros(AB.shape[0])
        their_relax.block_gauss_seidel(sp.bsr_array(AB), x, bb.copy(),
                                       iterations=4, blocksize=4)
    return ours_fn, theirs_fn


@case('gauss_seidel_ne x 4 (10k dofs)')
def _():
    def ours_fn():
        x = np.zeros(A2.shape[0])
        our_relax.gauss_seidel_ne(A2, x, b2, iterations=4)
    def theirs_fn():
        x = np.zeros(A2.shape[0])
        their_relax.gauss_seidel_ne(sp.csr_array(A2), x, b2.copy(), iterations=4)
    return ours_fn, theirs_fn


@case('filter_matrix_rows, theta=0.5 (10k dofs)')
def _():
    return (lambda: our_utils.filter_matrix_rows(A2, 0.5),
            lambda: their_utils.filter_matrix_rows(A2, 0.5))


@case('truncate_rows, 3 per row (10k dofs)')
def _():
    return (lambda: our_utils.truncate_rows(A2, 3),
            lambda: their_utils.truncate_rows(A2, 3))


@case('smoothed_aggregation_solver setup (10k dofs)', repeat=3)
def _():
    return (lambda: ours.smoothed_aggregation_solver(A2),
            lambda: pyamg.smoothed_aggregation_solver(A2))


@case('smoothed_aggregation_solver setup (13.8k dofs, 7pt)', repeat=3)
def _():
    return (lambda: ours.smoothed_aggregation_solver(A3),
            lambda: pyamg.smoothed_aggregation_solver(A3))


@case('V-cycle, 10 cycles (10k dofs)', repeat=3)
def _():
    np.random.seed(0)
    ours_ml = ours.smoothed_aggregation_solver(A2)
    np.random.seed(0)
    theirs_ml = pyamg.smoothed_aggregation_solver(A2)
    return (lambda: ours_ml.solve(b2, tol=1e-10, maxiter=10),
            lambda: theirs_ml.solve(b2, tol=1e-10, maxiter=10))


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    return f'{cpu}, {os.cpu_count()} threads'


def main() -> None:
    print(f'machine: {machine()}')
    print(f'python {platform.python_version()}, numpy {np.__version__}, '
          f'pyamg {pyamg.__version__}')
    print()
    print(f'| case | mojo-pyamg | pyamg | ratio |')
    print(f'| --- | ---: | ---: | ---: |')
    rows = []
    for name, make, repeat in CASES:
        ours_fn, theirs_fn = make()
        a = timeit(ours_fn, repeat)
        c = timeit(theirs_fn, repeat)
        ratio = c / a if a > 0 else math.inf
        flag = '' if ratio >= 1.0 else ' (mojo slower)'
        print(f'| {name} | {a * 1e3:.1f} ms | {c * 1e3:.1f} ms | '
              f'{ratio:.2f}x{flag} |')
        rows.append((name, a, c, ratio))
    slower = [r for r in rows if r[3] < 1.0]
    print()
    if slower:
        print(f'{len(slower)} of {len(rows)} cases are slower than upstream:')
        for name, a, c, ratio in slower:
            print(f'  {name}: {ratio:.2f}x')
    else:
        print('every case is at least as fast as upstream')


if __name__ == '__main__':
    main()
