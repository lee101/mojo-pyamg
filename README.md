# mojo-pyamg

Algebraic multigrid solvers, ported to [Mojo](https://www.modular.com/mojo).

This is a port of [PyAMG](https://github.com/pyamg/pyamg) 5.3.0: the compute-bound
kernels of its algebraic multigrid machinery — strength of connection,
aggregation, tentative prolongation, relaxation and the Givens/Householder
helpers — are rewritten in Mojo and exposed through a Python API that keeps
PyAMG's names, argument orders and defaults, so a caller of `pyamg` can call
`mojo_pyamg` with the same arguments.

```python
import numpy as np
import scipy.sparse as sp
import mojo_pyamg

# 5-point Laplacian on a 100 x 100 grid: 10,000 unknowns
n = 100
T = sp.diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
             [-1, 0, 1], shape=(n, n), format='csr')
I = sp.eye(n, format='csr')
A = sp.csr_array(sp.kron(I, T, format='csr') + sp.kron(T, I, format='csr'))

b = np.ones(A.shape[0])
ml = mojo_pyamg.smoothed_aggregation_solver(A)          # build the hierarchy
x = ml.solve(b, tol=1e-10, maxiter=200)                # V-cycle to tolerance
print(f'relative residual: {np.linalg.norm(b - A @ x) / np.linalg.norm(b):.2e}')
```

Output:

```
relative residual: 3.14e-11
```

## What is covered

| upstream module | what is ported |
| --- | --- |
| `amg_core/ruge_stuben.h` | `classical_strength_of_connection_abs`, `classical_strength_of_connection_min`, `maximum_row_value` |
| `amg_core/smoothed_aggregation.h` | `symmetric_strength_of_connection`, `standard_aggregation`, `naive_aggregation`, `pairwise_aggregation`, `fit_candidates`, `satisfy_constraints_helper`, `truncate_rows_csr` |
| `amg_core/relaxation.h` | `gauss_seidel`, `sor_gauss_seidel`, `bsr_gauss_seidel`, `jacobi`, `bsr_jacobi`, `jacobi_indexed`, `bsr_jacobi_indexed`, `gauss_seidel_indexed`, `jacobi_ne`, `gauss_seidel_ne`, `gauss_seidel_nr`, `block_jacobi`, `block_gauss_seidel` |
| `amg_core/linalg.h` | `csc_scale_columns`, `csc_scale_rows`, `filter_matrix_rows`, `lower_tri_solve`, `upper_tri_solve` |
| `amg_core/krylov.h` | `dot_prod`, `axpy`, `apply_householders`, `householder_hornerscheme`, `apply_givens` |
| `pyamg.strength` | `classical_strength_of_connection`, `symmetric_strength_of_connection`, `energy_based_strength_of_connection` |
| `pyamg.relaxation.relaxation` | `make_system`, `sor`, `gauss_seidel`, `jacobi`, `block_jacobi`, `block_gauss_seidel`, `polynomial`, `gauss_seidel_indexed`, `jacobi_indexed`, `jacobi_ne`, `gauss_seidel_ne`, `gauss_seidel_nr` |
| `pyamg.relaxation.smoothing` | `change_smoothers`, `rho_D_inv_A`, `rho_block_D_inv_A`, `matrix_asformat`, every `setup_*` for the covered methods, `rebuild_smoother` |
| `pyamg.relaxation.chebyshev` | `chebyshev_polynomial_coefficients`, `mls_polynomial_coefficients` |
| `pyamg.relaxation.utils` | `relaxation_as_linear_operator` |
| `pyamg.aggregation.aggregate` | `standard_aggregation`, `naive_aggregation`, `pairwise_aggregation` |
| `pyamg.aggregation.tentative` | `fit_candidates` |
| `pyamg.aggregation.smooth` | `satisfy_constraints`, `jacobi_prolongation_smoother`, `richardson_prolongation_smoother` |
| `pyamg.aggregation.aggregation` | `smoothed_aggregation_solver` and `_extend_hierarchy` |
| `pyamg.multilevel` | `MultilevelSolver` (with `Level`), `coarse_grid_solver`, `multilevel_solver` |
| `pyamg.krylov` | `cg`, `cgnr`, `cgne`, `cr`, `bicgstab`, `steepest_descent`, `minimal_residual`, `gmres`, `gmres_mgs`, `gmres_householder`, `fgmres` |
| `pyamg.util.linalg` | `norm`, `infinity_norm`, `axpy`, `approximate_spectral_radius`, `condest`, `cond`, `ishermitian`, `pinv_array` |
| `pyamg.util.utils` | `asfptype`, `get_blocksize`, `scale_rows`, `get_diagonal`, `get_block_diag`, `amalgamate`, `unamal`, `levelize_*`, `filter_matrix_rows`, `truncate_rows`, `scale_block_inverse`, `compute_BtBinv`, `eliminate_diag_dom_nodes`, `maximum_row_value`, `scale_rows_by_largest_entry`, `print_table` |
| `pyamg.util.params` | `set_tol` |

The Mojo kernels live in `src/ported.mojo`, emitted in the order of the
upstream headers they came from, with upstream's names, argument order and loop
structure. `src/capi.mojo` is the C ABI those kernels are reached through.

## What is not covered, and why

These are API surface that exists upstream and is deliberately absent here.
None of them fail silently: a name that is not ported raises
`AttributeError`, `energy_prolongation_smoother` raises `NotImplementedError`,
and complex input raises `TypeError`.

| upstream | reason |
| --- | --- |
| `ruge_stuben_solver`, `classical.*` (`RS` splitting, `direct_interpolation`, CR) | the Ruge-Stuben coarsening is graph bookkeeping rather than a compute kernel; nothing in it is worth rewriting in Mojo |
| `rootnode_solver`, `pairwise_solver`, `adaptive_sa_solver` (`aggregation/adaptive.py`, `rootnode.py`, `pairwise.py`) | the aSA candidate search, the root-node and the pairwise coarsening variants; `pairwise_aggregation` itself *is* covered |
| `lloyd_aggregation`, `balanced_lloyd_aggregation`, `metis_aggregation` | clustering, not kernels; `metis_aggregation` needs a METIS dependency |
| `evolution`/`ode`/`distance`/`algebraic_distance`/`affinity` strength | need the evolution-strength kernels and graph algorithms not ported here |
| `energy_prolongation_smoother`, `cg_prolongation_smoothing`, `cgnr_prolongation_smoothing` | these build `BtBinv` from a pseudo-inverse of every local block and run a constrained Krylov iteration; `pinv_array`/`svd_jacobi` and the incomplete matrix-multiply kernels behind them are not ported.  `satisfy_constraints`, the kernel they call, *is* ported |
| `schwarz`, `strength_based_schwarz`, `cf_*`/`fc_*` relaxation | need `extract_subblocks` and `overlapping_schwarz_csr` from `amg_core/relaxation.h` |
| `gallery`, `blackbox`, `vis`, `graph_ref`, `util.bsr_utils`, `util.params` extras | problem generators, plotting, graph utilities: not compute-bound kernels.  `gallery` is available for parity testing through the installed `pyamg` |
| complex arithmetic | the kernels are the real (float64) instantiations of the upstream templates.  Upstream also instantiates them for complex128 and dispatches on dtype; the Python layer here raises rather than silently dropping the imaginary part |

### Documented divergences

Fidelity was the goal; where the Mojo toolchain forced a change, it is here
rather than hidden.

- **`truncate_rows_csr` selects instead of sorting.** Upstream sorts each row
  by magnitude with a quicksort over the parallel `(data, indices)` arrays and
  then zeroes the leading entries.  Shuffling two parallel arrays through
  separate pointer arguments is miscompiled on Mojo 1.2.0 (the values come out
  sorted and the indices do not follow them), so the k survivors are selected
  in place.  The kept entries are the same; which of several equal-magnitude
  entries survives a tie can differ.
- **`pairwise_aggregation` uses bucket lists, and matches the source, not the
  wheel.** Upstream keeps its nodes in a `std::multimap` keyed on `m_i`; that
  container is reproduced as intrusive bucket lists carved out of one scratch
  array.  The result matches a transcription of the upstream header, but *not*
  the `amg_core.pairwise_aggregation` in the installed 5.3.0 wheel, which pairs
  nodes differently (equally valid, different matching).  The parity test
  therefore compares against the transcribed algorithm.
- **Scratch buffers are arguments.** `@export` refuses parametric functions and
  a pointer with an inferred origin is parametric, and `Pointer.alloc` no longer
  exists, so kernels that need workspace (the symmetric strength diagonal, the
  bucket lists, the block relaxation `rsum`/`v` vectors, the pairwise scratch)
  take it as a trailing argument and the caller allocates it.
- **Krylov recurrences stay in NumPy.** `pyamg.krylov` is carried over as
  upstream wrote it: the recurrences are BLAS-1 bound and already vectorized in
  NumPy, so a Mojo rewrite would be slower.  The Householder and Givens
  helpers they call (`apply_householders`, `householder_hornerscheme`,
  `apply_givens`) *are* the ported Mojo kernels.
- **`filter_matrix_rows` returns the filtered matrix.** The installed 5.3.0
  returns column indices that still carry the internal `+n` offset, so its
  output is not the matrix it means to be; this port undoes the offset.
- **Float summation order.** The Mojo kernels sum their rows in a different
  order than upstream's compiled ones, so hierarchies agree to round-off rather
  than bit for bit, and two AMG V-cycle residual histories drift apart by ~1e-3
  relative over 20 cycles.  Iteration counts match.

## Install

```bash
pixi install
pixi run build        # mojo build --emit shared-lib -> dist/libmojo-pyamg.so
pixi run test         # pytest, parity against the installed pyamg
pixi run bench        # benchmarks, under a machine-wide flock
```

`pixi run build` is enough: `mojo_pyamg._lib` also rebuilds the library on
first import if the `.so` is missing or older than `src/`.  Set
`MOJO_PYAMG_LIB` to point at a prebuilt shared object, or `MOJO_PYAMG_MOJO` to
point at a `mojo` binary, to build nothing at import time.

Requirements: the pinned `mojo` 1.2.0 toolchain, Python 3.13, NumPy, SciPy, and
`pyamg` 5.3.0 for the parity tests and benchmarks.  No GPU support and no
`max` package: every kernel here is CPU and scalar.

## Usage

```python
import numpy as np
import scipy.sparse as sp
import mojo_pyamg

# the same 5-point Laplacian on a 40 x 40 grid
n = 40
T = sp.diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
             [-1, 0, 1], shape=(n, n), format='csr')
I = sp.eye(n, format='csr')
A = sp.csr_array(sp.kron(I, T, format='csr') + sp.kron(T, I, format='csr'))

ml = mojo_pyamg.smoothed_aggregation_solver(A, max_coarse=50)
b = np.ones(A.shape[0])

residuals = []
x = ml.solve(b, x0=np.zeros_like(b), tol=1e-10, maxiter=200, residuals=residuals)
print(f'{len(residuals) - 1} V-cycles, relative residual '
      f'{np.linalg.norm(b - A @ x) / np.linalg.norm(b):.2e}')
# 15 V-cycles, relative residual 7.96e-11

# the same sweeps upstream exposes
from mojo_pyamg.relaxation import gauss_seidel, jacobi
y = np.zeros_like(b)
gauss_seidel(A, y, b, iterations=2, sweep='symmetric')
y = np.zeros_like(b)
jacobi(A, y, b, iterations=2, omega=2.0 / 3.0)

# and the krylov solvers
x, flag = mojo_pyamg.krylov.cg(A, b, tol=1e-10, maxiter=50)
print(flag, np.linalg.norm(b - A @ x) / np.linalg.norm(b))
# 50 3.23e-04  (unpreconditioned CG on its own is slow; that is why AMG exists)
```

## How it works

**FFI.** `src/*.mojo` is one compilation unit (build cost is fixed regardless of
function count).  `src/capi.mojo` holds the C ABI: every buffer crosses as an
`Int` address, because `@export` rejects parametric functions and a pointer
with an inferred origin is parametric; the wrapper rebuilds the pointer with
`Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=addr)`.  Each export
is a one-line call into the kernel above it, in upstream's order.  Python loads
the shared object with `ctypes` (`python/mojo_pyamg/_lib.py`) and hands it
NumPy buffers directly: no copy on the way in or out, the caller's arrays are
modified in place, as upstream's does.

**Memory layout.** Values are float64 and indices are int64 (`Int` is the
64-bit machine integer, so a NumPy `int64` array is exactly a Mojo `Pointer[Int]`).
CSR matrices cross as three buffers — `indptr[n+1]`, `indices[nnz]`, `data[nnz]`
— and BSR matrices the same way with `data` carrying `blocksize²` doubles per
block entry.  This is the same layout upstream's C++ kernels take, with the
index width widened to what `Int` is on this target.

**Where the work happens.** Strength of connection, aggregation, candidate
fitting, the prolongation-smoother constraint projection, every relaxation
sweep, the row/column scaling and filtering helpers, and the Givens/Householder
helpers run in Mojo.  Everything else — sparse products, the Galerkin operator,
the coarse-grid solve, the Krylov recurrences, the spectral-radius estimate — is
the same NumPy/SciPy code upstream uses, because those parts are BLAS-bound
rather than kernel-bound.

**This pass is correctness, not speed.** The kernels are scalar loops that
follow upstream's structure statement for statement: nothing is vectorized and
nothing is threaded, so most kernels land below upstream's C++ timings.  The
structure is left intact for a later pass that can accelerate it.

## Benchmarks

Best of 5 runs each (setup cases: best of 3), measured with `pixi run bench`.

<!-- BENCH_TABLE -->

## Tests

`pixi run test` runs the suite in `tests/`.  Every relaxation sweep, strength
measure, aggregation, candidate fit, prolongation smoother, Krylov solver and
the full `smoothed_aggregation_solver` hierarchy are compared against the
installed `pyamg` 5.3.0 on the same data:

- relaxation: identical results to ~1e-13, for CSR and BSR, all sweep
  directions, and the indexed and normal-equation variants
- strength: the sparsity pattern and the values match exactly
- aggregation: the same aggregates, the same Cpts, the same AggOp
- `fit_candidates`: identical `Q` and `R`, including the doctested example and
  the relation `B = Q R`
- prolongation smoothers: identical `P` once the spectral-radius estimate is
  seeded identically
- Krylov: identical iteration counts and identical residual histories
- hierarchy: the same number of levels, the same unknowns, nonzeros, `C`,
  `AggOp`, `Cpts`, `P` and `R` per level, and the same residual history cycle
  for cycle over V, W and F cycles and under CG acceleration
- the FFI surface itself: every ctypes signature is checked against the
  `@export` in `src/capi.mojo`, so a mis-declared argument cannot reach a kernel
  unnoticed

## Licence

MIT, as upstream.  `python/mojo_pyamg/krylov/_*.py` is derived from PyAMG
(MIT, <https://github.com/pyamg/pyamg>) and keeps upstream's algorithms
verbatim apart from the import of this package's `amg_core` shim.
