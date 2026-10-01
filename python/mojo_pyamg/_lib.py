"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_PYAMG_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pyamg.so"
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "mpa_classical_strength_of_connection_abs": ([I, F, I, I, I, I, I, I], None),
    "mpa_classical_strength_of_connection_min": ([I, F, I, I, I, I, I, I], None),
    "mpa_maximum_row_value": ([I, I, I, I, I], None),
    "mpa_symmetric_strength_of_connection": ([I, F, I, I, I, I, I, I, I], None),
    "mpa_standard_aggregation": ([I, I, I, I, I], I),
    "mpa_naive_aggregation": ([I, I, I, I, I], I),
    "mpa_pairwise_aggregation": ([I] * 7, I),
    "mpa_fit_candidates": ([I, I, I, I, I, I, I, I, I, F], None),
    "mpa_satisfy_constraints_helper": ([I] * 12, None),
    "mpa_truncate_rows_csr": ([I, I, I, I, I], None),
    "mpa_gauss_seidel": ([I] * 8, None),
    "mpa_sor_gauss_seidel": ([I] * 8 + [F], None),
    "mpa_bsr_gauss_seidel": ([I] * 11, None),
    "mpa_jacobi": ([I] * 9 + [F], None),
    "mpa_jacobi_indexed": ([I] * 8 + [F, I], None),
    "mpa_bsr_jacobi": ([I] * 10 + [F, I, I], None),
    "mpa_bsr_jacobi_indexed": ([I] * 9 + [F, I, I, I], None),
    "mpa_gauss_seidel_indexed": ([I] * 9, None),
    "mpa_jacobi_ne": ([I] * 10 + [F], None),
    "mpa_gauss_seidel_ne": ([I] * 9 + [F], None),
    "mpa_gauss_seidel_nr": ([I] * 9 + [F], None),
    "mpa_block_jacobi": ([I] * 10 + [F] + [I] * 3, None),
    "mpa_block_gauss_seidel": ([I] * 12, None),
    "mpa_csc_scale_columns": ([I, I, I, I], None),
    "mpa_csc_scale_rows": ([I, I, I, I, I], None),
    "mpa_filter_matrix_rows": ([I, F, I, I, I, I], None),
    "mpa_lower_tri_solve": ([I, I, I, I, I, I], None),
    "mpa_upper_tri_solve": ([I, I, I, I, I, I], None),
    "mpa_apply_householders": ([I, I, I, I, I, I], None),
    "mpa_householder_hornerscheme": ([I, I, I, I, I, I, I], None),
    "mpa_apply_givens": ([I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJO_PYAMG_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJO_PYAMG_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-pyamg.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        path = build()
        _lib = ctypes.CDLL(path)
        for name, (argtypes, restype) in _SIGNATURES.items():
            try:
                fn = getattr(_lib, name)
            except AttributeError as exc:
                raise BuildError(f"missing symbol {name} in {path}") from exc
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.asarray(a)
    if copy or not arr.flags["C_CONTIGUOUS"] or arr.dtype != np.float64:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def i64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous int64 view of `a`, copying only when it has to."""
    arr = np.asarray(a)
    if copy or not arr.flags["C_CONTIGUOUS"] or arr.dtype != np.int64:
        arr = np.ascontiguousarray(arr, dtype=np.int64)
    return arr


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojo_pyamg._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
