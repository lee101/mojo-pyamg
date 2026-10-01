"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  Callers own all
memory, including scratch -- nothing here allocates, so nothing here can leak.

`src/ported.mojo` holds the kernels in upstream source order; this file only
rebuilds the pointers, in the same order as the kernel they belong to.
"""

from ported import (
    apply_givens,
    apply_householders,
    bsr_gauss_seidel,
    bsr_jacobi,
    bsr_jacobi_indexed,
    block_gauss_seidel,
    block_jacobi,
    classical_strength_of_connection_abs,
    classical_strength_of_connection_min,
    csc_scale_columns,
    csc_scale_rows,
    filter_matrix_rows,
    fit_candidates,
    gauss_seidel,
    gauss_seidel_indexed,
    gauss_seidel_ne,
    gauss_seidel_nr,
    householder_hornerscheme,
    jacobi,
    jacobi_indexed,
    jacobi_ne,
    lower_tri_solve,
    maximum_row_value,
    naive_aggregation,
    pairwise_aggregation,
    sor_gauss_seidel,
    standard_aggregation,
    satisfy_constraints_helper,
    symmetric_strength_of_connection,
    truncate_rows_csr,
    upper_tri_solve,
)

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ------------------------------------------- amg_core/ruge_stuben.h
@export("mpa_classical_strength_of_connection_abs")
def mpa_classical_strength_of_connection_abs(
    n_row: Int, theta: Float64, Ap: Int, Aj: Int, Ax: Int, Sp: Int, Sj: Int, Sx: Int
) abi("C"):
    classical_strength_of_connection_abs(
        n_row, theta, ip(Ap), ip(Aj), p(Ax), ip(Sp), ip(Sj), p(Sx)
    )


@export("mpa_classical_strength_of_connection_min")
def mpa_classical_strength_of_connection_min(
    n_row: Int, theta: Float64, Ap: Int, Aj: Int, Ax: Int, Sp: Int, Sj: Int, Sx: Int
) abi("C"):
    classical_strength_of_connection_min(
        n_row, theta, ip(Ap), ip(Aj), p(Ax), ip(Sp), ip(Sj), p(Sx)
    )


@export("mpa_maximum_row_value")
def mpa_maximum_row_value(n_row: Int, x: Int, Ap: Int, Aj: Int, Ax: Int) abi("C"):
    maximum_row_value(n_row, p(x), ip(Ap), ip(Aj), p(Ax))


# -------------------------------------- amg_core/smoothed_aggregation.h
@export("mpa_symmetric_strength_of_connection")
def mpa_symmetric_strength_of_connection(
    n_row: Int,
    theta: Float64,
    Ap: Int,
    Aj: Int,
    Ax: Int,
    Sp: Int,
    Sj: Int,
    Sx: Int,
    diags: Int,
) abi("C"):
    symmetric_strength_of_connection(
        n_row, theta, ip(Ap), ip(Aj), p(Ax), ip(Sp), ip(Sj), p(Sx), p(diags)
    )


@export("mpa_standard_aggregation")
def mpa_standard_aggregation(n_row: Int, Ap: Int, Aj: Int, x: Int, y: Int) abi("C") -> Int:
    return standard_aggregation(n_row, ip(Ap), ip(Aj), ip(x), ip(y))


@export("mpa_naive_aggregation")
def mpa_naive_aggregation(n_row: Int, Ap: Int, Aj: Int, x: Int, y: Int) abi("C") -> Int:
    return naive_aggregation(n_row, ip(Ap), ip(Aj), ip(x), ip(y))


@export("mpa_pairwise_aggregation")
def mpa_pairwise_aggregation(
    n_row: Int,
    Sp: Int,
    Sj: Int,
    Sx: Int,
    x: Int,
    y: Int,
    scratch: Int,
) abi("C") -> Int:
    return pairwise_aggregation(
        n_row, ip(Sp), ip(Sj), p(Sx), ip(x), ip(y), ip(scratch),
    )


@export("mpa_fit_candidates")
def mpa_fit_candidates(
    n_row: Int,
    n_col: Int,
    K1: Int,
    K2: Int,
    Ap: Int,
    Ai: Int,
    Ax: Int,
    B: Int,
    R: Int,
    tol: Float64,
) abi("C"):
    fit_candidates(n_row, n_col, K1, K2, ip(Ap), ip(Ai), p(Ax), p(B), p(R), tol)


@export("mpa_satisfy_constraints_helper")
def mpa_satisfy_constraints_helper(
    rows_per_block: Int,
    cols_per_block: Int,
    num_block_rows: Int,
    NullDim: Int,
    x: Int,
    y: Int,
    z: Int,
    Sp: Int,
    Sj: Int,
    Sx: Int,
    Update: Int,
    C: Int,
) abi("C"):
    satisfy_constraints_helper(
        rows_per_block, cols_per_block, num_block_rows, NullDim,
        p(x), p(y), p(z), ip(Sp), ip(Sj), p(Sx), p(Update), p(C),
    )


@export("mpa_truncate_rows_csr")
def mpa_truncate_rows_csr(n_row: Int, k: Int, Sp: Int, Sj: Int, Sx: Int) abi("C"):
    truncate_rows_csr(n_row, k, ip(Sp), ip(Sj), p(Sx))


# ------------------------------------------- amg_core/relaxation.h
@export("mpa_gauss_seidel")
def mpa_gauss_seidel(
    Ap: Int, Aj: Int, Ax: Int, x: Int, b: Int, row_start: Int, row_stop: Int, row_step: Int
) abi("C"):
    gauss_seidel(ip(Ap), ip(Aj), p(Ax), p(x), p(b), row_start, row_stop, row_step)


@export("mpa_sor_gauss_seidel")
def mpa_sor_gauss_seidel(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
) abi("C"):
    sor_gauss_seidel(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), row_start, row_stop, row_step, omega
    )


@export("mpa_bsr_gauss_seidel")
def mpa_bsr_gauss_seidel(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    rsum: Int,
    Axloc: Int,
) abi("C"):
    bsr_gauss_seidel(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), row_start, row_stop, row_step, blocksize,
        p(rsum), p(Axloc),
    )


@export("mpa_jacobi")
def mpa_jacobi(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    temp: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
) abi("C"):
    jacobi(ip(Ap), ip(Aj), p(Ax), p(x), p(b), p(temp), row_start, row_stop, row_step, omega)


@export("mpa_bsr_jacobi")
def mpa_bsr_jacobi(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    temp: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    omega: Float64,
    rsum: Int,
    Axloc: Int,
) abi("C"):
    bsr_jacobi(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), p(temp), row_start, row_stop, row_step,
        blocksize, omega, p(rsum), p(Axloc),
    )


@export("mpa_bsr_jacobi_indexed")
def mpa_bsr_jacobi_indexed(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    x_size: Int,
    indices: Int,
    n_indices: Int,
    blocksize: Int,
    omega: Float64,
    rsum: Int,
    Axloc: Int,
    temp: Int,
) abi("C"):
    bsr_jacobi_indexed(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), x_size, ip(indices), n_indices,
        blocksize, omega, p(rsum), p(Axloc), p(temp),
    )


@export("mpa_jacobi_indexed")
def mpa_jacobi_indexed(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    x_size: Int,
    indices: Int,
    n_indices: Int,
    omega: Float64,
    temp: Int,
) abi("C"):
    jacobi_indexed(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), x_size, ip(indices), n_indices, omega, p(temp)
    )


@export("mpa_gauss_seidel_indexed")
def mpa_gauss_seidel_indexed(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    Id: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
) abi("C"):
    gauss_seidel_indexed(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), ip(Id), row_start, row_stop, row_step
    )


@export("mpa_jacobi_ne")
def mpa_jacobi_ne(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    Tx: Int,
    temp: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
) abi("C"):
    jacobi_ne(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), p(Tx), p(temp), row_start, row_stop, row_step, omega
    )


@export("mpa_gauss_seidel_ne")
def mpa_gauss_seidel_ne(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    Tx: Int,
    omega: Float64,
) abi("C"):
    gauss_seidel_ne(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), row_start, row_stop, row_step, p(Tx), omega
    )


@export("mpa_gauss_seidel_nr")
def mpa_gauss_seidel_nr(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    z: Int,
    col_start: Int,
    col_stop: Int,
    col_step: Int,
    Tx: Int,
    omega: Float64,
) abi("C"):
    gauss_seidel_nr(
        ip(Ap), ip(Aj), p(Ax), p(x), p(z), col_start, col_stop, col_step, p(Tx), omega
    )


@export("mpa_block_jacobi")
def mpa_block_jacobi(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    Tx: Int,
    temp: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
    blocksize: Int,
    rsum: Int,
    v: Int,
) abi("C"):
    block_jacobi(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), p(Tx), p(temp),
        row_start, row_stop, row_step, omega, blocksize, p(rsum), p(v),
    )


@export("mpa_block_gauss_seidel")
def mpa_block_gauss_seidel(
    Ap: Int,
    Aj: Int,
    Ax: Int,
    x: Int,
    b: Int,
    Tx: Int,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    rsum: Int,
    v: Int,
) abi("C"):
    block_gauss_seidel(
        ip(Ap), ip(Aj), p(Ax), p(x), p(b), p(Tx), row_start, row_stop, row_step,
        blocksize, p(rsum), p(v),
    )


# ----------------------------------------------- amg_core/linalg.h
@export("mpa_csc_scale_columns")
def mpa_csc_scale_columns(n_col: Int, Ap: Int, Ax: Int, Xx: Int) abi("C"):
    csc_scale_columns(n_col, ip(Ap), p(Ax), p(Xx))


@export("mpa_csc_scale_rows")
def mpa_csc_scale_rows(n_col: Int, Ap: Int, Aj: Int, Ax: Int, Xx: Int) abi("C"):
    csc_scale_rows(n_col, ip(Ap), ip(Aj), p(Ax), p(Xx))


@export("mpa_filter_matrix_rows")
def mpa_filter_matrix_rows(
    n_row: Int, theta: Float64, Ap: Int, Aj: Int, Ax: Int, lump: Int
) abi("C"):
    filter_matrix_rows(n_row, theta, ip(Ap), ip(Aj), p(Ax), lump)


@export("mpa_lower_tri_solve")
def mpa_lower_tri_solve(
    L: Int, rhs: Int, x: Int, m: Int, n: Int, is_col_major: Int
) abi("C"):
    lower_tri_solve(p(L), p(rhs), p(x), m, n, is_col_major)


@export("mpa_upper_tri_solve")
def mpa_upper_tri_solve(
    R: Int, rhs: Int, x: Int, m: Int, n: Int, is_col_major: Int
) abi("C"):
    upper_tri_solve(p(R), p(rhs), p(x), m, n, is_col_major)


# ----------------------------------------------- amg_core/krylov.h
@export("mpa_apply_householders")
def mpa_apply_householders(
    z: Int, B: Int, n: Int, start: Int, stop: Int, step: Int
) abi("C"):
    apply_householders(p(z), p(B), n, start, stop, step)


@export("mpa_householder_hornerscheme")
def mpa_householder_hornerscheme(
    z: Int, B: Int, y: Int, n: Int, start: Int, stop: Int, step: Int
) abi("C"):
    householder_hornerscheme(p(z), p(B), p(y), n, start, stop, step)


@export("mpa_apply_givens")
def mpa_apply_givens(B: Int, x: Int, n: Int, nrot: Int) abi("C"):
    apply_givens(p(B), p(x), n, nrot)
