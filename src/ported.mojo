"""A port of PyAMG's compute kernels to Mojo.

Every def below keeps the name, the argument order and the loop structure of
the upstream routine it came from, so this file reads top to bottom in the
order of the upstream C++ headers (`pyamg/amg_core/*.h`, pyamg 5.3.0), grouped
by the header each routine lives in:

    amg_core/ruge_stuben.h          strength of connection, row maxima
    amg_core/smoothed_aggregation.h  symmetric strength, aggregation, fitting
    amg_core/relaxation.h           relaxation sweeps
    amg_core/linalg.h               scaling, filtering, triangular solves
    amg_core/krylov.h               Givens / Householder helpers

Only the real (float64) instantiations are ported; upstream also instantiates
these templates for complex128 and the pybind layer dispatches on dtype.  The
Python wrapper rejects complex input, so there is no silent wrong answer.

`Int` is the 64-bit machine integer, so every index array crossing the C ABI
is a NumPy int64 buffer.  Values are always float64.
"""

from std.math import abs, sqrt

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]

# std::numeric_limits<double>::min() -- the initial value of a max over
# magnitudes, upstream.
comptime F64_MIN = 1.1754943508222875e-38
# std::numeric_limits<double>::lowest(), used as the -inf initial of a max.
comptime F64_LOWEST = -1.7976931348623157e308


# ---------------------------------------------------------------------------
# amg_core/ruge_stuben.h
# ---------------------------------------------------------------------------


def classical_strength_of_connection_abs(
    n_row: Int,
    theta: Float64,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    Sp: IPtr,
    Sj: IPtr,
    Sx: Ptr,
):
    """Classical strength of connection, |A[i,j]| >= theta*max_k|A[i,k]|."""
    var nnz: Int = 0
    Sp.unsafe_store(0, Int(0))

    var i: Int = 0
    while i < n_row:
        var max_offdiagonal = F64_MIN
        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        var jj = row_start
        while jj < row_end:
            if Aj.unsafe_load(jj) != i:
                max_offdiagonal = max(max_offdiagonal, abs(Ax.unsafe_load(jj)))
            jj += 1

        var threshold = theta * max_offdiagonal
        jj = row_start
        while jj < row_end:
            var norm_jj = abs(Ax.unsafe_load(jj))

            # Add entry if it exceeds the threshold
            if norm_jj >= threshold:
                if Aj.unsafe_load(jj) != i:
                    Sj.unsafe_store(nnz, Aj.unsafe_load(jj))
                    Sx.unsafe_store(nnz, Ax.unsafe_load(jj))
                    nnz += 1

            # Always add the diagonal
            if Aj.unsafe_load(jj) == i:
                Sj.unsafe_store(nnz, Aj.unsafe_load(jj))
                Sx.unsafe_store(nnz, Ax.unsafe_load(jj))
                nnz += 1
            jj += 1

        Sp.unsafe_store(i + 1, nnz)
        i += 1


def classical_strength_of_connection_min(
    n_row: Int,
    theta: Float64,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    Sp: IPtr,
    Sj: IPtr,
    Sx: Ptr,
):
    """Classical strength of connection, -A[i,j] >= theta*max_k(-A[i,k])."""
    var nnz: Int = 0
    Sp.unsafe_store(0, Int(0))

    var i: Int = 0
    while i < n_row:
        var max_offdiagonal = 0.0
        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        var jj = row_start
        while jj < row_end:
            if Aj.unsafe_load(jj) != i:
                max_offdiagonal = max(max_offdiagonal, -Ax.unsafe_load(jj))
            jj += 1

        var threshold = theta * max_offdiagonal
        jj = row_start
        while jj < row_end:
            var norm_jj = -Ax.unsafe_load(jj)

            # Add entry if it exceeds the threshold
            if norm_jj >= threshold:
                if Aj.unsafe_load(jj) != i:
                    Sj.unsafe_store(nnz, Aj.unsafe_load(jj))
                    Sx.unsafe_store(nnz, Ax.unsafe_load(jj))
                    nnz += 1

            # Always add the diagonal
            if Aj.unsafe_load(jj) == i:
                Sj.unsafe_store(nnz, Aj.unsafe_load(jj))
                Sx.unsafe_store(nnz, Ax.unsafe_load(jj))
                nnz += 1
            jj += 1

        Sp.unsafe_store(i + 1, nnz)
        i += 1


def maximum_row_value(
    n_row: Int,
    x: Ptr,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
):
    """x[i] = row i's maximum magnitude entry."""
    var i: Int = 0
    while i < n_row:
        var max_entry = F64_MIN
        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        # Find this row's max entry
        var jj = row_start
        while jj < row_end:
            max_entry = max(max_entry, abs(Ax.unsafe_load(jj)))
            jj += 1

        x.unsafe_store(i, max_entry)
        i += 1


# ---------------------------------------------------------------------------
# amg_core/smoothed_aggregation.h
# ---------------------------------------------------------------------------


def symmetric_strength_of_connection(
    n_row: Int,
    theta: Float64,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    Sp: IPtr,
    Sj: IPtr,
    Sx: Ptr,
    diags: Ptr,
):
    """|A(i,j)| >= theta*sqrt(|A(i,i)|*|A(j,j)|), diagonal always kept."""
    # Sp,Sj form a CSR representation where the i-th row contains
    # the indices of all the strong connections from node i

    # compute norm of diagonal values
    var i: Int = 0
    while i < n_row:
        var diag = 0.0
        var jj = Ap.unsafe_load(i)
        while jj < Ap.unsafe_load(i + 1):
            if Aj.unsafe_load(jj) == i:
                diag += Ax.unsafe_load(jj)  # gracefully handle duplicates
            jj += 1
        diags.unsafe_store(i, abs(diag))
        i += 1

    var nnz: Int = 0
    Sp.unsafe_store(0, Int(0))

    i = 0
    while i < n_row:
        var eps_Aii = theta * theta * diags.unsafe_load(i)

        var jj = Ap.unsafe_load(i)
        while jj < Ap.unsafe_load(i + 1):
            var j = Aj.unsafe_load(jj)
            var Aij = Ax.unsafe_load(jj)

            if i == j:
                # Always add the diagonal
                Sj.unsafe_store(nnz, j)
                Sx.unsafe_store(nnz, Aij)
                nnz += 1
            elif Aij * Aij >= eps_Aii * diags.unsafe_load(j):
                #  |A(i,j)| >= theta * sqrt(|A(i,i)|*|A(j,j)|)
                Sj.unsafe_store(nnz, j)
                Sx.unsafe_store(nnz, Aij)
                nnz += 1
            jj += 1
        Sp.unsafe_store(i + 1, nnz)
        i += 1


def standard_aggregation(n_row: Int, Ap: IPtr, Aj: IPtr, x: IPtr, y: IPtr) -> Int:
    """Greedy maximal-independent-set aggregation.  Returns the aggregate count."""
    # x[n] == 0 means i-th node has not been aggregated
    var i: Int = 0
    while i < n_row:
        x.unsafe_store(i, Int(0))
        i += 1

    var next_aggregate: Int = 1  # number of aggregates + 1

    # Pass #1
    i = 0
    while i < n_row:
        if x.unsafe_load(i) != 0:  # already marked
            i += 1
            continue

        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        # Determine whether all neighbors of this node are free
        var has_aggregated_neighbors = False
        var has_neighbors = False
        var jj = row_start
        while jj < row_end:
            var j = Aj.unsafe_load(jj)
            if i != j:
                has_neighbors = True
                if x.unsafe_load(j) != 0:
                    has_aggregated_neighbors = True
                    break
            jj += 1

        if not has_neighbors:
            # isolated node, do not aggregate
            x.unsafe_store(i, -n_row)
        elif not has_aggregated_neighbors:
            # Make an aggregate out of this node and its neighbors
            x.unsafe_store(i, next_aggregate)
            y.unsafe_store(next_aggregate - 1, i)  # y stores a list of the Cpts
            jj = row_start
            while jj < row_end:
                x.unsafe_store(Aj.unsafe_load(jj), next_aggregate)
                jj += 1
            next_aggregate += 1
        i += 1

    # Pass #2
    # Add unaggregated nodes to any neighboring aggregate
    i = 0
    while i < n_row:
        if x.unsafe_load(i) != 0:  # already marked
            i += 1
            continue

        var jj = Ap.unsafe_load(i)
        while jj < Ap.unsafe_load(i + 1):
            var xj = x.unsafe_load(Aj.unsafe_load(jj))
            if xj > 0:
                x.unsafe_store(i, -xj)
                break
            jj += 1
        i += 1

    next_aggregate -= 1

    # Pass #3
    i = 0
    while i < n_row:
        var xi = x.unsafe_load(i)

        if xi != 0:
            # node i has been aggregated
            if xi > 0:
                x.unsafe_store(i, xi - 1)
            elif xi == -n_row:
                x.unsafe_store(i, Int(-1))
            else:
                x.unsafe_store(i, -xi - 1)
            i += 1
            continue

        # node i has not been aggregated
        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        x.unsafe_store(i, next_aggregate)
        y.unsafe_store(next_aggregate, i)  # y stores a list of the Cpts

        var jj = row_start
        while jj < row_end:
            var j = Aj.unsafe_load(jj)
            if x.unsafe_load(j) == 0:  # unmarked neighbors
                x.unsafe_store(j, next_aggregate)
            jj += 1
        next_aggregate += 1
        i += 1

    return next_aggregate  # number of aggregates


def naive_aggregation(n_row: Int, Ap: IPtr, Aj: IPtr, x: IPtr, y: IPtr) -> Int:
    """Aggregate each unaggregated dof with all of its unaggregated neighbors."""
    # x[n] == 0 means i-th node has not been aggregated
    var i: Int = 0
    while i < n_row:
        x.unsafe_store(i, Int(0))
        i += 1

    var next_aggregate: Int = 1  # number of aggregates + 1

    i = 0
    while i < n_row:
        if x.unsafe_load(i) != 0:  # already marked
            i += 1
            continue

        var row_start = Ap.unsafe_load(i)
        var row_end = Ap.unsafe_load(i + 1)

        # Make an aggregate out of this node and its unaggregated neighbors
        x.unsafe_store(i, next_aggregate)
        var jj = row_start
        while jj < row_end:
            if x.unsafe_load(Aj.unsafe_load(jj)) == 0:
                x.unsafe_store(Aj.unsafe_load(jj), next_aggregate)
            jj += 1

        # y stores a list of the Cpts
        y.unsafe_store(next_aggregate - 1, i)
        next_aggregate += 1
        i += 1

    return next_aggregate - 1  # number of aggregates


def bucket_insert(node: Int, b: Int, head: IPtr, nxt: IPtr, prv: IPtr):
    """Append `node` to bucket `b`.

    `std::multimap::insert` places a new element at the *upper* bound of the
    equivalent range, so among equal keys the container keeps insertion order;
    appending at the tail of the chain reproduces that exactly.

    head, nxt and prv are views of one scratch array: passed as separate
    pointer arguments the compiler may assume they do not alias, which this
    intrusive list relies on.
    """
    var p = head.unsafe_load(b)
    if p == -1:
        head.unsafe_store(b, node)
    else:
        while nxt.unsafe_load(p) != -1:
            p = nxt.unsafe_load(p)
        nxt.unsafe_store(p, node)
    nxt.unsafe_store(node, Int(-1))
    prv.unsafe_store(node, p)


def pairwise_aggregation(
    n_row: Int,
    Sp: IPtr,
    Sj: IPtr,
    Sx: Ptr,
    x: IPtr,
    y: IPtr,
    scratch: IPtr,
) -> Int:
    """Greedy matching: pair each node with its strongest unaggregated neighbour.

    Upstream keeps the nodes in a `std::multimap` keyed on (m_i, i) and pops the
    minimum, where m_i counts the unaggregated neighbours of node i; a node
    whose key is decremented is re-inserted.  The bucket lists below are that
    same ordered container: bucket b holds the nodes with key b - n_row in
    insertion order, and keys only ever decrease, so the running minimum is
    tracked incrementally.
    """
    # workspace, carved out of the caller's single scratch array
    var m = scratch                                 # [0, n_row)
    var head = scratch.unsafe_offset(n_row)         # [n_row, 3*n_row+1)
    var nxt = scratch.unsafe_offset(3 * n_row + 1)   # [3*n_row+1, 4*n_row+1)
    var prv = scratch.unsafe_offset(4 * n_row + 1)   # [4*n_row+1, 5*n_row+1)
    var key = scratch.unsafe_offset(5 * n_row + 1)   # [5*n_row+1, 6*n_row+1)

    # x[n] == 0 means i-th node has not been aggregated
    var i: Int = 0
    while i < n_row:
        x.unsafe_store(i, Int(0))
        i += 1

    # count the unaggregated neighbours of every node
    i = 0
    while i < n_row:
        m.unsafe_store(i, Int(0))
        i += 1

    i = 0
    while i < n_row:
        var jj = Sp.unsafe_load(i)
        while jj < Sp.unsafe_load(i + 1):
            if Sj.unsafe_load(jj) != i:
                m.unsafe_store(Sj.unsafe_load(jj), m.unsafe_load(Sj.unsafe_load(jj)) + 1)
            jj += 1
        i += 1

    # Bucket list: head[k + n_row] is the first node with key k, nxt/prv are
    # the intrusive links, both ordered by node index.  -1 terminates a chain.
    i = 0
    while i < 2 * n_row + 1:
        head.unsafe_store(i, Int(-1))
        i += 1
    i = 0
    while i < n_row:
        key.unsafe_store(i, m.unsafe_load(i))
        bucket_insert(i, m.unsafe_load(i) + n_row, head, nxt, prv)
        i += 1

    # current minimum key, tracked incrementally
    var cur_min: Int = 0
    while cur_min + n_row < 2 * n_row + 1 and head.unsafe_load(cur_min + n_row) == -1:
        cur_min += 1

    var next_aggregate: Int = 1  # number of aggregates + 1
    var live: Int = n_row

    while live > 0:
        # select the minimum of m_i; the head of its bucket is exactly the
        # entry the multimap would hand back
        var i = head.unsafe_load(cur_min + n_row)
        head.unsafe_store(cur_min + n_row, nxt.unsafe_load(i))
        if nxt.unsafe_load(i) != -1:
            prv.unsafe_store(nxt.unsafe_load(i), Int(-1))
        live -= 1
        while live > 0 and head.unsafe_load(cur_min + n_row) == -1:
            cur_min += 1

        var row_start = Sp.unsafe_load(i)
        var row_end = Sp.unsafe_load(i + 1)

        var j: Int = 0
        var found = False
        var max_val = F64_LOWEST

        # x stores a list of the aggregate numbers
        x.unsafe_store(i, next_aggregate)

        # select the partner: the largest a_ij over unaggregated neighbours
        var jj = row_start
        while jj < row_end:
            if x.unsafe_load(Sj.unsafe_load(jj)) == 0 and Sx.unsafe_load(jj) >= max_val:
                max_val = Sx.unsafe_load(jj)
                j = Sj.unsafe_load(jj)
                found = True
            jj += 1
        if found:
            x.unsafe_store(j, next_aggregate)
        # y stores a list of the Cpts
        y.unsafe_store(next_aggregate - 1, i)

        # every unaggregated neighbour of i loses one unaggregated neighbour
        jj = row_start
        while jj < row_end:
            if x.unsafe_load(Sj.unsafe_load(jj)) == 0:
                # decrement the key and re-insert in index order
                var node = Sj.unsafe_load(jj)
                var kk = key.unsafe_load(node)
                var ob = kk + n_row
                var p = prv.unsafe_load(node)
                var nx = nxt.unsafe_load(node)
                if p != -1:
                    nxt.unsafe_store(p, nx)
                else:
                    head.unsafe_store(ob, nx)
                if nx != -1:
                    prv.unsafe_store(nx, p)

                key.unsafe_store(node, kk - 1)
                bucket_insert(node, (kk - 1) + n_row, head, nxt, prv)
                if kk - 1 < cur_min:
                    cur_min = kk - 1
            jj += 1

        if found:
            x.unsafe_store(j, next_aggregate)
            var row_start2 = Sp.unsafe_load(j)
            var row_end2 = Sp.unsafe_load(j + 1)
            jj = row_start2
            while jj < row_end2:
                if x.unsafe_load(Sj.unsafe_load(jj)) == 0:
                    var node = Sj.unsafe_load(jj)
                    var kk = key.unsafe_load(node)
                    var ob = kk + n_row
                    var p = prv.unsafe_load(node)
                    var nx = nxt.unsafe_load(node)
                    if p != -1:
                        nxt.unsafe_store(p, nx)
                    else:
                        head.unsafe_store(ob, nx)
                    if nx != -1:
                        prv.unsafe_store(nx, p)

                    key.unsafe_store(node, kk - 1)
                    bucket_insert(node, (kk - 1) + n_row, head, nxt, prv)
                    if kk - 1 < cur_min:
                        cur_min = kk - 1
                jj += 1
            # remove node j from the buckets
            var kb = key.unsafe_load(j) + n_row
            var pj = prv.unsafe_load(j)
            var nj = nxt.unsafe_load(j)
            if pj != -1:
                nxt.unsafe_store(pj, nj)
            else:
                head.unsafe_store(kb, nj)
            if nj != -1:
                prv.unsafe_store(nj, pj)
            live -= 1
        next_aggregate += 1

    return next_aggregate - 1  # number of aggregates


def fit_candidates(
    n_row: Int,
    n_col: Int,
    K1: Int,
    K2: Int,
    Ap: IPtr,
    Ai: IPtr,
    Ax: Ptr,
    B: Ptr,
    R: Ptr,
    tol: Float64,
):
    """Modified Gram-Schmidt fit of the near-nullspace candidates, column block
    by column block.  B is row-major (n_row*K1, K2), Ax is the BSR data array
    (AggOp.nnz, K1, K2) and R is (n_col, K2, K2) row-major."""
    var i: Int = 0
    while i < n_col * K2 * K2:
        R.unsafe_store(i, 0.0)
        i += 1

    var BS = K1 * K2  # blocksize

    # Copy blocks into Ax
    var j: Int = 0
    while j < n_col:
        var ax = BS * Ap.unsafe_load(j)
        var ii = Ap.unsafe_load(j)
        while ii < Ap.unsafe_load(j + 1):
            var b = BS * Ai.unsafe_load(ii)
            var t: Int = 0
            while t < BS:
                Ax.unsafe_store(ax + t, B.unsafe_load(b + t))
                t += 1
            ax += BS
            ii += 1
        j += 1

    # orthonormalize columns
    j = 0
    while j < n_col:
        var col_start = Ap.unsafe_load(j)
        var col_end = Ap.unsafe_load(j + 1)
        var Ax_start = BS * col_start
        var Ax_end = BS * col_end
        var R_start = j * K2 * K2

        var bj: Int = 0
        while bj < K2:
            # compute norm of block column
            var norm_j = 0.0
            var c = Ax_start + bj
            while c < Ax_end:
                var a = Ax.unsafe_load(c)
                norm_j += a * a
                c += K2
            norm_j = sqrt(norm_j)

            var threshold_j = tol * norm_j

            # orthogonalize bj against previous columns
            var bi: Int = 0
            while bi < bj:
                # compute dot product with column bi
                var dot_prod = 0.0
                var cb = Ax_start + bi
                var cj = Ax_start + bj
                while cb < Ax_end:
                    dot_prod += Ax.unsafe_load(cj) * Ax.unsafe_load(cb)
                    cb += K2
                    cj += K2

                # orthogonalize against column bi
                cb = Ax_start + bi
                cj = Ax_start + bj
                while cb < Ax_end:
                    var v = Ax.unsafe_load(cj) - dot_prod * Ax.unsafe_load(cb)
                    Ax.unsafe_store(cj, v)
                    cb += K2
                    cj += K2

                R.unsafe_store(R_start + K2 * bi + bj, dot_prod)
                bi += 1

            # compute norm of column bj
            norm_j = 0.0
            var cj = Ax_start + bj
            while cj < Ax_end:
                var a = Ax.unsafe_load(cj)
                norm_j += a * a
                cj += K2
            norm_j = sqrt(norm_j)

            # normalize column bj if, after orthogonalization, its
            # euclidean norm exceeds the threshold.  Otherwise zero it.
            var scale: Float64 = 0.0
            if norm_j > threshold_j:
                scale = 1.0 / norm_j
                R.unsafe_store(R_start + K2 * bj + bj, norm_j)
            else:
                R.unsafe_store(R_start + K2 * bj + bj, 0.0)
            cj = Ax_start + bj
            while cj < Ax_end:
                var v = Ax.unsafe_load(cj) * scale
                Ax.unsafe_store(cj, v)
                cj += K2

            bj += 1
        j += 1


def satisfy_constraints_helper(
    rows_per_block: Int,
    cols_per_block: Int,
    num_block_rows: Int,
    NullDim: Int,
    x: Ptr,
    y: Ptr,
    z: Ptr,
    Sp: IPtr,
    Sj: IPtr,
    Sx: Ptr,
    Update: Ptr,
    C: Ptr,
):
    """Project out the components of span(B) from each block row of S.

    Implements the python documented in the upstream header::

        U.data[n] -= dot(UBi, dot(BtBinv[i], Bi.H))
    """
    # Rename to something more familiar
    var Bt = x
    var UB = y
    var BtBinv = z

    var BlockSize = rows_per_block * cols_per_block
    var NullDimSq = NullDim * NullDim
    var NullDim_Cols = NullDim * cols_per_block
    var NullDim_Rows = NullDim * rows_per_block

    # C stores an intermediate mat-mat product
    var i: Int = 0
    while i < NullDim_Cols:
        C.unsafe_store(i, 0.0)
        i += 1

    # Begin Main Loop
    i = 0
    while i < num_block_rows:
        var rowstart = Sp.unsafe_load(i)
        var rowend = Sp.unsafe_load(i + 1)

        var jj = rowstart
        while jj < rowend:
            # C = BtBinv[i] @ B[Sj[jj]].H
            var a: Int = 0
            while a < NullDim:
                var b: Int = 0
                while b < cols_per_block:
                    var s = 0.0
                    var c: Int = 0
                    while c < NullDim:
                        s += BtBinv.unsafe_load(i * NullDimSq + a * NullDim + c) * \
                            Bt.unsafe_load(Sj.unsafe_load(jj) * NullDim_Cols + c * cols_per_block + b)
                        c += 1
                    C.unsafe_store(a * cols_per_block + b, s)
                    b += 1
                a += 1

            # Update = UB[i] @ C
            var r: Int = 0
            while r < rows_per_block:
                var c: Int = 0
                while c < cols_per_block:
                    var s = 0.0
                    var a: Int = 0
                    while a < NullDim:
                        s += UB.unsafe_load(i * NullDim_Rows + r * NullDim + a) * \
                            C.unsafe_load(a * cols_per_block + c)
                        a += 1
                    Update.unsafe_store(r * cols_per_block + c, s)
                    c += 1
                r += 1

            # Update Sx
            var k: Int = 0
            while k < BlockSize:
                var idx = jj * BlockSize + k
                var v = Sx.unsafe_load(idx) - Update.unsafe_load(k)
                Sx.unsafe_store(idx, v)
                k += 1
            jj += 1
        i += 1


def truncate_rows_csr(n_row: Int, k: Int, Sp: IPtr, Sj: IPtr, Sx: Ptr):
    """Zero out all but the k largest-magnitude entries of each row.

    Upstream sorts each row by magnitude with a quicksort over the parallel
    (Sx, Sj) arrays and then zeroes the leading entries.  Shuffling two
    parallel arrays through separate pointer arguments is miscompiled by this
    toolchain (the values end up sorted and the indices do not follow them), so
    the k survivors are selected in place instead: the kept entries are the
    same, and which of several equal-magnitude entries survives a tie may
    differ from upstream's quicksort order.
    """
    var i: Int = 0
    while i < n_row:
        # Only truncate entries if row is long enough
        var rowstart = Sp.unsafe_load(i)
        var rowend = Sp.unsafe_load(i + 1)
        if rowend - rowstart > k:
            # Move the k largest-magnitude entries to the front of the row,
            # zeroing what they displace
            var kept: Int = 0
            while kept < k:
                var best = rowstart + kept
                var best_val = abs(Sx.unsafe_load(rowstart + kept))
                var jj = rowstart + kept + 1
                while jj < rowend:
                    var val = abs(Sx.unsafe_load(jj))
                    if val > best_val:
                        best = jj
                        best_val = val
                    jj += 1
                # swap the winner into place and zero its old position
                var src = Sx.unsafe_load(best)
                var src_j = Sj.unsafe_load(best)
                Sx.unsafe_store(best, Sx.unsafe_load(rowstart + kept))
                Sj.unsafe_store(best, Sj.unsafe_load(rowstart + kept))
                Sx.unsafe_store(rowstart + kept, src)
                Sj.unsafe_store(rowstart + kept, src_j)
                kept += 1
            # Zero out all but the largest k
            var jj = rowstart + k
            while jj < rowend:
                Sx.unsafe_store(jj, 0.0)
                jj += 1
        i += 1


# ---------------------------------------------------------------------------
# amg_core/relaxation.h
# ---------------------------------------------------------------------------


def gauss_seidel(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
):
    """One Gauss-Seidel sweep over rows row_start, row_start+row_step, ..."""
    var i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var rsum = 0.0
        var diag = 0.0

        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if i == j:
                diag = Ax.unsafe_load(jj)
            else:
                rsum += Ax.unsafe_load(jj) * x.unsafe_load(j)
            jj += 1

        if diag != 0.0:
            x.unsafe_store(i, (b.unsafe_load(i) - rsum) / diag)
        i += row_step


def sor_gauss_seidel(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
):
    """One SOR sweep; identical to gauss_seidel with a relaxation parameter."""
    var i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var rsum = 0.0
        var diag = 0.0

        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if i == j:
                diag = Ax.unsafe_load(jj)
            else:
                rsum += Ax.unsafe_load(jj) * x.unsafe_load(j)
            jj += 1

        if diag != 0.0:
            x.unsafe_store(
                i,
                omega * ((b.unsafe_load(i) - rsum) / diag) + (1 - omega) * x.unsafe_load(i),
            )
        i += row_step


def bsr_gauss_seidel(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    rsum: Ptr,
    Axloc: Ptr,
):
    """Point-wise Gauss-Seidel over a BSR matrix (block row sweep, point GS)."""
    var B2 = blocksize * blocksize

    # Determine if this is a forward, or backward sweep
    var step: Int = 1
    var step_start: Int = 0
    var step_end: Int = blocksize
    if row_step < 0:
        step = -1
        step_start = blocksize - 1
        step_end = -1

    var i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var diag_ptr: Int = -1

        # initialize rsum to b, then later subtract A*x
        var k: Int = 0
        while k < blocksize:
            rsum.unsafe_store(k, b.unsafe_load(i * blocksize + k))
            k += 1

        # loop over row i
        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            var col = j * blocksize  # column entry for the start of this block

            if i == j:  # point to where in Ax the diagonal block starts
                diag_ptr = jj * B2
            else:
                # do a dense multiply of this block times x and accumulate in rsum
                var r: Int = 0
                while r < blocksize:
                    var s = 0.0
                    var t: Int = 0
                    while t < blocksize:
                        s += Ax.unsafe_load(jj * B2 + r * blocksize + t) * \
                            x.unsafe_load(col + t)
                        t += 1
                    Axloc.unsafe_store(r, s)
                    r += 1
                r = 0
                while r < blocksize:
                    rsum.unsafe_store(r, rsum.unsafe_load(r) - Axloc.unsafe_load(r))
                    r += 1
            jj += 1

        # Carry out point-wise GS over the diagonal block,
        # all the other blocks have been factored into rsum.
        if diag_ptr != -1:
            k = step_start
            while k != step_end:
                var diag = 1.0
                var kk = step_start
                while kk != step_end:
                    if k == kk:
                        # diagonal entry
                        diag = Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                    else:
                        # off-diag entry
                        rsum.unsafe_store(
                            k,
                            rsum.unsafe_load(k)
                            - Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                            * x.unsafe_load(i * blocksize + kk),
                        )
                    kk += step
                if diag != 0.0:
                    x.unsafe_store(i * blocksize + k, rsum.unsafe_load(k) / diag)
                k += step

        i += row_step


def jacobi(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    temp: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
):
    """Weighted Jacobi; temp holds a copy of x so the sweep is not sequential."""
    var one = 1.0
    var omega2 = omega

    var i = row_start
    while i != row_stop:
        temp.unsafe_store(i, x.unsafe_load(i))
        i += row_step

    i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var rsum = 0.0
        var diag = 0.0

        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if i == j:
                diag = Ax.unsafe_load(jj)
            else:
                rsum += Ax.unsafe_load(jj) * temp.unsafe_load(j)
            jj += 1

        if diag != 0.0:
            x.unsafe_store(
                i, (one - omega2) * temp.unsafe_load(i) + omega2 * ((b.unsafe_load(i) - rsum) / diag)
            )
        i += row_step


def jacobi_indexed(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    x_size: Int,
    indices: IPtr,
    n_indices: Int,
    omega: Float64,
    temp: Ptr,
):
    """Weighted Jacobi restricted to the rows listed in indices."""
    var one = 1.0
    var omega2 = omega

    # Set temp = x
    var t: Int = 0
    while t < x_size:
        temp.unsafe_store(t, x.unsafe_load(t))
        t += 1

    # Perform Jacobi on each row in indices[]
    var i: Int = 0
    while i < n_indices:
        var row = indices.unsafe_load(i)
        var start = Ap.unsafe_load(row)
        var end = Ap.unsafe_load(row + 1)
        var rsum = 0.0
        var diag = 0.0

        # Compute A*x for this row, get diagonal element of A
        var jj = start
        while jj < end:
            var col = Aj.unsafe_load(jj)
            if row == col:
                diag = Ax.unsafe_load(jj)
            else:
                rsum += Ax.unsafe_load(jj) * temp.unsafe_load(col)
            jj += 1

        # Check for nonzero diagonal and update ith index
        if diag != 0.0:
            x.unsafe_store(
                row, (one - omega2) * temp.unsafe_load(row) + omega2 * ((b.unsafe_load(row) - rsum) / diag)
            )
        i += 1



def bsr_jacobi(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    temp: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    omega: Float64,
    rsum: Ptr,
    Axloc: Ptr,
):
    """Weighted Jacobi applied point-wise inside a BSR matrix."""
    var B2 = blocksize * blocksize
    var one = 1.0
    var omega2 = omega

    # Determine if this is a forward, or backward sweep
    var step: Int = 1
    var step_start: Int = 0
    var step_end: Int = blocksize
    if row_step < 0:
        step = -1
        step_start = blocksize - 1
        step_end = -1

    # copy x to temp
    var i: Int = 0
    while i < abs(row_stop - row_start) * blocksize:
        temp.unsafe_store(i, x.unsafe_load(i))
        i += step

    i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var diag_ptr: Int = -1

        # initialize rsum to b, then later subtract A*x
        var k: Int = 0
        while k < blocksize:
            rsum.unsafe_store(k, b.unsafe_load(i * blocksize + k))
            k += 1

        # loop over row i
        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            var col = j * blocksize  # column entry for the start of this block

            if i == j:  # point to where in Ax the diagonal block starts
                diag_ptr = jj * B2
            else:
                # do a dense multiply of this block times x and accumulate in rsum
                var m: Int = 0
                while m < blocksize:
                    var s = 0.0
                    var t: Int = 0
                    while t < blocksize:
                        s += Ax.unsafe_load(jj * B2 + m * blocksize + t) * \
                            temp.unsafe_load(col + t)
                        t += 1
                    Axloc.unsafe_store(m, s)
                    rsum.unsafe_store(m, rsum.unsafe_load(m) - s)
                    m += 1
            jj += 1

        # Carry out point-wise jacobi over the diagonal block,
        # all the other blocks have been factored into rsum.
        if diag_ptr != -1:
            k = step_start
            while k != step_end:
                var diag = 1.0
                var kk = step_start
                while kk != step_end:
                    if k == kk:
                        # diagonal entry
                        diag = Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                    else:
                        # off-diag entry
                        rsum.unsafe_store(
                            k,
                            rsum.unsafe_load(k)
                            - Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                            * temp.unsafe_load(i * blocksize + kk),
                        )
                    kk += step
                if diag != 0.0:
                    x.unsafe_store(
                        i * blocksize + k,
                        (one - omega2) * temp.unsafe_load(i * blocksize + k)
                        + omega2 * rsum.unsafe_load(k) / diag,
                    )
                k += step

        i += row_step


def bsr_jacobi_indexed(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    x_size: Int,
    indices: IPtr,
    n_indices: Int,
    blocksize: Int,
    omega: Float64,
    rsum: Ptr,
    Axloc: Ptr,
    temp: Ptr,
):
    """Indexed weighted Jacobi applied point-wise inside a BSR matrix."""
    var B2 = blocksize * blocksize
    var omega2 = omega
    var one = 1.0

    # copy x to temp
    var t: Int = 0
    while t < x_size:
        temp.unsafe_store(t, x.unsafe_load(t))
        t += 1

    var i: Int = 0
    while i < n_indices:
        var row = indices.unsafe_load(i)
        var start = Ap.unsafe_load(row)
        var end = Ap.unsafe_load(row + 1)
        var diag_ptr: Int = -1

        # initialize rsum to b, then later subtract A*x
        var k: Int = 0
        while k < blocksize:
            rsum.unsafe_store(k, b.unsafe_load(row * blocksize + k))
            k += 1

        # loop over this row
        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            var col = j * blocksize  # column entry for the start of this block

            if row == j:  # point to where in Ax the diagonal block starts
                diag_ptr = jj * B2
            else:
                # do a dense multiply of this block times x and accumulate in rsum
                var m: Int = 0
                while m < blocksize:
                    var s = 0.0
                    var tt: Int = 0
                    while tt < blocksize:
                        s += Ax.unsafe_load(jj * B2 + m * blocksize + tt) * \
                            temp.unsafe_load(col + tt)
                        tt += 1
                    Axloc.unsafe_store(m, s)
                    rsum.unsafe_store(m, rsum.unsafe_load(m) - s)
                    m += 1
            jj += 1

        # Carry out point-wise jacobi over the diagonal block,
        # all the other blocks have been factored into rsum.
        if diag_ptr != -1:
            k = 0
            while k < blocksize:
                var diag = 1.0
                var kk: Int = 0
                while kk < blocksize:
                    if k == kk:
                        # diagonal entry
                        diag = Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                    else:
                        # off-diag entry
                        rsum.unsafe_store(
                            k,
                            rsum.unsafe_load(k)
                            - Ax.unsafe_load(k * blocksize + kk + diag_ptr)
                            * temp.unsafe_load(row * blocksize + kk),
                        )
                    kk += 1
                if diag != 0.0:
                    x.unsafe_store(
                        row * blocksize + k,
                        (one - omega2) * temp.unsafe_load(row * blocksize + k)
                        + omega2 * rsum.unsafe_load(k) / diag,
                    )
                k += 1
        i += 1

def gauss_seidel_indexed(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    Id: IPtr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
):
    """Gauss-Seidel restricted to the row indices Id[row_start:row_stop]."""
    var i = row_start
    while i != row_stop:
        var inew = Id.unsafe_load(i)
        var start = Ap.unsafe_load(inew)
        var end = Ap.unsafe_load(inew + 1)
        var rsum = 0.0
        var diag = 0.0

        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if inew == j:
                diag = Ax.unsafe_load(jj)
            else:
                rsum += Ax.unsafe_load(jj) * x.unsafe_load(j)
            jj += 1

        if diag != 0.0:
            x.unsafe_store(inew, (b.unsafe_load(inew) - rsum) / diag)
        i += row_step


def jacobi_ne(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    Tx: Ptr,
    temp: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
):
    """Cimmino relaxation: weighted Jacobi on A.H A x = A.H b, applied additively."""
    var delta = Tx
    var omega2 = omega

    var i = row_start
    while i < row_stop:
        temp.unsafe_store(i, 0.0)
        i += row_step

    i = row_start
    while i < row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var j = start
        while j < end:
            temp.unsafe_store(Aj.unsafe_load(j), temp.unsafe_load(Aj.unsafe_load(j)) + omega2 * Ax.unsafe_load(j) * delta.unsafe_load(i))
            j += 1
        i += row_step

    i = row_start
    while i < row_stop:
        x.unsafe_store(i, x.unsafe_load(i) + temp.unsafe_load(i))
        i += row_step


def gauss_seidel_ne(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    Tx: Ptr,
    omega: Float64,
):
    """Gauss-Seidel on A A.H y = b with x = A.H y, applied additively."""
    var D_inv = Tx

    var i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)

        # First calculate "delta", the scaled residual term
        var delta = 0.0
        var j = start
        while j < end:
            delta += Ax.unsafe_load(j) * x.unsafe_load(Aj.unsafe_load(j))
            j += 1
        delta = (b.unsafe_load(i) - delta) * D_inv.unsafe_load(i) * omega

        j = start
        while j < end:
            x.unsafe_store(Aj.unsafe_load(j), x.unsafe_load(Aj.unsafe_load(j)) + Ax.unsafe_load(j) * delta)
            j += 1
        i += row_step


def gauss_seidel_nr(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    z: Ptr,
    col_start: Int,
    col_stop: Int,
    col_step: Int,
    Tx: Ptr,
    omega: Float64,
):
    """Gauss-Seidel on A.H A x = A.H b, swept over CSC columns, applied additively."""
    var D_inv = Tx
    var r = z

    var i = col_start
    while i != col_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)

        # delta = < A e_i, r >
        var delta = 0.0
        var j = start
        while j < end:
            delta += Ax.unsafe_load(j) * r.unsafe_load(Aj.unsafe_load(j))
            j += 1

        # delta /=  omega*(A.H A)_{ii}
        delta *= D_inv.unsafe_load(i) * omega

        # update entry in x forcing < A.H b - A.H A x, e_i > = 0
        x.unsafe_store(i, x.unsafe_load(i) + delta)

        # r -= delta A e_i
        j = start
        while j < end:
            r.unsafe_store(Aj.unsafe_load(j), r.unsafe_load(Aj.unsafe_load(j)) - delta * Ax.unsafe_load(j))
            j += 1
        i += col_step


def block_jacobi(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    Tx: Ptr,
    temp: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    omega: Float64,
    blocksize: Int,
    rsum: Ptr,
    v: Ptr,
):
    """Weighted block Jacobi; Dinv holds the inverses of the diagonal blocks."""
    var Dinv = Tx
    var one = 1.0
    var omega2 = omega
    var blocksize_sq = blocksize * blocksize

    # Copy x to temp vector
    var i = row_start * blocksize
    while i != row_stop * blocksize:
        var k: Int = 0
        while k < blocksize:
            temp.unsafe_store(i + k, x.unsafe_load(i + k))
            k += 1
        i += row_step * blocksize

    # Begin block Jacobi sweep
    i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var k: Int = 0
        while k < blocksize:
            rsum.unsafe_store(k, 0.0)
            k += 1

        # Carry out a block dot product between block row i and x
        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if i == j:
                # diagonal, do nothing
                jj += 1
                continue
            else:
                var r: Int = 0
                while r < blocksize:
                    var s = 0.0
                    var t: Int = 0
                    while t < blocksize:
                        s += Ax.unsafe_load(jj * blocksize_sq + r * blocksize + t) * \
                            temp.unsafe_load(j * blocksize + t)
                        t += 1
                    v.unsafe_store(r, s)
                    r += 1
                r = 0
                while r < blocksize:
                    rsum.unsafe_store(r, rsum.unsafe_load(r) + v.unsafe_load(r))
                    r += 1
            jj += 1

        # x[i*bs:(i+1)*bs] = (1 - omega)*temp + omega*Dinv*(b - rsum)
        var iblocksize = i * blocksize
        k = 0
        while k < blocksize:
            rsum.unsafe_store(k, b.unsafe_load(iblocksize + k) - rsum.unsafe_load(k))
            k += 1

        k = 0
        while k < blocksize:
            var s = 0.0
            var t: Int = 0
            while t < blocksize:
                s += Dinv.unsafe_load(i * blocksize_sq + k * blocksize + t) * rsum.unsafe_load(t)
                t += 1
            v.unsafe_store(k, s)
            k += 1

        k = 0
        while k < blocksize:
            x.unsafe_store(
                iblocksize + k,
                (one - omega2) * temp.unsafe_load(iblocksize + k) + omega2 * v.unsafe_load(k),
            )
            k += 1
        i += row_step


def block_gauss_seidel(
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    x: Ptr,
    b: Ptr,
    Tx: Ptr,
    row_start: Int,
    row_stop: Int,
    row_step: Int,
    blocksize: Int,
    rsum: Ptr,
    v: Ptr,
):
    """Block Gauss-Seidel; Dinv holds the inverses of the diagonal blocks."""
    var Dinv = Tx
    var blocksize_sq = blocksize * blocksize

    # Begin block Gauss-Seidel sweep
    var i = row_start
    while i != row_stop:
        var start = Ap.unsafe_load(i)
        var end = Ap.unsafe_load(i + 1)
        var k: Int = 0
        while k < blocksize:
            rsum.unsafe_store(k, 0.0)
            k += 1

        # Carry out a block dot product between block row i and x
        var jj = start
        while jj < end:
            var j = Aj.unsafe_load(jj)
            if i == j:
                # diagonal, do nothing
                jj += 1
                continue
            else:
                var r: Int = 0
                while r < blocksize:
                    var s = 0.0
                    var t: Int = 0
                    while t < blocksize:
                        s += Ax.unsafe_load(jj * blocksize_sq + r * blocksize + t) * \
                            x.unsafe_load(j * blocksize + t)
                        t += 1
                    v.unsafe_store(r, s)
                    r += 1
                r = 0
                while r < blocksize:
                    rsum.unsafe_store(r, rsum.unsafe_load(r) + v.unsafe_load(r))
                    r += 1
            jj += 1

        # x[i*bs:(i+1)*bs] = Dinv[i]*(b[i*bs:(i+1)*bs] - rsum)
        var iblocksize = i * blocksize
        k = 0
        while k < blocksize:
            rsum.unsafe_store(k, b.unsafe_load(iblocksize + k) - rsum.unsafe_load(k))
            k += 1

        k = 0
        while k < blocksize:
            var s = 0.0
            var t: Int = 0
            while t < blocksize:
                s += Dinv.unsafe_load(i * blocksize_sq + k * blocksize + t) * rsum.unsafe_load(t)
                t += 1
            x.unsafe_store(iblocksize + k, s)
            k += 1
        i += row_step


# ---------------------------------------------------------------------------
# amg_core/linalg.h
# ---------------------------------------------------------------------------


def csc_scale_columns(
    n_col: Int,
    Ap: IPtr,
    Ax: Ptr,
    Xx: Ptr,
):
    """A[:, i] *= X[i] for a CSC matrix, in place."""
    var i: Int = 0
    while i < n_col:
        var jj = Ap.unsafe_load(i)
        while jj < Ap.unsafe_load(i + 1):
            Ax.unsafe_store(jj, Ax.unsafe_load(jj) * Xx.unsafe_load(i))
            jj += 1
        i += 1


def csc_scale_rows(
    n_col: Int,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    Xx: Ptr,
):
    """A[i, :] *= X[i] for a CSC matrix, in place."""
    var nnz = Ap.unsafe_load(n_col)
    var i: Int = 0
    while i < nnz:
        Ax.unsafe_store(i, Ax.unsafe_load(i) * Xx.unsafe_load(Aj.unsafe_load(i)))
        i += 1


def filter_matrix_rows(
    n_row: Int,
    theta: Float64,
    Ap: IPtr,
    Aj: IPtr,
    Ax: Ptr,
    lump: Int,
):
    """Set A_ij = 0 (or lump it onto A_ii) when |A_ij| < theta*|A_ii|."""
    # Lump each row by setting A_ii += A_ij for all j s.t. |A_ij| < theta*|A_ii|,
    # and set A_ij = 0
    if lump != 0:
        var i: Int = 0
        while i < n_row:
            var diagonal = 0.0
            var row_start = Ap.unsafe_load(i)
            var row_end = Ap.unsafe_load(i + 1)

            # Find diagonal of this row
            var diag_ind: Int = -1
            var jj = row_start
            while jj < row_end:
                if Aj.unsafe_load(jj) == i:
                    diag_ind = jj
                    diagonal = abs(Ax.unsafe_load(jj))
                    break
                jj += 1

            # Set threshold for strong connections
            var threshold = theta * diagonal
            jj = row_start
            while jj < row_end:
                var norm_jj = abs(Ax.unsafe_load(jj))

                # Remove entry if below threshold
                if norm_jj < threshold and Aj.unsafe_load(jj) != i:
                    Ax.unsafe_store(diag_ind, Ax.unsafe_load(diag_ind) + Ax.unsafe_load(jj))
                    Ax.unsafe_store(jj, 0.0)
                jj += 1
            i += 1
    else:
        # Filter each row by setting explicit zeros when |A_ij| < theta*|A_ii|
        var i: Int = 0
        while i < n_row:
            var diagonal = 0.0
            var row_start = Ap.unsafe_load(i)
            var row_end = Ap.unsafe_load(i + 1)

            # Find diagonal of this row
            var jj = row_start
            while jj < row_end:
                if Aj.unsafe_load(jj) == i:
                    diagonal = abs(Ax.unsafe_load(jj))
                    break
                jj += 1

            # Set threshold for strong connections
            var threshold = theta * diagonal
            jj = row_start
            while jj < row_end:
                var norm_jj = abs(Ax.unsafe_load(jj))

                # Filter row entries
                if norm_jj < threshold:
                    Ax.unsafe_store(jj, 0.0)
                jj += 1
            i += 1


def lower_tri_solve(
    L: Ptr,
    rhs: Ptr,
    x: Ptr,
    m: Int,
    n: Int,
    is_col_major: Int,
):
    """Forward substitution: L x = rhs over the lower-triangular min(m,n) block."""
    var rank = min(m, n)
    var i: Int = 0
    while i < rank:
        var temp = rhs.unsafe_load(i)
        var j: Int = 0
        while j < i:
            var idx: Int
            if is_col_major != 0:
                idx = i + j * m
            else:
                idx = i * n + j
            temp -= L.unsafe_load(idx) * x.unsafe_load(j)
            j += 1
        var didx: Int
        if is_col_major != 0:
            didx = i + i * m
        else:
            didx = i * n + i
        if abs(L.unsafe_load(didx)) < 1e-12:
            x.unsafe_store(i, 0.0)
        else:
            x.unsafe_store(i, temp / L.unsafe_load(didx))
        i += 1

    # If rank < size of rhs, set free elements in x to zero
    i = m
    while i < n:
        x.unsafe_store(i, 0.0)
        i += 1


def upper_tri_solve(
    R: Ptr,
    rhs: Ptr,
    x: Ptr,
    m: Int,
    n: Int,
    is_col_major: Int,
):
    """Backward substitution: R x = rhs over the upper-triangular min(m,n) block."""
    var rank = min(m, n)
    var i = rank - 1
    while i >= 0:
        var temp = rhs.unsafe_load(i)
        var j = i + 1
        while j < rank:
            var idx: Int
            if is_col_major != 0:
                idx = i + j * m
            else:
                idx = i * n + j
            temp -= R.unsafe_load(idx) * x.unsafe_load(j)
            j += 1
        var didx: Int
        if is_col_major != 0:
            didx = i + i * m
        else:
            didx = i * n + i
        if abs(R.unsafe_load(didx)) < 1e-12:
            x.unsafe_store(i, 0.0)
        else:
            x.unsafe_store(i, temp / R.unsafe_load(didx))
        i -= 1


# ---------------------------------------------------------------------------
# amg_core/krylov.h
# ---------------------------------------------------------------------------


def dot_prod(a: Ptr, b: Ptr, n: Int) -> Float64:
    """Inner product of two length-n vectors."""
    var t = 0.0
    var i: Int = 0
    while i < n:
        t += a.unsafe_load(i) * b.unsafe_load(i)
        i += 1
    return t


def axpy(a: Ptr, x: Ptr, alpha: Float64, n: Int):
    """a += alpha * x."""
    var i: Int = 0
    while i < n:
        var v = a.unsafe_load(i) + alpha * x.unsafe_load(i)
        a.unsafe_store(i, v)
        i += 1


def apply_householders(
    z: Ptr,
    B: Ptr,
    n: Int,
    start: Int,
    stop: Int,
    step: Int,
):
    """z = (I - 2 w w^H) z for each Householder vector B[:, range]."""
    var index = start * n
    var index_step = step * n
    var i = start
    while i != stop:
        var Bptr = B.unsafe_offset(index)
        var alpha = dot_prod(Bptr, z, n)
        alpha *= -2
        axpy(z, Bptr, alpha, n)
        index += index_step
        i += step


def householder_hornerscheme(
    z: Ptr,
    B: Ptr,
    y: Ptr,
    n: Int,
    start: Int,
    stop: Int,
    step: Int,
):
    """Map the least-squares solution y back through the Householder reflectors."""
    var index = start * n
    var index_step = step * n
    var i = start
    while i != stop:
        z.unsafe_store(i, z.unsafe_load(i) + y.unsafe_load(i))
        var Bptr = B.unsafe_offset(index)

        var alpha = dot_prod(Bptr, z, n)
        alpha *= -2
        axpy(z, Bptr, alpha, n)

        index += index_step
        i += step


def apply_givens(B: Ptr, x: Ptr, n: Int, nrot: Int):
    """Apply the first nrot Givens rotations in B to x, in sequence."""
    var ind1: Int = 0
    var ind2: Int = 1
    var ind3: Int = 2
    var ind4: Int = 3

    var rot: Int = 0
    while rot < nrot:
        # Apply rotation
        var x_temp = x.unsafe_load(rot)
        x.unsafe_store(rot, B.unsafe_load(ind1) * x_temp + B.unsafe_load(ind2) * x.unsafe_load(rot + 1))
        x.unsafe_store(rot + 1, B.unsafe_load(ind3) * x_temp + B.unsafe_load(ind4) * x.unsafe_load(rot + 1))

        # Increment indices
        ind1 += 4
        ind2 += 4
        ind3 += 4
        ind4 += 4
        rot += 1
