"""Linear algebra helpers.  Mirrors `pyamg.util.linalg` for the covered subset."""

import numpy as np
import scipy as sp
from scipy import sparse
from scipy.linalg import eig
from scipy.sparse.linalg import aslinearoperator

from .params import set_tol
from .utils import get_diagonal


def norm(x, pnorm='2'):
    """2-norm or infinity-norm of a vector."""
    x = np.ravel(x)
    if pnorm == '2':
        return np.sqrt(np.inner(x.conj(), x).real)
    if pnorm == 'inf':
        return np.max(np.abs(x))
    raise ValueError('Only the 2-norm and infinity-norm are supported')


def infinity_norm(A):
    """Largest absolute row sum of A."""
    if sparse.issparse(A):
        if A.format in ('csc', 'csr'):
            # avoid copying index and ptr arrays
            abs_A = A.__class__((np.abs(A.data), A.indices, A.indptr), shape=A.shape)
            return (abs_A @ np.ones((A.shape[1]), dtype=np.float64)).max()
        return (abs(A) @ np.ones((A.shape[1]), dtype=np.float64)).max()
    return np.dot(np.abs(A), np.ones((A.shape[1],), dtype=np.float64)).max()


def axpy(x, y, a=1.0):
    """y = y + a*x, in place on y."""
    y += a * x


def _approximate_eigenvalues(A, maxiter, symmetric=None, initial_guess=None):
    """Krylov (Arnoldi or Lanczos) approximation of A's dominant eigenvalues."""
    A = aslinearoperator(A)  # A could be dense or sparse, or something weird
    # Choose tolerance for deciding if break-down has occurred
    breakdown = set_tol(A.dtype)
    breakdown_flag = False
    if A.shape[0] != A.shape[1]:
        raise ValueError('expected square matrix')
    maxiter = min(A.shape[0], maxiter)
    if initial_guess is None:
        v0 = np.random.rand(A.shape[1], 1)
        if A.dtype == complex:
            v0 = v0 + 1.0j * np.random.rand(A.shape[1], 1)
    else:
        v0 = initial_guess
    v0 = v0 / norm(v0)
    # Important to type H based on v0, so that a real nonsymmetric matrix, can
    # have an imaginary initial guess for its Arnoldi Krylov space
    H = np.zeros((maxiter + 1, maxiter), dtype=np.float64)
    V = [v0]
    beta = 0.0
    for j in range(maxiter):
        w = A @ V[-1]
        if symmetric:
            if j >= 1:
                H[j - 1, j] = beta
                w -= beta * V[-2]
            alpha = np.dot(np.conjugate(w.ravel()), V[-1].ravel())
            H[j, j] = alpha
            w -= alpha * V[-1]  # axpy(V[-1],w,-alpha)
            beta = norm(w)
            H[j + 1, j] = beta
            if (H[j + 1, j] < breakdown):
                breakdown_flag = True
                break
            w /= beta
            V.append(w)
            V = V[-2:]  # retain only last two vectors
        else:
            # orthogonalize against Vs
            for i, v in enumerate(V):
                H[i, j] = np.dot(np.conjugate(v.ravel()), w.ravel())
                w = w - H[i, j] * v
            H[j + 1, j] = norm(w)
            if (H[j + 1, j] < breakdown):
                breakdown_flag = True
                if H[j + 1, j] != 0.0:
                    w = w / H[j + 1, j]
                V.append(w)
                break
            w = w / H[j + 1, j]
            V.append(w)
    Eigs, Vects = eig(H[:j + 1, :j + 1], left=False, right=True)
    return (Vects, Eigs, H, V, breakdown_flag)


def approximate_spectral_radius(A, tol=0.01, maxiter=15, restart=5,
                                symmetric=None, initial_guess=None,
                                return_vector=False):
    """Estimate the spectral radius of A with a restarted Krylov iteration.

    The result is cached on the matrix as ``A.rho``; pass ``return_vector`` (or
    a matrix without the attribute) to force a fresh estimate.
    """
    if not hasattr(A, 'rho') or return_vector:
        # The use of the restart vector v0 requires that the full Krylov
        # subspace V be stored.  So, set symmetric to False.
        symmetric = False
        if maxiter < 1:
            raise ValueError('expected maxiter > 0')
        if restart < 0:
            raise ValueError('expected restart >= 0')
        if A.dtype == int:
            raise ValueError('expected A to be float (complex or real)')
        if A.shape[0] != A.shape[1]:
            raise ValueError('expected square A')
        if initial_guess is None:
            v0 = np.random.rand(A.shape[1], 1)
            if A.dtype == complex:
                v0 = v0 + 1.0j * np.random.rand(A.shape[1], 1)
        else:
            if initial_guess.shape[0] != A.shape[0]:
                raise ValueError('initial_guess and A must have same shape')
            if (len(initial_guess.shape) > 1) and (initial_guess.shape[1] > 1):
                raise ValueError('initial_guess must be an (n,1) or (n,) vector')
            v0 = initial_guess.reshape(-1, 1)
            v0 = np.array(v0, dtype=np.float64)
        for j in range(restart + 1):
            [evect, ev, H, V, breakdown_flag] = \
                _approximate_eigenvalues(A, maxiter, symmetric, initial_guess=v0)
            # Calculate error in dominant eigenvector
            nvecs = ev.shape[0]
            max_index = np.abs(ev).argmax()
            error = H[nvecs, nvecs - 1] * evect[-1, max_index]
            v0 = np.dot(np.hstack(V[:-1]), evect[:, max_index].reshape(-1, 1))
            if np.abs(error) / np.abs(ev[max_index]) < tol:
                # halt if below relative tolerance
                break
            if breakdown_flag:
                break
        # end j-loop
        rho = np.abs(ev[max_index])
        if sparse.issparse(A):
            A.rho = rho
        if return_vector:
            return (rho, v0)
        return rho
    return A.rho


def condest(A, maxiter=25, symmetric=False):
    """Estimate the condition number of A with `approximate_spectral_radius`."""
    x = np.ones((A.shape[0], 1))
    T = A
    while True:
        Tabs = T
        rho = approximate_spectral_radius(Tabs, tol=0.01, maxiter=maxiter)
        T = A @ (Tabs @ x)
        if symmetric:
            T = T.T
        r1 = rho * norm(x)
        r2 = norm(T, 'inf')
        x = (T @ x) / np.sqrt(np.sum((T @ x) ** 2))
        if r1 <= r2:
            break
    return 1.0 / rho


def cond(A):
    """Condition number of A: 1 / approximate spectral radius of inv(A)."""
    return condest(sparse.linalg.spsolve(A, np.eye(A.shape[0], dtype=np.float64)))


def ishermitian(A, fast_check=True, tol=1e-6, verbose=False):
    """True if A is (numerically) hermitian."""
    # convert to array type
    if not sparse.issparse(A):
        A = np.asarray(A)
    if fast_check:
        x = np.random.rand(A.shape[0])
        y = np.random.rand(A.shape[0])
        xAy = np.dot((A.dot(x)).conjugate().T, y)
        xAty = np.dot(x.conjugate().T, A.dot(y))
        diff = float(np.abs(xAy - xAty) / np.sqrt(np.abs(xAy * xAty)))
    else:
        # compute the difference, A - A.conj().T
        if sparse.issparse(A):
            diff = np.ravel((A - A.conj().T).data)
        else:
            diff = np.ravel(A - A.conj().T)
        if np.max(diff.shape) == 0:
            diff = 0
        else:
            diff = np.max(np.abs(diff))
    if diff < tol:
        return True
    if verbose:
        print(diff)
    return False


def pinv_array(a, tol=None):
    """Moore-Penrose pseudo inverse of a dense 2-D array, in place."""
    u, s, vt = sp.linalg.svd(a, full_matrices=False)
    s = np.atleast_2d(s.T)
    tol = s.max() * (a.shape[0] * np.finfo(s.dtype).eps) if tol is None else tol
    si = np.where(s <= tol, 0.0, 1.0 / s)
    a[...] = np.dot(vt.T * si, u.T)
    return a


__all__ = ['norm', 'infinity_norm', 'axpy', 'approximate_spectral_radius',
           'condest', 'cond', 'ishermitian', 'pinv_array', 'get_diagonal']
