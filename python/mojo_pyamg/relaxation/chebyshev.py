"""Compute coefficients for polynomial smoothers.  Mirrors `pyamg.relaxation.chebyshev`."""

import numpy as np


def chebyshev_polynomial_coefficients(a, b, degree):
    """Chebyshev polynomial coefficients for the interval [a,b].

    Parameters
    ----------
    a,b : float
        The left and right endpoints of the interval.
    degree : int
        Degree of desired Chebyshev polynomial.

    Returns
    -------
    Coefficients of the Chebyshev polynomial C(t) with minimum magnitude on the
    interval [a,b] such that C(0) = 1.0, in descending order.

    Examples
    --------
    >>> print(chebyshev_polynomial_coefficients(1.0, 2.0, 3))
    [-0.32323232  1.45454545 -2.12121212  1.        ]

    """
    if a >= b or a <= 0:
        raise ValueError(f'invalid interval [{a},{b}]')

    # Chebyshev roots for the interval [-1,1]
    std_roots = np.cos(np.pi * (np.arange(degree) + 0.5) / degree)

    # Chebyshev roots for the interval [a,b]
    scaled_roots = 0.5 * (b - a) * (1 + std_roots) + a

    # Compute monic polynomial coefficients of polynomial with scaled roots
    scaled_poly = np.poly(scaled_roots)

    # Scale coefficients to enforce C(0) = 1.0
    scaled_poly /= np.polyval(scaled_poly, 0)

    return scaled_poly


def mls_polynomial_coefficients(rho, degree):
    """Coefficients for a multilevel-smoothed MLS polynomial smoother.

    Parameters
    ----------
    rho : float
        Spectral radius of the matrix in question.
    degree : int
        Degree of polynomial coefficients to generate.

    Returns
    -------
    (coeffs, roots) for the (symmetric) polynomial smoother and the roots of
    the polynomial prolongation smoother.  The coefficients are in descending
    order.

    References
    ----------
    .. [1] Parallel multigrid smoothing: polynomial versus Gauss--Seidel
       M. F. Adams, M. Brezina, J. J. Hu, and R. S. Tuminaro
       J. Comp. Phys., 188 (2003), pp. 593--610

    """
    roots = rho / 2.0 * \
        (1.0 - np.cos(2 * np.pi * (np.arange(degree, dtype='float64') + 1)
                      / (2.0 * degree + 1.0)))
    roots = 1.0 / roots

    S = np.poly(roots)[::-1]  # monomial coefficients of S error propagator

    SSA_max = rho / ((2.0 * degree + 1.0) ** 2)  # upper bound spectral radius of S^2A
    S_hat = np.polymul(S, S)  # monomial coefficients of \hat{S} propagator
    S_hat = np.hstack(((-1.0 / SSA_max) * S_hat, [1]))

    # coeff for combined error propagator \hat{S}S
    coeffs = np.polymul(S_hat, S)
    coeffs = -coeffs[:-1]             # coeff for smoother

    return (coeffs, roots)


__all__ = ['chebyshev_polynomial_coefficients', 'mls_polynomial_coefficients']
