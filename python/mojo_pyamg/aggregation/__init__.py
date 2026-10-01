"""Aggregation-based AMG.  Mirrors `pyamg.aggregation` for the covered subset."""

from .aggregate import (naive_aggregation, pairwise_aggregation,
                        standard_aggregation)
from .aggregation import smoothed_aggregation_solver
from .tentative import fit_candidates
from .smooth import (jacobi_prolongation_smoother, richardson_prolongation_smoother,
                     satisfy_constraints)

__all__ = [
    'fit_candidates',
    'jacobi_prolongation_smoother',
    'naive_aggregation',
    'pairwise_aggregation',
    'richardson_prolongation_smoother',
    'satisfy_constraints',
    'smoothed_aggregation_solver',
    'standard_aggregation',
]
