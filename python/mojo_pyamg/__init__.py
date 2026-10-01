"""mojo-pyamg: algebraic multigrid solvers, ported to Mojo.

Mirrors the covered subset of PyAMG's API: see the README for the coverage
table, including what is deliberately not ported.
"""

from . import aggregation
from . import krylov
from . import relaxation
from . import strength
from . import util

from .multilevel import MultilevelSolver, coarse_grid_solver, multilevel_solver
from .aggregation import smoothed_aggregation_solver

__version__ = '0.1.0'

__all__ = [
    'MultilevelSolver',
    '__version__',
    'aggregation',
    'coarse_grid_solver',
    'krylov',
    'multilevel_solver',
    'relaxation',
    'smoothed_aggregation_solver',
    'strength',
    'util',
]
