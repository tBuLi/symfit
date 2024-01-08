try:
    # As a commercial solver, gurobi is optional.
    from .gurobi import GurobiBackend
except ImportError:
    pass

from .scipopt import SCIPOptBackend
