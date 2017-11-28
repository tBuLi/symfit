# Overwrite behavior of sympy objects to make more sense for this project.
import symfit.core.operators

# Expose useful objects.
from symfit.core.fit import (
    Fit, Model, Constraint,
    LinearLeastSquares, NonLinearLeastSquares,
    TaylorModel, ODEModel, ModelError
)
from symfit.core.fit_results import FitResults
from symfit.core.argument import Variable, Parameter
from symfit.core.support import variables, parameters, D

# Expose the sympy API
from sympy import *