# Overwrite behavior of sympy objects to make more sense for this project.
import symfit.core.operators

# Expose useful objects.
from symfit.core.fit import Fit, FitResults, Maximize, Minimize, Likelihood, Model, NumericalLeastSquares
from symfit.core.argument import Variable, Parameter
from symfit.core.support import variables, parameters

# Expose the sympy API
from sympy import *