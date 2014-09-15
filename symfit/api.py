# Overwrite == and != behavior to be symbolic if rhs is a number.
import symfit.core.operators

# Expose useful objects.
from symfit.core.fit import Minimize, Fit, FitResults
from symfit.core.argument import Variable, Parameter

# Expose the sympy API
from sympy import *