# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

# Overwrite behavior of sympy objects to make more sense for this project.
import symfit.core.operators

# Expose useful objects.
from symfit.core.fit import Fit
from symfit.core.models import (
    Model, ODEModel, ModelError, CallableModel, CallableNumericalModel,
    GradientModel
)
from symfit.core.fit_results import FitResults
from symfit.core.argument import Variable, Parameter
from symfit.core.support import variables, parameters, D

try:
    # MIP is an optional feature. If no solvers are installed this will raise an import error.
    from symfit.symmip import MIP
except ImportError:
    pass

# Expose the sympy API
from sympy import *