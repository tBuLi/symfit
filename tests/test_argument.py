from __future__ import division, print_function
import pickle
import pytest
import sys
import sympy
import warnings

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit, minimize

from symfit import (
    Variable, Parameter, Fit, FitResults, log, variables,
    parameters, Model, Eq, Ge
)
from symfit.core.minimizers import BFGS, MINPACK, SLSQP, LBFGSB
from symfit.core.objectives import LogLikelihood
from symfit.distributions import Gaussian, Exp

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


def test_parameter_add():
        """
        Makes sure the __add__ method of Parameters behaves as expected.
        """
        a = Parameter(value=1.0, min=0.5, max=1.5)
        b = Parameter(value=1.0, min=0.0)
        new = a + b
        assert isinstance(new, sympy.Add)

def test_argument_unnamed():
        """
        Make sure the generated parameter names follow the pattern
        """
        a = Parameter()
        b = Parameter('b', 10)
        c = Parameter(value=10)
        x = Variable()
        y = Variable('y')

        assert str(a) == '{}_{}'.format(a._argument_name, a._argument_index)
        assert str(a) == 'par_{}'.format(a._argument_index)
        assert str(b) != '{}_{}'.format(b._argument_name, b._argument_index)
        assert str(c) == '{}_{}'.format(c._argument_name, c._argument_index)
        assert c.value == 10
        assert b.value == 10
        assert str(x) == 'var_{}'.format(x._argument_index)
        assert str(y) == 'y'

        with pytest.raises(TypeError):
            d = Parameter(10)


def test_pickle():
        """
        Make sure attributes are preserved when pickling
        """
        A = Parameter('A', min=0., max=1e3, fixed=True)
        new_A = pickle.loads(pickle.dumps(A))
        assert (A.min, A.value, A.max, A.fixed, A.name) == (new_A.min, new_A.value, new_A.max, new_A.fixed, new_A.name)

        A = Parameter(min=0., max=1e3, fixed=True)
        new_A = pickle.loads(pickle.dumps(A))
        assert (A.min, A.value, A.max, A.fixed, A.name) == (new_A.min, new_A.value, new_A.max, new_A.fixed, new_A.name)


def test_slots():
        """
        Make sure Parameters and Variables don't have a __dict__
        """
        P = Parameter('P')

        # If you only have __slots__ you can't set arbitrary attributes, but
        # you *should* be able to set those that are in your __slots__
        try:
            P.min = 0
        except AttributeError:
            assert False

        with pytest.raises(AttributeError):
            P.foo = None

        V = Variable('V')
        with pytest.raises(AttributeError):
            V.bar = None

