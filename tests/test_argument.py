from __future__ import division, print_function
import unittest
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


class TestArgument(unittest.TestCase):
    def test_parameter_add(self):
        """
        Makes sure the __add__ method of Parameters behaves as expected.
        """
        a = Parameter(value=1.0, min=0.5, max=1.5)
        b = Parameter(value=1.0, min=0.0)
        new = a + b
        self.assertIsInstance(new, sympy.Add)

    def test_argument_unnamed(self):
        """
        Make sure the generated parameter names follow the pattern
        """
        a = Parameter()
        b = Parameter('b', 10)
        c = Parameter(value=10)
        x = Variable()
        y = Variable('y')

        self.assertEqual(str(a), '{}_{}'.format(a._argument_name, a._argument_index))
        self.assertEqual(str(a), 'par_{}'.format(a._argument_index))
        self.assertNotEqual(str(b), '{}_{}'.format(b._argument_name, b._argument_index))
        self.assertEqual(str(c), '{}_{}'.format(c._argument_name, c._argument_index))
        self.assertEqual(c.value, 10)
        self.assertEqual(b.value, 10)
        self.assertEqual(str(x), 'var_{}'.format(x._argument_index))
        self.assertEqual(str(y), 'y')

        with self.assertRaises(TypeError):
            d = Parameter(10)


    def test_argument_name(self):
        """
        Make sure that Parameters have a name attribute with the expected
        value.
        """
        a = Parameter()
        b = Parameter(name='b')
        c = Parameter(name='d')
        self.assertNotEqual(a.name, 'a')
        self.assertEqual(b.name, 'b')
        self.assertEqual(c.name, 'd')

    def test_symbol_add(self):
        """
        Makes sure the __add__ method of symbols behaves as expected.
        """
        x, y = sympy.symbols('x y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

if __name__ == '__main__':
    try:
        unittest.main(warnings='ignore')
        # Note that unittest will catch and handle exceptions raised by tests.
        # So this line will *only* deal with exceptions raised by the line
        # above.
    except TypeError:
        # In Py2, unittest.main doesn't take a warnings argument
        warnings.simplefilter('ignore')
        unittest.main()
