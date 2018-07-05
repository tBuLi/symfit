from __future__ import division, print_function
import pickle
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

    def test_pickle(self):
        """
        Make sure attributes are preserved when pickling
        """
        A = Parameter('A', min=0., max=1e3, fixed=True)
        new_A = pickle.loads(pickle.dumps(A))
        self.assertEqual((A.min, A.value, A.max, A.fixed, A.name),
                         (new_A.min, new_A.value, new_A.max, new_A.fixed, new_A.name))

        A = Parameter(min=0., max=1e3, fixed=True)
        new_A = pickle.loads(pickle.dumps(A))
        self.assertEqual((A.min, A.value, A.max, A.fixed, A.name),
                         (new_A.min, new_A.value, new_A.max, new_A.fixed, new_A.name))

    def test_slots(self):
        """
        Make sure Parameters and Variables don't have a __dict__
        """
        P = Parameter('P')

        # If you only have __slots__ you can't set arbitrary attributes, but
        # you *should* be able to set those that are in your __slots__
        try:
            P.min = 0
        except AttributeError:
            self.fail()

        with self.assertRaises(AttributeError):
            P.foo = None

        V = Variable('V')
        with self.assertRaises(AttributeError):
            V.bar = None


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
