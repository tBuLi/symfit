from __future__ import division
import unittest
import warnings

import sympy

from symfit import Variable, Parameter
from symfit.distributions import Gaussian, Exp


class TestDistributions(unittest.TestCase):
    def test_gaussian(self):
        """
        Make sure that symfit.distributions.Gaussians produces the expected
        sympy expression.
        """
        x0 = Parameter()
        sig = Parameter(positive=True)
        x = Variable()

        new = sympy.exp(-(x - x0)**2/(2*sig**2))/sympy.sqrt((2*sympy.pi*sig**2))
        self.assertIsInstance(new, sympy.Expr)
        g = Gaussian(x, x0, sig)
        self.assertTrue(issubclass(g.__class__, sympy.Expr))
        self.assertEqual(new, g)

        # A pdf should always integrate to 1 on its domain
        self.assertEqual(sympy.integrate(g, (x, -sympy.oo, sympy.oo)), 1)

    def test_exp(self):
        """
        Make sure that symfit.distributions.Exp produces the expected
        sympy expression.
        """
        l = Parameter(positive=True)
        x = Variable()

        new = l * sympy.exp(- l * x)
        self.assertIsInstance(new, sympy.Expr)
        e = Exp(x, l)
        self.assertTrue(issubclass(e.__class__, sympy.Expr))
        self.assertEqual(new, e)

        # A pdf should always integrate to 1 on its domain
        self.assertEqual(sympy.integrate(e, (x, 0, sympy.oo)), 1)

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