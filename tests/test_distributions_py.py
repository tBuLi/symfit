from __future__ import division
import pytest
import warnings

import sympy

from symfit import Variable, Parameter
from symfit.distributions import Gaussian, Exp


def test_gaussian():
    """
    Make sure that symfit.distributions.Gaussians produces the expected
    sympy expression.
    """
    x0 = Parameter()
    sig = Parameter(positive=True)
    x = Variable()

    new = sympy.exp(-(x - x0)**2/(2*sig**2))/sympy.sqrt((2*sympy.pi*sig**2))
    assert isinstance(new, sympy.Expr)
    g = Gaussian(x, x0, sig)
    assert issubclass(g.__class__, sympy.Expr)
    assert new == g

    # A pdf should always integrate to 1 on its domain
    assert sympy.integrate(g, (x, -sympy.oo, sympy.oo)) == 1

def test_exp():
    """
    Make sure that symfit.distributions.Exp produces the expected
    sympy expression.
    """
    l = Parameter(positive=True)
    x = Variable()

    new = l * sympy.exp(- l * x)
    assert isinstance(new, sympy.Expr)
    e = Exp(x, l)
    assert issubclass(e.__class__, sympy.Expr)
    assert new == e

    # A pdf should always integrate to 1 on its domain
    assert sympy.integrate(e, (x, 0, sympy.oo)) == 1