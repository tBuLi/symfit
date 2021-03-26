# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

"""
Some common distributions are defined in this module. That way, users can easily build
more complicated expressions without making them look hard.

I have deliberately chosen to start these function with a capital, e.g.
Gaussian instead of gaussian, because this makes the resulting expressions more
readable.
"""
import sympy

def Gaussian(x, mu, sig):
    """
    .. math::

        f(x) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}} e^{- \\frac{(x - \\mu)^2}{2 \\sigma^2}}

    Gaussian pdf.

    :param x: free variable.
    :param mu: mean of the distribution.
    :param sig: standard deviation of the distribution.
    :return: sympy.Expr for a Gaussian pdf.
    """
    return sympy.exp(-(x - mu)**2/(2*sig**2))/sympy.sqrt(2*sympy.pi*sig**2)

def BivariateGaussian(x, y, mu_x, mu_y, sig_x, sig_y, rho):
    """
    `Bivariate Gaussian pdf
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_.

    :param x: :class:`symfit.core.argument.Variable`
    :param y: :class:`symfit.core.argument.Variable`
    :param mu_x: :class:`symfit.core.argument.Parameter` for the mean of `x`
    :param mu_y: :class:`symfit.core.argument.Parameter` for the mean of `y`
    :param sig_x: :class:`symfit.core.argument.Parameter` for the standard
        deviation of `x`
    :param sig_y: :class:`symfit.core.argument.Parameter` for the standard
        deviation of `y`
    :param rho: :class:`symfit.core.argument.Parameter` for the correlation
        between `x` and `y`.
    :return: sympy expression for a Bivariate Gaussian pdf.
    """
    exponent = - 1 / (2 * (1 - rho**2))
    exponent *= (x - mu_x)**2 / sig_x**2 + (y - mu_y)**2 / sig_y**2 \
                - 2 * rho * (x - mu_x) * (y - mu_y) / (sig_x * sig_y)
    return sympy.exp(exponent) / (2 * sympy.pi * sig_x * sig_y * sympy.sqrt(1 - rho**2))

def Exp(x, l):
    """
    .. math::

        f(x) = l e^{- l x}

    Exponential Distribution pdf.

    :param x: free variable.
    :param l: rate parameter.
    :return: sympy.Expr for an Exponential Distribution pdf.
    """
    return l * sympy.exp(- l * x)

# def Beta():
#     sympy.stats.Beta()