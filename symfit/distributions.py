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
    Gaussian pdf.
    :param x: free variable.
    :param mu: mean of the distribution.
    :param sig: standard deviation of the distribution.
    :return: sympy.Expr for a Gaussian pdf.
    """
    return sympy.exp(-(x - mu)**2/(2*sig**2))/sympy.sqrt(2*sympy.pi*sig**2)

def Exp(x, l):
    """
    Exponential Distribution pdf.
    :param x: free variable.
    :param l: rate parameter.
    :return: sympy.Expr for an Exponential Distribution pdf.
    """
    return l * sympy.exp(- l * x)

# def Beta():
#     sympy.stats.Beta()