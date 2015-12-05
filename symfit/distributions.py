"""
Some common functions are defined in this file. That way, users can easily build
more complicated expressions without making them look hard.

We have delibaratelly choosen to start these function with a capital, e.g.
Gaussian instead of gaussian, because this makes the resulting expressions more
readable.
"""
import sympy

def Gaussian(x, mu, sig):
    return sympy.exp(-(x - mu)**2/(2*sig**2))/sympy.sqrt(2*sympy.pi*sig**2)

def Exp(x, l):
    return l * sympy.exp(- l * x)

# def Beta():
#     sympy.stats.Beta()