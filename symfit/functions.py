"""
Some common functions are defined in this file. That way, users can easely build
more complicated expressions without making them look hard.

We have delibaratelly choosen to start these function with a capital, e.g.
Gaussian instead of gaussian, because this makes the resulting expressions more
readable.
"""
import sympy

def Gaussian(x, x0, sig):
    return sympy.exp(-(x - x0)**2/(2*sig**2))#/(2*sympy.pi*sig)

def Exp(x, x0, k):
    return sympy.exp(k * (x - x0))

# def Beta():
#     sympy.stats.Beta()