"""
For whatever reason, sympy overloaded <, <=, > and >= to be symbolic by default
but == and != are pythonic by default. I find this inconsistent, and it does not
allow this package to use the beautiful syntax I want for constraints, e.g.

constraints = [
    x**3 - y == 0,
    y - 1 >= 0,
]

Importing this file overwrites the default behavior to be symbolic.
"""

from sympy import Eq, Ne
from sympy.core.expr import Expr
import warnings

# Overwrite the behavior opun equality checking. But we want to be able to fall
# back on default behavior.
orig_eq = Expr.__class__.__eq__
orig_ne = Expr.__class__.__ne__

def eq(self, other):
    """
    Hack to get an Eq object back. Seems to work when used this way,
    but backwards compatibility with sympy is not guaranteed.
    """
    if isinstance(other, float) or isinstance(other, int):
        if other == -1:
            # SymPy's printing check for this and might therefore produce an
            # error for fractions. Therefore we raise a warning in this case and
            # ask the user to use the Eq object manually.
            warnings.warn(str(self) + " == -1 is not available for fractions. If you used -1 as a constraint, please use the symfit.Eq object manually.", UserWarning)
            return orig_eq(self.__class__, other)
        else:
            return Eq(self, other)
    else:
        return orig_eq(self.__class__, other)

def ne(self, other):
    if isinstance(other, float) or isinstance(other, int):
        return Ne(self, other)
    else:
        return orig_ne(self.__class__, other)

Expr.__eq__ = eq
Expr.__ne__ = ne