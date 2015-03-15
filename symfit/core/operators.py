"""
Point 1:
For whatever reason, sympy overloaded <, <=, > and >= to be symbolic by default
but == and != are used for structural equality by default.
I find this inconsistent, and it does not allow this package to use the
beautiful syntax I want for constraints, e.g.

constraints = [
    x**3 - y == 0,
    y - 1 >= 0,
]

Importing this file overwrites the default behavior to be symbolic.

Point 2:
Make the sympy Epressions callable, which makes the whole project feel more
consistent.
"""

from sympy import Eq, Ne
from sympy.core.expr import Expr
import warnings
from symfit.core.support import sympy_to_py, seperate_symbols

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
        if abs(other) == 1:
            # SymPy's printing check for this and might therefore produce an
            # error for fractions. Therefore we raise a warning in this case and
            # ask the user to use the Eq object manually.
            warnings.warn(str(self) + " == -1 and == 1 are not available for constraints. If you used +/-1 as a constraint, please use the symfit.Eq object manually.", UserWarning)
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

def call(self, **kwargs):
    """
    :param self: Any subclass of sympy.Expr
    :param args:
    :param kwargs:
    :return:
    """
    # Convert to a pythonic function
    vars, params = seperate_symbols(self)
    func = sympy_to_py(self, vars, params)
    # Prepare only the relevant kwargs
    arg_names = [arg.name for arg in params + vars]
    args = dict([(name, value) for name, value in kwargs.items() if name in arg_names])
    return func(**args)


# Expr.__eq__ = eq
# Expr.__ne__ = ne
Expr.__call__ = call