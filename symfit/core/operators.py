"""
This module makes sympy Epressions callable, which makes the whole project feel more consistent.
"""

from sympy import Eq, Ne
from sympy.core.expr import Expr
import warnings
from symfit.core.support import sympy_to_py, seperate_symbols
from symfit.core.argument import Parameter

# # Overwrite the behavior opun equality checking. But we want to be able to fall
# # back on default behavior.
# orig_eq = Expr.__class__.__eq__
# orig_ne = Expr.__class__.__ne__
#
# def eq(self, other):
#     """
#     Hack to get an Eq object back. Seems to work when used this way,
#     but backwards compatibility with sympy is not guaranteed.
#     """
#     if isinstance(other, float) or isinstance(other, int):
#         if abs(other) == 1:
#             # SymPy's printing check for this and might therefore produce an
#             # error for fractions. Therefore we raise a warning in this case and
#             # ask the user to use the Eq object manually.
#             warnings.warn(str(self) + " == -1 and == 1 are not available for constraints. If you used +/-1 as a constraint, please use the symfit.Eq object manually.", UserWarning)
#             return orig_eq(self.__class__, other)
#         else:
#             return Eq(self, other)
#     else:
#         return orig_eq(self.__class__, other)
#
# def ne(self, other):
#     if isinstance(other, float) or isinstance(other, int):
#         return Ne(self, other)
#     else:
#         return orig_ne(self.__class__, other)

def call(self, **values):
    """
    :param self: Any subclass of sympy.Expr
    :param values: Values for the Parameters and Variables of the Expr.
    :return: The function evaluated at ``values``. Depending on the Expr and ``values``, this could be a single number or an array.
    """
    # Convert to a pythonic function
    vars, params = seperate_symbols(self)
    func = sympy_to_py(self, vars, params)
    # Prepare only the relevant values
    arg_names = [arg.name for arg in params + vars]
    args = dict([(name, value) for name, value in values.items() if name in arg_names])
    return func(**args)


# Expr.__eq__ = eq
# Expr.__ne__ = ne
Expr.__call__ = call
Parameter.__call__ = call