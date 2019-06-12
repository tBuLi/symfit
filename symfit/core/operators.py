"""
Monkey Patching module.

This module makes ``sympy`` Expressions callable, which makes the whole project feel more consistent.
"""
import sys

from sympy import Eq, Ne, MatrixSymbol, Tuple
from sympy.core.expr import Expr
import sympy
import warnings
from symfit.core.support import sympy_to_py, seperate_symbols
from symfit.core.argument import Parameter

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig

def call(self, *values, **named_values):
    """
    Call an expression to evaluate it at the given point.

    Future improvements: I would like if func and signature could be buffered after the
    first call so they don't have to be recalculated for every call. However, nothing
    can be stored on self as sympy uses __slots__ for efficiency. This means there is no
    instance dict to put stuff in! And I'm pretty sure it's ill advised to hack into the
    __slots__ of Expr.

    However, for the moment I don't really notice a performance penalty in running tests.

    p.s. In the current setup signature is not even needed since no introspection is possible
    on the Expr before calling it anyway, which makes calculating the signature absolutely useless.
    However, I hope that someday some monkey patching expert in shining armour comes by and finds
    a way to store it in __signature__ upon __init__ of any ``symfit`` expr such that calling
    inspect_sig.signature on a symbolic expression will tell you which arguments to provide.

    :param self: Any subclass of sympy.Expr
    :param values: Values for the Parameters and Variables of the Expr.
    :param named_values: Values for the vars and params by name. ``named_values`` is
        allowed to contain too many values, as this sometimes happens when using
        \*\*fit_result.params on a submodel. The irrelevant params are simply ignored.
    :return: The function evaluated at ``values``. The type depends entirely on the input.
        Typically an array or a float but nothing is enforced.
    """
    independent_vars, params = seperate_symbols(self)
    # Convert to a pythonic function
    func = sympy_to_py(self, independent_vars + params)

    # Handle args and kwargs according to the allowed names.
    parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
        inspect_sig.Parameter(arg.name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD)
            for arg in independent_vars + params
    ]

    arg_names = [arg.name for arg in independent_vars + params]
    relevant_named_values = {
        name: value for name, value in named_values.items() if name in arg_names
    }

    signature = inspect_sig.Signature(parameters=parameters)
    bound_arguments = signature.bind(*values, **relevant_named_values)

    return func(**bound_arguments.arguments)


Expr.__call__ = call
Parameter.__call__ = call

# Monkeypatch MatrixSymbols because they are currently broken
def _eval_subs(self, old, new):
    # only do substitutions in shape
    shape = Tuple(*self.shape)._subs(old, new)
    return MatrixSymbol(self.args[0], *shape)

def doit(self, **hints):
    if hints.get('deep', True):
        return type(self)(self.args[0], self.args[1].doit(**hints),
                self.args[2].doit(**hints))
    else:
        return self

MatrixSymbol._eval_subs = _eval_subs
MatrixSymbol.doit = doit