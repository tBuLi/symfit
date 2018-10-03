"""
This module contains support functions and convenience methods used
throughout symfit. Some are used predominantly internally, others are
designed for users.
"""
from __future__ import print_function
from collections import OrderedDict
import sys
import warnings

import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from sympy.tensor import Idx
from sympy import symbols
from sympy.core.expr import Expr

from symfit.core.argument import Parameter, Variable

if sys.version_info >= (3,0):
    import inspect as inspect_sig
    from functools import wraps
else:
    import funcsigs as inspect_sig
    from functools32 import wraps

if sys.version_info >= (3, 5):
    from functools import partial
else:
    from ._repeatable_partial import repeatable_partial as partial

class deprecated(object):
    """
    Decorator to raise a DeprecationWarning.
    """
    def __init__(self, replacement=None):
        """
        :param replacement: The function which should now be used instead.
        """
        self.replacement = replacement

    def __call__(self, func):
        @wraps(func)
        def deprecated_func(*args, **kwargs):
            warnings.warn(DeprecationWarning(
                '`{}` has been deprecated.'.format(func.__name__)
                + ' Use `{}` instead.'.format(self.replacement)) if self.replacement else ''
            )
            return func(*args, **kwargs)
        return deprecated_func

def seperate_symbols(func):
    """
    Seperate the symbols in symbolic function func. Return them in alphabetical
    order.

    :param func: scipy symbolic function.
    :return: (vars, params), a tuple of all variables and parameters, each 
        sorted in alphabetical order.
    :raises TypeError: only symfit Variable and Parameter are allowed, not sympy
        Symbols.
    """
    params = []
    vars = []
    for symbol in func.free_symbols:
        if isinstance(symbol, Parameter):
            params.append(symbol)
        elif isinstance(symbol, Idx):
            # Idx objects are not seen as parameters or vars.
            pass
        elif isinstance(symbol, Expr):
            vars.append(symbol)
        else:
            raise TypeError('model contains an unknown symbol type, {}'.format(type(symbol)))
    params.sort(key=lambda symbol: symbol.name)
    vars.sort(key=lambda symbol: symbol.name)
    return vars, params

def sympy_to_py(func, vars, params):
    """
    Turn a symbolic expression into a Python lambda function,
    which has the names of the variables and parameters as it's argument names.

    :param func: sympy expression
    :param vars: variables in this model
    :param params: parameters in this model
    :return: lambda function to be used for numerical evaluation of the model. Ordering of the arguments will be vars
        first, then params.
    """
    return lambdify((vars + params), func, modules='numpy', dummify=False)

def sympy_to_scipy(func, vars, params):
    """
    Convert a symbolic expression to one scipy digs. Not used by ``symfit`` any more.

    :param func: sympy expression
    :param vars: variables
    :param params: parameters
    :return: Scipy-style function to be used for numerical evaluation of the model.
    """
    lambda_func = sympy_to_py(func, vars, params)
    def f(x, p):
        """
        Scipy style function.

        :param x: list of arrays, NxM
        :param p: tuple of parameter values.
        """
        x = np.atleast_2d(x)
        y = [x[i] for i in range(len(x))] if len(x[0]) else []
        try:
            ans = lambda_func(*(y + list(p)))
        except TypeError:
            # Possibly this is a constant function in which case it only has Parameters.
            ans = lambda_func(*list(p))# * np.ones(x_shape)
        return ans

    return f

def variables(names, **kwargs):
    """
    Convenience function for the creation of multiple variables. For more
    control, consider using ``symbols(names, cls=Variable, **kwargs)`` directly.

    :param names: string of variable names.
        Example: x, y = variables('x, y')
    :param kwargs: kwargs to be passed onto :func:`sympy.core.symbol.symbols`
    :return: iterable of :class:`symfit.core.argument.Variable` objects
    """
    return symbols(names, cls=Variable, seq=True, **kwargs)

def parameters(names, **kwargs):
    """
    Convenience function for the creation of multiple parameters. For more
    control, consider using ``symbols(names, cls=Parameter, **kwargs)`` directly.

    The `Parameter` attributes `value`, `min`, `max` and `fixed` can also be provided
    directly. If given as a single value, the same value will be set for all
    `Parameter`'s. When a sequence, it must be of the same length as the number of
    parameters created.

    Example::
        x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=0.0)

    :param names: string of parameter names.
        Example: a, b = parameters('a, b')
    :param kwargs: kwargs to be passed onto :func:`sympy.core.symbol.symbols`.
        `value`, `min` and `max` will be handled separately if they are sequences.
    :return: iterable of :class:`symfit.core.argument.Parameter` objects
    """
    sequence_fields = ['value', 'min', 'max', 'fixed']
    sequences = {}
    for attr in sequence_fields:
        try:
            iter(kwargs[attr])
        except (TypeError, KeyError):
            # Not iterable or not provided
            pass
        else:
            sequences[attr] = kwargs.pop(attr)

    if 'min' in sequences and 'max' in sequences:
        for min, max in zip(sequences['min'], sequences['max']):
            if min > max:
                raise ValueError('The value of `min` should be less than or'
                                 ' equal to the value of `max`.')

    params = symbols(names, cls=Parameter, seq=True, **kwargs)
    for key, values in sequences.items():
        try:
            assert len(values) == len(params)
        except AssertionError:
            raise ValueError(
                '`len` of keyword-argument `{}` does not match the number of '
                '`Parameter`s created.'.format(attr)
            )
        except TypeError:
            # Iterator do not have a `len` but are allowed.
            pass
        finally:
            for param, value in zip(params, values):
                setattr(param, key, value)
    return params


class cached_property(property):
    """
    A property which cashes the output of the first ever call and always returns
    that value from then on, unless delete is called on the attribute.

    This is typically used in converting `sympy` code into `scipy` compatible
    code, which is computationally a very expensive step we would like to
    perform only once.

    Does not allow setting of the attribute.
    """
    def __get__(self, obj, objtype=None):
        """
        In case of a first call, this will call the decorated function and
        return it's output. On every subsequent call, the same output will be
        returned.

        :param obj: the parent object this property is attached to.
        :param objtype:
        :return: Output of the first call to the decorated function.
        """
        cache_attr = '_{}'.format(self.fget.__name__)
        try:
            return getattr(obj, cache_attr)
        except AttributeError:
            # Call the wrapped function with the obj instance as argument
            setattr(obj, cache_attr, self.fget(obj))
            return getattr(obj, cache_attr)

    def __delete__(self, obj):
        """
        Calling delete on the attribute will delete the cache.
        :param obj: parent object.
        """
        cache_attr = '_{}'.format(self.fget.__name__)
        try:
            delattr(obj, cache_attr)
        except AttributeError:
            pass


def jacobian(expr, symbols):
    """
    Derive a symbolic expr w.r.t. each symbol in symbols. This returns a symbolic jacobian vector.

    :param expr: A sympy Expr.
    :param symbols: The symbols w.r.t. which to derive.
    """
    jac = []
    for symbol in symbols:
        # Differentiate to every param
        f = sympy.diff(expr, symbol)
        jac.append(f)
    return jac

def key2str(target):
    """
    In ``symfit`` there are many dicts with symbol: value pairs.
    These can not be used immediately as \*\*kwargs, even though this would make
    a lot of sense from the context.
    This function wraps such dict to make them usable as \*\*kwargs immediately.

    :param target: `Mapping` to be made save
    :return: `Mapping` of str(symbol): value pairs.
    """
    return target.__class__((str(symbol), value) for symbol, value in target.items())

class RequiredKeyword(object):
    """ Flag variable to indicate that this is a required keyword. """

class RequiredKeywordError(Exception):
    """ Error raised in case a keyword-only argument is not treated as such. """

class keywordonly(object):
    """
    Decorator class which wraps a python 2 function into one with keyword-only arguments.

    Example::

      @keywordonly(floor=True)
      def f(x, **kwargs):
          floor = kwargs.pop('floor')
          return np.floor(x**2) if floor else x**2

    This decorator is not much more than::

      floor = kwargs.pop('floor') if 'floor' in kwargs else True

    However, I prefer it's usage because:
 
    - it's clear from reading the function declaration there is an option to provide this 
      argument. The information on possible keywords is where you'd expect it to be.
    - you're guaranteed that the pop works.
    - It is fully inspect compatible such that sphynx is able to index these
      properly as keyword only arguments just like it would for native py3
      keyword only arguments.

    Please note that this decorator needs a ** argument on the wrapped function
    in order to work.
    """
    def __init__(self, **kwonly_arguments):
        self.kwonly_arguments = kwonly_arguments
        # Mark which are required
        self.required_keywords = {
            kw for kw, value in kwonly_arguments.items() if value is RequiredKeyword
        }
        # Transform all into keywordonly inspect.Parameter objects.
        self.keywordonly_parameters = OrderedDict(
            (kw, inspect_sig.Parameter(kw,
                                       kind=inspect_sig.Parameter.KEYWORD_ONLY,
                                       default=value)
             )
            for kw, value in kwonly_arguments.items()
        )

    def __call__(self, func):
        """
        Returns a decorated version of `func`, who's signature now includes the
        keyword-only arguments.

        :param func: the function to be decorated
        :return: the decorated function
        """
        sig = inspect_sig.signature(func)
        params = []
        # A var keyword has to be found for this function to be decorated
        for name, param in sig.parameters.items():
            if param.kind == param.VAR_KEYWORD:
                # Keyword only's go before the **kwargs parameter.
                params.extend(self.keywordonly_parameters.values())
                params.append(param)
                break
            params.append(param)
        else:
            raise RequiredKeywordError(
                'The keywordonly decorator requires the function to '
                'accept a **kwargs argument.'
            )
        # Update signature
        sig = sig.replace(parameters=params)
        func.__signature__ = sig

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            """
            :param args: args used to call the function
            :param kwargs: kwargs used to call the function
            :return: Wrapped function which behaves like it has keyword-only arguments.
            :raises: ``RequiredKeywordError`` if not all required keywords were specified.
            """
            bound_args = func.__signature__.bind(*args, **kwargs)
            # Apply defaults
            for param in sig.parameters.values():
                if param.name not in bound_args.arguments:
                    if param.default is RequiredKeyword:
                        raise RequiredKeywordError(
                            'Keyword `{}` is a required keyword. '
                            'Please provide a value.'.format(param.name)
                        )
                    elif param.kind == inspect_sig.Parameter.VAR_KEYWORD:
                        bound_args.arguments[param.name] = {}
                    elif param.kind == inspect_sig.Parameter.VAR_POSITIONAL:
                        bound_args.arguments[param.name] = tuple()
                    else:
                        bound_args.arguments[param.name] = param.default
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapped_func


class D(sympy.Derivative):
    """
    Convenience wrapper for ``sympy.Derivative``. Used most notably in defining
    ``ODEModel``'s.
    """
