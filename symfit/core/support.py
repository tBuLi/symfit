"""
This module contains support functions and convenience methods used
throughout symfit. Some are used predominantly internally, others are
designed for users.
"""
from __future__ import print_function
from collections import OrderedDict
import sys
import warnings
import re
import keyword

import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy

from sympy.tensor import Idx
from sympy import symbols, MatrixExpr
from sympy.core.expr import Expr

from symfit.core.argument import Parameter, Variable
from symfit.core.printing import SymfitNumPyPrinter

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


def isidentifier(s):
    if hasattr(s, 'isidentifier'):
        return s.isidentifier()
    else:
        # In py27 no such method exists, so we built one ourselves. Notice that
        # this cannot be used by default because py3 supports unicode
        # identifiers.
        if s in keyword.kwlist:
            return False
        return re.match(r'^[a-z_][a-z0-9_]*$', s, re.I) is not None


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
        if not isidentifier(str(symbol)):
            continue  # E.g. Indexed objects might print to A[i, j]
        if isinstance(symbol, Parameter):
            params.append(symbol)
        elif isinstance(symbol, Idx):
            # Idx objects are not seen as parameters or vars.
            pass
        elif isinstance(symbol, (MatrixExpr, Expr)):
            vars.append(symbol)
        else:
            raise TypeError('model contains an unknown symbol type, {}'.format(type(symbol)))

    for der in func.atoms(sympy.Derivative):
        # Used by jacobians and hessians, where derivatives are treated as
        # Variables. This way of writing it is purposefully discriminatory
        # against derivatives wrt variables, since such derivatives should be
        # performed explicitly in the case of jacs/hess, and are treated
        # differently in the case of ODEModels.
        if der.expr in vars and all(isinstance(s, Parameter) for s in der.variables):
            vars.append(der)

    params.sort(key=lambda symbol: symbol.name)
    vars.sort(key=lambda symbol: symbol.name)
    return vars, params

def sympy_to_py(func, args):
    """
    Turn a symbolic expression into a Python lambda function,
    which has the names of the variables and parameters as it's argument names.

    :param func: sympy expression
    :param args: variables and parameters in this model
    :return: lambda function to be used for numerical evaluation of the model.
    """
    # replace the derivatives with printable variables.
    derivatives = {var: Variable(var.name) for var in args
                   if isinstance(var, sympy.Derivative)}
    func = func.xreplace(derivatives)
    args = [derivatives[var] if isinstance(var, sympy.Derivative) else var
            for var in args]
    lambdafunc = lambdify(args, func, printer=SymfitNumPyPrinter,
                          dummify=False)
    # Check if the names of the lambda function are what we expect
    signature = inspect_sig.signature(lambdafunc)
    sig_parameters = OrderedDict(signature.parameters)
    for arg, lambda_arg in zip(args, sig_parameters):
        if arg.name != lambda_arg:
            break
    else:  # Lambdifying succesful!
        return lambdafunc

    # If we are here (very rare), then one of the lambda arg is still a Dummy.
    # In this case we will manually handle the naming.
    lambda_names = sig_parameters.keys()
    arg_names = [arg.name for arg in args]
    conversion = dict(zip(arg_names, lambda_names))

    # Wrap the lambda such that arg names are translated into the correct dummy
    # symbol names
    @wraps(lambdafunc)
    def wrapped_lambdafunc(*ordered_args, **kwargs):
        converted_kwargs = {conversion[k]: v for k, v in kwargs.items()}
        return lambdafunc(*ordered_args, **converted_kwargs)

    # Update the signature of wrapped_lambdafunc to math our args
    new_sig_parameters = OrderedDict()
    for arg_name, dummy_name in conversion.items():
        if arg_name == dummy_name:  # Already has the correct name
            new_sig_parameters[arg_name] = sig_parameters[arg_name]
        else:  # Change the dummy inspect.Parameter to the correct name
            param = sig_parameters[dummy_name]
            param = param.replace(name=arg_name)
            new_sig_parameters[arg_name] = param

    wrapped_lambdafunc.__signature__ = signature.replace(
        parameters=new_sig_parameters.values()
    )
    return wrapped_lambdafunc

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
    base_str = '_cached'
    def __init__(self, *args, **kwargs):
        super(cached_property, self).__init__(*args, **kwargs)
        self.cache_attr = '{}_{}'.format(self.base_str, self.fget.__name__)

    def __get__(self, obj, objtype=None):
        """
        In case of a first call, this will call the decorated function and
        return it's output. On every subsequent call, the same output will be
        returned.

        :param obj: the parent object this property is attached to.
        :param objtype:
        :return: Output of the first call to the decorated function.
        """
        try:
            return getattr(obj, self.cache_attr)
        except AttributeError:
            # Call the wrapped function with the obj instance as argument
            setattr(obj, self.cache_attr, self.fget(obj))
            return getattr(obj, self.cache_attr)

    def __delete__(self, obj):
        """
        Calling delete on the attribute will delete the cache.
        :param obj: parent object.
        """
        try:
            delattr(obj, self.cache_attr)
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


def D(*args, **kwargs):
    # ToDo: Investigate sympy's inheritance properly to see if I can give a
    # .name atribute to Derivative objects or subclasses.
    return sympy.Derivative(*args, **kwargs)

def name(self):
    """
    Save name which can be used for alphabetic sorting and can be turned
    into a kwarg.
    """
    base_str = 'd{}{}_'.format(self.derivative_count if
                               self.derivative_count > 1 else '', self.expr)
    for var, count in self.variable_count:
        base_str += 'd{}{}'.format(var,  count if count > 1 else '')
    return base_str

sympy.Derivative.name = property(name)