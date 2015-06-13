"""
This module contains support functions and conveniance methods used
throughout symfit. Some are used prodominantly internaly, others are 
designed for users.
"""
from functools import wraps

import numpy as np
from symfit.core.argument import Parameter, Variable
from sympy.utilities.lambdify import lambdify

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
    from sympy.tensor import IndexedBase
    from sympy import Symbol
    for symbol in func.free_symbols:
        if isinstance(symbol, Parameter):
            params.append(symbol)
        elif isinstance(symbol, (Variable, IndexedBase, Symbol)):
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
    :return: lambda function to be used for numerical evaluation of the model.
    """
    return lambdify((vars + params), func, modules='numpy', dummify=False)

def sympy_to_scipy(func, vars, params):
    """
    Convert a symbolic expression to one scipy digs.
    :param func: sympy expression
    :param vars: variables
    :param params: parameters
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

def variables(names):
    """
    Convenience function for the creation of multiple variables.
    :param names: string of variable names. Should be comma seperated.
    Example: x, y = variables('x, y')
    """
    return [Variable(name=name.strip()) for name in names.split(',')]

def parameters(names):
    """
    Convenience function for the creation of multiple parameters.
    :param names: string of parameter names. Should be comma seperated.
    Example: a, b = parameters('a, b')
    """
    return [Parameter(name=name.strip()) for name in names.split(',')]

def cache(func):
    """
    Decorator function that gets a method as its input and either buffers the input,
    or returns the buffered output. Used in conjunction with properties to take away
    the standard buffering logic.
    :param func:
    :return:
    """
    @wraps(func)
    def new_func(self):
        try:
            return getattr(self, '_{}'.format(func.__name__))
        except AttributeError:
            setattr(self, '_{}'.format(func.__name__), func(self))
            return getattr(self, '_{}'.format(func.__name__))

    return new_func

def r_squared(residuals, ydata, sigma=None):
    """
    Calculate the squared regression coefficient from the given residuals and data.
    :param residuals: array of residuals, f(x, p) - y.
    :param ydata: y in the above equation.
    :param sigma: sigma in the y_i
    """
    ss_err = np.sum(residuals ** 2)
    if sigma is not None:
        ss_tot = np.sum(((ydata - ydata.mean())/sigma) ** 2)
    else:
        ss_tot = np.sum((ydata - ydata.mean()) ** 2)

    return 1 - (ss_err / ss_tot)