import re
import numpy as np
from symfit.core.argument import Parameter, Variable
from sympy.utilities.lambdify import lambdify

def seperate_symbols(func):
    params = []
    vars = []
    for symbol in func.free_symbols:
        if isinstance(symbol, Parameter):
            params.append(symbol)
        elif isinstance(symbol, Variable):
            vars.append(symbol)
        else:
            raise TypeError('model contains an unknown symbol type, {}'.format(type(symbol)))
    return vars, params

def sympy_to_py(func, vars, params):
    """
    Turn a symbolic expression to a Python (Lambda) function.
    :param func: sympy expression
    :param vars: varianbles
    :param params: parameters
    :return: lambda function.
    """
    return lambdify((vars + params), func, modules='numpy')

def sympy_to_scipy(func, vars, params):
    """
    Convert a symbolic expression to one scipy digs.
    :param func: sympy expression
    :param vars: varianbles
    :param params: parameters
    """
    lambda_func = sympy_to_py(func, vars, params)
    def f(x, p):
        """ Scipy style function.
        :param x: list of arrays, NxM
        :param p: tuple of parameter values.
        """
        x = np.atleast_2d(x)
        y = [x[i] for i in range(len(x))]
        return lambda_func(*(y + list(p)))

    return f



# def sympy_to_scipy(func, vars, params):
#     """
#     Turn the wonderfully elegant sympy expression into an ugly scipy
#     function.
#     """
#     func_str = str(func)
#     # if len(vars) == 1:
#     #     func_str = func_str.replace(str(vars[0]), 'x')
#     # else:
#     #     for key, var in enumerate(vars):
#     #         func_str = func_str.replace(str(var), 'x[{}]'.format(key))
#     # The following for does something along the lines of:
#     # Find all ocurrences of a var, and replace it by x[0] etc.
#     # Found by applying the hit or mis-method.
#     new_func_str = ''
#     for key, var in enumerate(vars):
#         # func_str = func_str.replace(str(var), 'x[{}]'.format(key))
#         # Regex explained by using http://regexpal.com
#         variable_positions = re.finditer('[^a-zA-Z0-9]{0}[^a-zA-Z0-9]|(^{0}[^a-zA-Z0-9])|([^a-zA-Z0-9]{0}$)'.format(str(var)), func_str)
#         prev_stop = 0
#         for variable_position in variable_positions:
#             start, stop = variable_position.span()
#             new_var = func_str[start:stop].replace(str(var), 'x[{}]'.format(key))
#             new_func_str += func_str[prev_stop:start] + new_var
#             prev_stop = stop
#         new_func_str += func_str[prev_stop:]
#         # raise Exception(new_func_str)
#         func_str = new_func_str
#         new_func_str = ''
#
#     try:
#         param_str = str(params[0])
#     except IndexError:
#         param_str = ''
#     for param in params[1:]:
#         param_str += ', {}'.format(param)
#
#     # Replace mathematical functions by their numpy equivalent.
#     expressions = {
#         'log': 'np.log',
#         'exp': 'np.exp',
#         'sin': 'np.sin',
#         'cos': 'np.cos',
#         'pi': 'np.pi',
#     }
#     for key, value in expressions.iteritems():
#         func_str = func_str.replace(key, value)
#
#     import imp
#
#     code = """
# def f(x, {0}):
#     return {1}
# """.format(param_str, func_str)
#     # print code
#     # Compile to fully working python function
#     module = imp.new_module('scipy_function')
#     exec code in globals(), module.__dict__ # globals() is needed to have numpy defined
#     return module.f