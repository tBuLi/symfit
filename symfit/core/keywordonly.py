# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

"""
This module contains support functions and convenience methods used
throughout symfit. Some are used predominantly internally, others are
designed for users.
"""
from __future__ import print_function
from collections import OrderedDict
import sys

if sys.version_info >= (3,0):
    import inspect as inspect_sig
    from functools import wraps
else:
    import funcsigs as inspect_sig
    from functools32 import wraps


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