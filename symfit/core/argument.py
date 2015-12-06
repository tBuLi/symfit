from sympy.core.symbol import Symbol
import inspect

class Argument(Symbol):
    """
    Base class for ``symfit`` symbols. This helps make ``symfit`` symbols distinguishable from ``sympy`` symbols.

    The ``Argument`` class also makes DRY possible in defining ``Argument``'s: it uses ``inspect`` to read the lhs of the
    assignment and uses that as the name for the ``Argument`` is none is explicitly set.

    For example::

        x = Variable()
        print(x.name)
        >> 'x'
    """
    def __new__(cls, name=None, **assumptions):
        assumptions['real'] = True
        # Super dirty way? to determine the variable name from the calling line.
        if not name or type(name) != str:
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
            caller = lines[0].strip()
            if '==' in caller:
                pass
            else:
                try:
                    terms = caller.split('=')
                except ValueError:
                    generated_name = name
                else:
                    generated_name = terms[0].strip()  # lhs
                return super(Argument, cls).__new__(cls, generated_name, **assumptions)
        return super(Argument,cls).__new__(cls, name, **assumptions)

    def __init__(self, name=None, *sympy_args, **sympy_kwargs):
        if name is not None:
            self.name = name
        super(Argument, self).__init__(*sympy_args, **sympy_kwargs)


class Parameter(Argument):
    """ Parameter objects are used to facilitate bounds on function parameters. """
    def __init__(self, value=1.0, min=None, max=None, fixed=False, name=None, *sympy_args, **sympy_kwargs):
        """
        :param value: Initial guess value.
        :param min: Lower bound on the parameter value.
        :param max: Upper bound on the parameter value.
        :param fixed: Fix the parameter to ``value`` during fitting.
        :type fixed: bool
        :param name: Name of the Parameter.
        :param sympy_args: Args to pass to ``sympy``.
        :param sympy_kwargs: Kwargs to pass to ``sympy``.
        """
        super(Parameter, self).__init__(name, *sympy_args, **sympy_kwargs)
        self.value = value
        self.fixed = fixed
        if not self.fixed:
            self.min = min
            self.max = max


class Variable(Argument):
    """ Variable type."""
    pass