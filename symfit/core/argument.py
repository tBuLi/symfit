import numbers
import warnings

from sympy.core.symbol import Symbol

class Argument(Symbol):
    """
    Base class for :mod:`symfit` symbols. This helps make :mod:`symfit` symbols
    distinguishable from :mod:`sympy` symbols.

    If no name is explicitly provided a name will be generated.

    For example::

        y = Variable()
        print(y.name)
        >> 'x_0'

        y = Variable('y')
        print(y.name)
        >> 'y'
    """
    def __new__(cls, name=None, *args, **assumptions):
        assumptions['real'] = True
        # Generate a dummy name
        if not name:
            # Throw a warning that is is better to explicitly give names.
            warnings.warn(
                'It is recommended to provide names to {} explicitly'
                ' as automatic generation of names will be dropped in '
                'future `symfit` versions.'.format(cls.__name__),
                DeprecationWarning, stacklevel=2
            )

            name = '{}_{}'.format(cls._argument_name, cls._argument_index)
            instance = super(Argument, cls).__new__(cls, name, **assumptions)
            instance._argument_index = cls._argument_index
            cls._argument_index += 1
            return instance
        else:
            return super(Argument, cls).__new__(cls, name, **assumptions)

    def __init__(self, name=None, *args, **assumptions):
        # TODO: A more careful look at Symbol.__init__ is needed! However, it
        # seems we don't have to pass anything on to it.
        if name is not None:
            self.name = name
        super(Argument, self).__init__()

    def __getstate__(self):
        state = super(Argument, self).__getstate__()
        state.update(dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        ))
        return state


class Parameter(Argument):
    """
    Parameter objects are used to facilitate bounds on function parameters.
    Important change from `symfit>0.4.1`: the name needs to be the first keyword,
    followed by the guess value. If no name is provided, the initial value can
    be passed as a keyword argument, e.g.: `value=0.1`. A generic name will then
    be generated.
    """
    # Parameter index to be assigned to generated nameless parameters
    _argument_index = 0
    _argument_name = 'par'
    __slots__ = ['min', 'max', 'fixed']

    def __new__(cls, name=None, *args, **kwargs):
        try:
            return super(Parameter, cls).__new__(cls, name, *args, **kwargs)
        except TypeError as err:
            if isinstance(name, numbers.Number):
                raise TypeError('In symfit >0.4.1 the value needs to be assigned '
                                'as the second argument or by keyword argument.')
            else: raise err

    def __init__(self, name=None, value=1.0, min=None, max=None, fixed=False, **assumptions):
        """
        :param name: Name of the Parameter.
        :param value: Initial guess value.
        :param min: Lower bound on the parameter value.
        :param max: Upper bound on the parameter value.
        :param fixed: Fix the parameter to ``value`` during fitting.
        :type fixed: bool
        :param assumptions: assumptions to pass to ``sympy``.
        """
        super(Parameter, self).__init__(name, **assumptions)
        self.value = value
        self.fixed = fixed

        if min is not None and max is not None and min > max:
            if not self.fixed:
                print(min, max)
                raise ValueError('The value of `min` should be less than or'
                                 ' equal to the value of `max`.')
        else:
            self.min = min
            self.max = max


class Variable(Argument):
    """ Variable type."""
    # Variable index to be assigned to generated nameless variables
    _argument_index = 0
    _argument_name = 'var'