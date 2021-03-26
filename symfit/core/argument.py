# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
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
    __slots__ = ['_argument_index', '_argument_name']
    # TODO: Make sure this also survives a pickle/unpickle to a fresh(!)
    #       interpreter.
    _argument_indices = defaultdict(int)

    def __new__(cls, name=None, **assumptions):
        """
        Create a new ``Argument``. See :class:`~sympy.core.symbol.Symbol`
        for more information.
        """
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

            name = '{}_{}'.format(cls._argument_name, cls._argument_indices[cls])
            instance = super(Argument, cls).__new__(cls, name, **assumptions)
        else:
            instance = super(Argument, cls).__new__(cls, name, **assumptions)
        instance._argument_index = cls._argument_indices[cls]
        cls._argument_indices[cls] += 1
        return instance

    def __init__(self, name=None, *args, **assumptions):
        # TODO: A more careful look at Symbol.__init__ is needed! However, it
        # seems we don't have to pass anything on to it.
        if name is not None:
            self.name = name
        super(Argument, self).__init__()

    def __getstate__(self):
        state = super(Argument, self).__getstate__()
        state.update({slot: getattr(self, slot) for slot in self.__slots__
                      if hasattr(self, slot)})
        return state

    def _sympystr(self, printer, *args, **kwargs):
        return printer.doprint(self.name)

    _lambdacode = _sympystr
    _numpycode = _sympystr
    _pythoncode = _sympystr

class Parameter(Argument):
    """
    Parameter objects are used to facilitate bounds on function parameters.
    Important change from `symfit>0.4.1`: the name needs to be the first keyword,
    followed by the guess value. If no name is provided, the initial value can
    be passed as a keyword argument, e.g.: `value=0.1`. A generic name will then
    be generated.
    """
    # Parameter index to be assigned to generated nameless parameters
    __slots__ = ['min', 'max', 'fixed', 'value']

    _argument_name = 'par'

    def __new__(cls, name=None, value=1.0, min=None, max=None, fixed=False, **kwargs):
        try:
            return super(Parameter, cls).__new__(cls, name, **kwargs)
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
                raise ValueError('The value of `min` should be less than or'
                                 ' equal to the value of `max`.')
        else:
            self.min = min
            self.max = max

    def __eq__(self, other):
        """
        Parameters are considered equal when their name, assumptions, and
        bounds are considered the same.
        """
        equal = super(Parameter, self).__eq__(other)
        if equal is NotImplemented:
            return equal

        if not equal:
            return False
        else:
            return (self.min == other.min and
                    self.max == other.max and
                    self.fixed == other.fixed and
                    self.value == other.value)

    __hash__ = Argument.__hash__


class Variable(Argument):
    """ Variable type."""
    # Variable index to be assigned to generated nameless variables
    _argument_name = 'var'
    __slots__ = ()