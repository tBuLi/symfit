import os
import sys
import inspect

from sympy.core.symbol import Symbol

try: # This defines the basestring in both py2/py3.
    basestring
except NameError:
    basestring = str

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
        if not name or not isinstance(name, basestring):
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]

            # Check if the script is running in an exe
            # If this is the case, symfit cannot find the vaiable name from the calling line and it will search
            # in the exe directory to find the source. Make sure to copy the required source files for this to work.
            if getattr(sys, 'frozen', False):
                basedir = sys._MEIPASS #  Find the exe base dir
                fname = os.path.basename(filename)
                # Find the source file in the pyinstaller directory
                source_file = None
                for root, dirs, files in os.walk(basedir):
                    if fname in files:
                        source_file = os.path.join(root, fname)
                        break

                if source_file is None:
                    raise IOError("Source code file not found")

                # Get the correct line from the source code
                l = 0
                with open(source_file, 'r') as fid:
                    for line in fid:
                        l+=1
                        if l == line_number:
                            lines = line
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

    def __init__(self, name=None, **assumptions):
        # TODO: A more careful look at Symbol.__init__ is needed! However, it
        # seems we don't have to pass anything on to it.
        if name is not None:
            self.name = name
        super(Argument, self).__init__()


class Parameter(Argument):
    """ Parameter objects are used to facilitate bounds on function parameters. """
    def __init__(self, value=1.0, min=None, max=None, fixed=False, name=None, **assumptions):
        """
        :param value: Initial guess value.
        :param min: Lower bound on the parameter value.
        :param max: Upper bound on the parameter value.
        :param fixed: Fix the parameter to ``value`` during fitting.
        :type fixed: bool
        :param name: Name of the Parameter.
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
    pass