from sympy.core.symbol import Symbol
import inspect
class Argument(Symbol):
    def __new__(cls, name=None, **assumptions):
        # Super dirty way? to determine the variable name from the calling line.
        if not name:
            frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
            caller = lines[0].strip()
            if '==' in caller:
                pass
            else:
                try:
                    lhs, rhs = caller.split('=')
                except ValueError:
                    generated_name = name
                else:
                    generated_name = lhs.strip()
                return super(Argument,cls).__new__(cls, generated_name, **assumptions)
        return super(Argument,cls).__new__(cls, name, **assumptions)

class Parameter(Argument):
    """ Parameter objects are used to facilitate bounds on function parameters,
    as well as to allow AbstractFunction instances to share parameters between
    them.
    """
    def __init__(self, name, value=1.0, min=None, max=None, fixed=False, *args, **kwargs):
        super(Parameter, self).__init__(name, *args, **kwargs)
        self.value = value
        self.fixed = fixed
        if not self.fixed:
            self.min = min
            self.max = max

class Variable(Argument):
    pass