from sympy.core.symbol import Symbol

class Argument(Symbol):
    pass

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