from sympy.core.symbol import Symbol
from sympy import Eq, Ne
from sympy.core.expr import Expr

class Argument(Symbol):
    pass
    # def __eq__(self, other):
    #     """
    #     We want our object to behave differently from sympy Symbol's
    #     to ensure beatiful and consistant syntax.
    #     :param other:
    #     :return:
    #     """
        # if isinstance(other, float) or isinstance(other, int) or \
        #         isinstance(other, bool) or isinstance(self, float) or \
        #         isinstance(self, int) or isinstance(self, bool):
        # if self.__class__ is other.__class__:
        #     print 'here!'
        #     return super(Argument, self).__eq__(other)
        # else:
        #     print 'no here!'
        #     return Eq(self, other)

    # def __ne__(self, other):
    #     print type(self), type(other)
    #     return Ne(self, other)



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