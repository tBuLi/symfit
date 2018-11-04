from sympy.printing.pycode import NumPyPrinter
from sympy.printing.codeprinter import CodePrinter

#########################################################
# Monkeypatch the printer until this is merged upstream #
#########################################################

class DontDeleteMe(object):
    def __init__(self, default_value):
        self.dont_delete = default_value
        self.default_value = default_value

    def __get__(self, instance, owner):
        return self.dont_delete

    def __set__(self, instance, value):
        self.dont_delete = value

    def __delete__(self, instance):
        self.dont_delete = self.default_value

CodePrinter._not_supported = DontDeleteMe(set())
CodePrinter._number_symbols = DontDeleteMe(set())

#########################################################
# End of monkeypatch                                    #
#########################################################

class SymfitNumPyPrinter(NumPyPrinter):
    """
    Our own NumpyPrinter subclass, in case we need to print certain numpy
    features which are not yet supported in SymPy.
    """
    def _print_DiracDelta(self, expr):
        """
        Replaces a DiracDelta(x) by np.inf if x == 0, and 0 otherwise. This is
        wrong, but the only thing we can do by the time we are printing. To
        prevent mistakes, integrate before printing.
        """
        return "{0}({1}, [{1} == 0 , {1} != 0], [{2}, 0])".format(
                                        self._module_format('numpy.piecewise'),
                                        self._print(expr.args[0]),
                                        self._module_format('numpy.inf'))