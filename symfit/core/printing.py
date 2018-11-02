from sympy.printing.pycode import NumPyPrinter

class SymfitNumPyPrinter(NumPyPrinter):
    def _print_OnesLike(self, expr):
        return "%s(%s)" % (self._module_format('numpy.ones_like'),
                           self._print(expr.args[0]))

    def _print_ZerosLike(self, expr):
        return "%s(%s)" % (self._module_format('numpy.zeros_like'),
                           self._print(expr.args[0]))