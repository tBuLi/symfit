from sympy.printing.pycode import NumPyPrinter
from sympy.printing.codeprinter import CodePrinter

#########################################################
# Monkeypatch the printer until this is merged upstream #
#########################################################
from sympy.core import sympify
from sympy.core.basic import Basic
from sympy.core.compatibility import string_types
from sympy.core.symbol import Symbol

# Backwards compatibility
from sympy.codegen.ast import Assignment

def doprint(self, expr, assign_to=None):
    """
    Print the expression as code.

    Parameters
    ----------
    expr : Expression
        The expression to be printed.

    assign_to : Symbol, MatrixSymbol, or string (optional)
        If provided, the printed code will set the expression to a
        variable with name ``assign_to``.
    """
    from sympy.matrices.expressions.matexpr import MatrixSymbol

    if isinstance(assign_to, string_types):
        if expr.is_Matrix:
            assign_to = MatrixSymbol(assign_to, *expr.shape)
        else:
            assign_to = Symbol(assign_to)
    elif not isinstance(assign_to, (Basic, type(None))):
        raise TypeError("{0} cannot assign to object of type {1}".format(
            type(self).__name__, type(assign_to)))

    if assign_to:
        expr = Assignment(assign_to, expr)
    else:
        # _sympify is not enough b/c it errors on iterables
        expr = sympify(expr)

    # keep a set of expressions that are not strictly translatable to Code
    # and number constants that must be declared and initialized
    self._not_supported = set()
    self._number_symbols = set()

    lines = self._print(expr).splitlines()

    # format the output
    if self._settings["human"]:
        frontlines = []
        if len(self._not_supported) > 0:
            frontlines.append(self._get_comment(
                "Not supported in {0}:".format(self.language)))
            for expr in sorted(self._not_supported, key=str):
                frontlines.append(self._get_comment(type(expr).__name__))
        for name, value in sorted(self._number_symbols, key=str):
            frontlines.append(self._declare_number_const(name, value))
        lines = frontlines + lines
        lines = self._format_code(lines)
        result = "\n".join(lines)
    else:
        lines = self._format_code(lines)
        num_syms = set([(k, self._print(v)) for k, v in self._number_symbols])
        result = (num_syms, self._not_supported, "\n".join(lines))
    self._not_supported = set()
    self._number_symbols = set()
    return result

CodePrinter.doprint = doprint

#########################################################
# End of monkeypatch                                    #
#########################################################

class SymfitNumPyPrinter(NumPyPrinter):
    def _print_OnesLike(self, expr):
        return "%s(%s)" % (self._module_format('numpy.ones_like'),
                           self._print(expr.args[0]))

    def _print_ZerosLike(self, expr):
        return "%s(%s)" % (self._module_format('numpy.zeros_like'),
                           self._print(expr.args[0]))

    def _print_VarOnesLikeVar(self, expr):
        return self._print(expr.args[0])

    def _print_Variable(self, expr):
        return self._print(expr.name)

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