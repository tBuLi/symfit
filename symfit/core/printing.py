from sympy import HadamardProduct, MatMul, MatPow, Idx, Inverse
from sympy.printing.codeprinter import CodePrinter

##########################################################
# Monkeypatch the printers until this is merged upstream #
##########################################################

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

def _print_MatMul(self, printer):
    """
    Matrix multiplication printer. The sympy one turns everything into a
    dot product without type-checking.
    """
    from sympy import MatrixExpr
    links = []
    for i, j in zip(self.args[1:], self.args[:-1]):
        if isinstance(i, MatrixExpr) and isinstance(j, MatrixExpr):
            links.append(').dot(')
        else:
            links.append('*')
    printouts = [printer.doprint(i) for i in self.args]
    result = [printouts[0]]
    for link, printout in zip(links, printouts[1:]):
        result.extend([link, printout])
    return '({0})'.format(''.join(result))
MatMul._numpycode = _print_MatMul

def _print_Inverse(self, printer):
    return "%s(%s)" % (printer._module_format('numpy.linalg.inv'),
                       printer.doprint(self.args[0]))
Inverse._numpycode = _print_Inverse

def _print_HadamardProduct(self, printer):
    return "%s*%s" % (printer.doprint(self.args[0]),
                      printer.doprint(self.args[1]))
HadamardProduct._numpycode = _print_HadamardProduct

def _print_Idx(self, printer):
    """
    Print ``Idx`` objects.
    """
    return "{0}".format(printer.doprint(self.args[0]))
Idx._numpycode = _print_Idx

def _print_MatPow(self, printer):
    if self.shape == (1, 1):
        # Scalar, so we can take a normal power.
        return printer._print_Pow(self)
    else:
        return printer._print_MatPow(self)
MatPow._numpycode = _print_MatPow

#########################################################
# End of monkeypatch                                    #
#########################################################