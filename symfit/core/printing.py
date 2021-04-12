# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

"""
``symfit`` occasionally updates the printing of ``sympy`` objects, such that they
print into their ``numpy``/``scipy`` equivalent. This is done because sometimes
such printing has not been implemented in ``sympy`` yet, or because we want
slightly different behavior from the standard one.

Users using both ``symfit`` and ``sympy`` should be aware of this.
"""
import pkg_resources
from sympy import HadamardProduct, MatPow, Idx, Inverse
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

def _print_Inverse(self, printer):
    return "%s(%s)" % (printer._module_format('numpy.linalg.inv'),
                       printer.doprint(self.args[0]))
Inverse._numpycode = _print_Inverse

def _print_HadamardProduct(self, printer):
    return "%s*%s" % (printer.doprint(self.args[0]),
                      printer.doprint(self.args[1]))
HadamardProduct._numpycode = _print_HadamardProduct

if pkg_resources.parse_version(pkg_resources.get_distribution('sympy').version) \
        > pkg_resources.parse_version('1.4'):
    from sympy import HadamardPower

    def _print_HadamardPower(self, printer):
        return "%s**%s" % (printer.doprint(self.args[0]),
                          printer.doprint(self.args[1]))
    HadamardPower._numpycode = _print_HadamardPower

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