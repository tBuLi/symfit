"""
This module contains tests for functions in the :mod:`symfit.core.support` module.
"""

from __future__ import division, print_function
import unittest
import sys
import warnings

from symfit.core.support import keywordonly, RequiredKeyword, RequiredKeywordError

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


class TestSupport(unittest.TestCase):
    def setUp(self):
        @keywordonly(c=2, d=RequiredKeyword)
        def f(a, b, *args, **kwargs):
            c = kwargs.pop('c')
            d = kwargs.pop('d')
            return a + b + c + d

        self._f = f

    def test_keywordonly_signature(self):
        """
        Test the keywordonly decorators ability to update the signature of the
        function it wraps.
        """
        kinds = {
            'a': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'b': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'args': inspect_sig.Parameter.VAR_POSITIONAL,
            'kwargs': inspect_sig.Parameter.VAR_KEYWORD,
            'c': inspect_sig.Parameter.KEYWORD_ONLY,
            'd': inspect_sig.Parameter.KEYWORD_ONLY,
        }
        sig_f = inspect_sig.signature(self._f)
        for param in sig_f.parameters.values():
            self.assertTrue(param.kind == kinds[param.name])

    def test_keywordonly_call(self):
        """
        Call our test function with some values to see if it behaves as
        expected.
        """
        self.assertEqual(self._f(4, 3, c=5, d=6), 4 + 3 + 5 + 6)
        # In the next case the 5 is left behind since it ends up in *args.
        self.assertEqual(self._f(4, 3, 5, d=6), 4 + 3 + 2 + 6)

    def test_keywordonly_norequiredkeyword(self):
        """
        Try to not provide a RequiredKeyword with a value and get away with it.
        (we shouldn't get away with it if all is well.)
        """
        with self.assertRaises(RequiredKeywordError):
            self._f(4, 3, 5, 6)

    def test_keywordonly_nokwagrs(self):
        """
        Decorating a function with no **kwargs-like argument should not be
        allowed.
        """
        with self.assertRaises(RequiredKeywordError):
            @keywordonly(c=2, d=RequiredKeyword)
            def g(a, b, *args):
                pass

if __name__ == '__main__':
    try:
        unittest.main(warnings='ignore')
        # Note that unittest will catch and handle exceptions raised by tests.
        # So this line will *only* deal with exceptions raised by the line
        # above.
    except TypeError:
        # In Py2, unittest.main doesn't take a warnings argument
        warnings.simplefilter('ignore')
        unittest.main()
