"""
This module contains tests for functions in the :mod:`symfit.core.support` module.
"""

from __future__ import division, print_function
import unittest
import sys
import warnings
from itertools import repeat

from symfit.core.support import (
    keywordonly, RequiredKeyword, RequiredKeywordError, partial, parameters,
    cached_property
)

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

        class A(object):
            @keywordonly(c=2, d=RequiredKeyword)
            def __init__(self, a, b, **kwargs):
                pass

        class B(A):
            @keywordonly(e=5)
            def __init__(self, *args, **kwargs):
                e = kwargs.pop('e')
                super(B, self).__init__(*args, **kwargs)

        self._f = f
        self._A = A
        self._B = B

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

    def test_keywordonly_class(self):
        """
        Decorating a function with no **kwargs-like argument should not be
        allowed.
        """
        kinds = {
            'self': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'a': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'b': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'args': inspect_sig.Parameter.VAR_POSITIONAL,
            'kwargs': inspect_sig.Parameter.VAR_KEYWORD,
            'c': inspect_sig.Parameter.KEYWORD_ONLY,
            'd': inspect_sig.Parameter.KEYWORD_ONLY,
        }
        sig = inspect_sig.signature(self._A.__init__)
        for param in sig.parameters.values():
            self.assertTrue(param.kind == kinds[param.name])

    def test_keywordonly_inheritance(self):
        """
        Tests if the decorator deals with inheritance properly.
        """
        kinds_B = {
            'self': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'args': inspect_sig.Parameter.VAR_POSITIONAL,
            'kwargs': inspect_sig.Parameter.VAR_KEYWORD,
            'e': inspect_sig.Parameter.KEYWORD_ONLY,
        }
        kinds_A = {
            'self': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'a': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'b': inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
            'kwargs': inspect_sig.Parameter.VAR_KEYWORD,
            'c': inspect_sig.Parameter.KEYWORD_ONLY,
            'd': inspect_sig.Parameter.KEYWORD_ONLY,
        }
        sig_B = inspect_sig.signature(self._B.__init__)
        for param in sig_B.parameters.values():
            self.assertTrue(param.kind == kinds_B[param.name])
        self.assertEqual(len(sig_B.parameters), len(kinds_B))

        sig_A = inspect_sig.signature(self._A.__init__)
        for param in sig_A.parameters.values():
            self.assertTrue(param.kind == kinds_A[param.name])
        self.assertEqual(len(sig_A.parameters), len(kinds_A))

        with self.assertRaises(TypeError):
            b = self._B(3, 5, 7, d=2, e=6)

    def test_repeatable_partial(self):
        """
        Test the custom repeatable partial, which makes partial behave the same
        in older python versions as in the most recent.
        """
        def partial_me(a, b, c=None):
            return a, b, c

        partialed_one = partial(partial_me, a=2)
        partialed_two = partial(partialed_one, b='string')

        self.assertIsInstance(partialed_one, partial)
        self.assertEqual(partialed_one.func, partial_me)
        self.assertFalse(partialed_one.args)
        self.assertEqual(partialed_one.keywords, {'a': 2})

        # For the second partial, all should remain the same except the keywords
        # are extended by one item.
        self.assertIsInstance(partialed_two, partial)
        self.assertEqual(partialed_two.func, partial_me)
        self.assertFalse(partialed_two.args)
        self.assertEqual(partialed_two.keywords, {'a': 2, 'b': 'string'})

    def test_parameters(self):
        """
        Test the `parameter` convenience function.
        """
        x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=0.0)
        self.assertEqual(x1.value, 2.0)
        self.assertEqual(x2.value, 1.3)
        self.assertEqual(x1.min, 0.0)
        self.assertEqual(x2.min, 0.0)
        self.assertEqual(x1.fixed, False)
        self.assertEqual(x2.fixed, False)
        with self.assertRaises(ValueError):
            x1, x2 = parameters('x1, x2', value=[2.0, 1.3, 3.0], min=0.0)

        x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=[-30, -10], max=[300, 100], fixed=[True, False])
        self.assertEqual(x1.min, -30)
        self.assertEqual(x2.min, -10)
        self.assertEqual(x1.max, 300)
        self.assertEqual(x2.max, 100)
        self.assertEqual(x1.value, 2.0)
        self.assertEqual(x2.value, 1.3)
        self.assertEqual(x1.fixed, True)
        self.assertEqual(x2.fixed, False)

        # Illegal bounds
        with self.assertRaises(ValueError):
            x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=[400, -10], max=[300, 100])
        # Should not raise any error, as repeat is an endless source of values
        x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=repeat(0.0))

    def test_cached_property(self):
        class A(object):
            def __init__(self):
                self.counter = 0

            @cached_property
            def f(self):
                self.counter += 1
                return 2

        a = A()
        # Deleta before a cache was set will fail silently.
        del a.f
        with self.assertRaises(AttributeError):
            # Cache does not exist before f is called
            a._f
        self.assertEqual(a.f, 2)
        self.assertTrue(hasattr(a, '{}_f'.format(cached_property.base_str)))
        del a.f
        # check that deletion was successful
        with self.assertRaises(AttributeError):
            # Does not exist before f is called
            a._f
        # However, the function should still be there
        self.assertEqual(a.f, 2)
        with self.assertRaises(AttributeError):
            # Setting is not allowed.
            a.f = 3

        # Counter should read 2 at this point, the number of calls since
        # object creation.
        self.assertEqual(a.counter, 2)
        for _ in range(10):
            a.f
        # Should be returning from cache, so a.f is not actually called
        self.assertEqual(a.counter, 2)

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
