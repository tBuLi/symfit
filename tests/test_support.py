# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

"""
This module contains tests for functions in the :mod:`symfit.core.support` module.
"""

from __future__ import division, print_function
import pytest
import sys
from itertools import repeat

from symfit.core.support import (
    keywordonly, RequiredKeyword, RequiredKeywordError, partial, parameters,
    cached_property
)

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


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


def test_keywordonly_signature():
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
    sig_f = inspect_sig.signature(f)
    for param in sig_f.parameters.values():
        assert param.kind == kinds[param.name]


def test_keywordonly_call():
    """
    Call our test function with some values to see if it behaves as
    expected.
    """
    assert f(4, 3, c=5, d=6) == 4 + 3 + 5 + 6
    # In the next case the 5 is left behind since it ends up in *args.
    assert f(4, 3, 5, d=6) == 4 + 3 + 2 + 6


def test_keywordonly_norequiredkeyword():
    """
    Try to not provide a RequiredKeyword with a value and get away with it.
    (we shouldn't get away with it if all is well.)
    """
    with pytest.raises(RequiredKeywordError):
        f(4, 3, 5, 6)


def test_keywordonly_nokwagrs():
    """
    Decorating a function with no **kwargs-like argument should not be
    allowed.
    """
    with pytest.raises(RequiredKeywordError):
        @keywordonly(c=2, d=RequiredKeyword)
        def g(a, b, *args):
            pass


def test_keywordonly_class():
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
    sig = inspect_sig.signature(A.__init__)
    for param in sig.parameters.values():
        assert param.kind == kinds[param.name]


def test_keywordonly_inheritance():
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
    sig_B = inspect_sig.signature(B.__init__)
    for param in sig_B.parameters.values():
        assert param.kind == kinds_B[param.name]
    assert len(sig_B.parameters) == len(kinds_B)

    sig_A = inspect_sig.signature(A.__init__)
    for param in sig_A.parameters.values():
        assert param.kind == kinds_A[param.name]
    assert len(sig_A.parameters) == len(kinds_A)

    with pytest.raises(TypeError):
        b = B(3, 5, 7, d=2, e=6)


def test_repeatable_partial():
    """
    Test the custom repeatable partial, which makes partial behave the same
    in older python versions as in the most recent.
    """
    def partial_me(a, b, c=None):
        return a, b, c

    partialed_one = partial(partial_me, a=2)
    partialed_two = partial(partialed_one, b='string')

    assert isinstance(partialed_one, partial)
    assert partialed_one.func == partial_me
    assert not partialed_one.args
    assert partialed_one.keywords == {'a': 2}

    # For the second partial, all should remain the same except the keywords
    # are extended by one item.
    assert isinstance(partialed_two, partial)
    assert partialed_two.func == partial_me
    assert not partialed_two.args
    assert partialed_two.keywords == {'a': 2, 'b': 'string'}


def test_parameters():
    """
    Test the `parameter` convenience function.
    """
    x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=0.0)
    assert x1.value == 2.0
    assert x2.value == 1.3
    assert x1.min == 0.0
    assert x2.min == 0.0
    assert not x1.fixed
    assert not x2.fixed
    with pytest.raises(ValueError):
        x1, x2 = parameters('x1, x2', value=[2.0, 1.3, 3.0], min=0.0)

    x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=[-30, -10], max=[300, 100], fixed=[True, False])

    assert x1.min == -30
    assert x2.min == -10
    assert x1.max == 300
    assert x2.max == 100
    assert x1.value == 2.0
    assert x2.value == 1.3
    assert x1.fixed
    assert not x2.fixed

    # Illegal bounds
    with pytest.raises(ValueError):
        x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=[400, -10], max=[300, 100])
    # Should not raise any error, as repeat is an endless source of values
    x1, x2 = parameters('x1, x2', value=[2.0, 1.3], min=repeat(0.0))


def test_cached_property():
    class A(object):
        def __init__(self):
            self.counter = 0

        @cached_property
        def f(self):
            self.counter += 1
            return 2

    a = A()
    # Delete a.f before a cache was set will fail silently.
    del a.f
    with pytest.raises(AttributeError):
        # Cache does not exist before f is called
        a._f
    assert a.f == 2
    assert hasattr(a, '{}_f'.format(cached_property.base_str))
    del a.f
    # check that deletion was successful
    with pytest.raises(AttributeError):
        # Does not exist before f is called
        a._f
    # However, the function should still be there
    assert a.f == 2
    with pytest.raises(AttributeError):
        # Setting is not allowed.
        a.f = 3

    # Counter should read 2 at this point, the number of calls since
    # object creation.
    assert a.counter == 2
    for _ in range(10):
        a.f
    # Should be returning from cache, so a.f is not actually called
    assert a.counter == 2
