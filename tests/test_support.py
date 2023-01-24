# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

"""
This module contains tests for functions in the :mod:`symfit.core.support` module.
"""

from __future__ import division, print_function
import pytest
from itertools import repeat

from symfit.core.support import (
    parameters, cached_property
)


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
