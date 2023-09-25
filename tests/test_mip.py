import pytest
import numpy as np

from symfit import Parameter


def test_parameter_array_bounds():
    lb = np.zeros(10)
    ub = np.ones(10)
    with pytest.raises(ValueError):
        x = Parameter('x', min=ub, max=lb)
    x = Parameter('x', min=lb, max=ub)
