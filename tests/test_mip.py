import pytest
import numpy as np

from symfit import Parameter, parameters, Eq, MIP, Fit


def test_parameter_array_bounds():
    lb = np.zeros(10)
    ub = np.ones(10)
    with pytest.raises(ValueError):
        x = Parameter('x', min=ub, max=lb)
    x = Parameter('x', min=lb, max=ub)


def test_bilinear():
    # Create variables
    x, y, z = parameters('x, y, z', min=0)

    objective = 1.0 * x
    constraints = [
        x + y + z <= 10,
        x * y <= 2,
        Eq(x * z + y * z, 1),
    ]

    mip = MIP(- objective, constraints=constraints)
    mip_result = mip.execute()
    print(mip_result)

    fit = Fit(- objective, constraints=constraints)
    fit_result = fit.execute()
    print(fit_result)