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
    # Solve the bilinear example with MIP and compare to Fit.
    x, y, z = parameters('x, y, z', min=0)

    objective = 1.0 * x
    constraints = [
        x + y + z <= 10,
        x * y <= 2,
        Eq(x * z + y * z, 1),
    ]

    mip = MIP(- objective, constraints=constraints)
    mip_result = mip.execute()
    fit = Fit(- objective, constraints=constraints)
    fit_result = fit.execute()

    assert mip_result[x] == pytest.approx(fit_result.value(x), abs=1e-6)
    assert mip_result[y] == pytest.approx(fit_result.value(y), abs=1e-6)
    assert mip_result[z] == pytest.approx(fit_result.value(z), abs=1e-6)
    assert mip_result.objective_value == pytest.approx(fit_result.objective_value, abs=1e-6)