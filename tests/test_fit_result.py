from __future__ import division, print_function
import pytest
import pickle
from collections import OrderedDict

from tests.auto_variables import *

import numpy as np
from symfit import (
    Variable, Parameter, Fit, FitResults, Eq, Ge, CallableNumericalModel, Model
)
from symfit.distributions import BivariateGaussian
from symfit.core.minimizers import (
    BaseMinimizer, MINPACK, BFGS, NelderMead, ChainedMinimizer, BasinHopping
)
from symfit.core.objectives import (
    LogLikelihood, LeastSquares, VectorLeastSquares, MinimizeModel
)


# TODO Entfernen
# 1. Likelihood & Constraints Ergebnisse (a, b, (r²)) verschieden >>> mit wolfram alpha testen?
# 2. kein r² bei likelihood >>> check mit hasttr
# 3. Chained Minimizer & LeastSquare >>> einfach einsetzen, obwohl es nicht der normalfall ist?
# 4. r² zu hasttr Sektion hinzufügen? >>> why not
# 5. Zukunft: test_fitting_2 simpler/zu großem Test?    

def ge_constraint(a):  # Has to be in the global namespace for pickle.
    return a - 1


constraints = [
    Eq(a, b),
    CallableNumericalModel.as_constraint(
        {z: ge_constraint}, connectivity_mapping={z: {a}},
        constraint_type=Ge, model=Model({y: a * x ** b})
    )
]


@pytest.mark.parametrize('kwargs, objective_name, gof_presence', [
    ({'x': np.linspace(1, 10, 10), 'y': 3 * np.linspace(1, 10, 10) ** 2},
     LeastSquares, (True, True, True, False, False)),
    ({'x': np.linspace(1, 10, 10), 'y': 3 * np.linspace(1, 10, 10) ** 2,
      'minimizer': MINPACK}, VectorLeastSquares, (True, True, True, False, False)),
    ({'x': np.linspace(1, 10, 10), 'objective': LogLikelihood},
     LogLikelihood, (True, False, False, True, True)),
    ({'x': np.linspace(1, 10, 10), 'y': 3 * np.linspace(1, 10, 10) ** 2, 'minimizer': [BFGS, NelderMead]},
     LeastSquares, (True, True, True, False, False)),
    ({'x': np.linspace(1, 10, 10), 'y': 3 * np.linspace(1, 10, 10) ** 2,
      'constraints': constraints}, LeastSquares, (True, True, True, False, False)),
    ({'x': np.linspace(1, 10, 10), 'y': 3 * np.linspace(1, 10, 10) ** 2, 'constraints': constraints, 'minimizer': BasinHopping},
     LeastSquares, (True, True, True, False, False))
])
def test_model(kwargs, objective_name, gof_presence):
    model = Model({y: a * x ** b})
    fit = Fit(model, **kwargs)
    fit_result = fit.execute()

    assert isinstance(fit_result.params, OrderedDict)

    assert isinstance(fit_result.minimizer_output, dict)
    assert isinstance(fit_result, FitResults)

    assert fit_result.value(a) == pytest.approx(3.0)
    assert fit_result.value(b) == pytest.approx(2.0)

    assert isinstance(fit_result.stdev(a), float)
    assert isinstance(fit_result.stdev(b), float)

    if hasattr(fit_result, 'r_squared'):
        assert isinstance(fit_result.r_squared, float)
        # assert fit_result.r_squared == 1.0
        # by definition since there's no fuzzyness    

    assert isinstance(fit_result.minimizer, BaseMinimizer)

    assert isinstance(fit_result.objective, objective_name)

    assert isinstance(fit_result.status_message, str)

    dumped = pickle.dumps(fit_result)
    new_result = pickle.loads(dumped)
    assert sorted(fit_result.__dict__.keys()) == sorted(
        new_result.__dict__.keys())
    for k, v1 in fit_result.__dict__.items():
        v2 = new_result.__dict__[k]
        if k == 'minimizer':
            assert type(v1) == type(v2)
        elif k != 'minimizer_output':  # Ignore minimizer_output
            if isinstance(v1, np.ndarray):
                assert v1 == pytest.approx(v2, nan_ok=True)

    if fit_result.constraints:
        assert isinstance(fit_result.constraints, list)
        for constraint in fit_result.constraints:
            assert isinstance(constraint, MinimizeModel)

    assert hasattr(fit_result, 'objective_value') == gof_presence[0]
    assert hasattr(fit_result, 'r_squared') == gof_presence[1]
    assert hasattr(fit_result, 'chi_squared') == gof_presence[2]
    assert hasattr(fit_result, 'log_likelihood') == gof_presence[3]
    assert hasattr(fit_result, 'likelihood') == gof_presence[4]


def test_fitting_2():
    np.random.seed(43)
    mean = (0.62, 0.71)  # x, y mean 0.7, 0.7
    cov = [
        [0.102**2, 0],
        [0, 0.07**2]
    ]
    data_1 = np.random.multivariate_normal(mean, cov, 10**5)
    mean = (0.33, 0.28)  # x, y mean 0.3, 0.3
    cov = [  # rho = 0.25
        [0.05 ** 2, 0.25 * 0.05 * 0.101],
        [0.25 * 0.05 * 0.101, 0.101 ** 2]
    ]
    data_2 = np.random.multivariate_normal(mean, cov, 10**5)
    data = np.vstack((data_1, data_2))

    # Insert them as y,x here as np fucks up cartesian conventions.
    ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=200,
                                           range=[[0.0, 1.0], [0.0, 1.0]],
                                           density=True)
    xcentres = (xedges[:-1] + xedges[1:]) / 2
    ycentres = (yedges[:-1] + yedges[1:]) / 2

    # Make a valid grid to match ydata
    xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)

    x = Variable('x')
    y = Variable('y')

    x0_1 = Parameter('x0_1', value=0.6, min=0.5, max=0.7)
    sig_x_1 = Parameter('sig_x_1', value=0.1, min=0.0, max=0.2)
    y0_1 = Parameter('y0_1', value=0.7, min=0.6, max=0.8)
    sig_y_1 = Parameter('sig_y_1', value=0.05, min=0.0, max=0.2)
    rho_1 = Parameter('rho_1', value=0.0, min=-0.5, max=0.5)
    A_1 = Parameter('A_1', value=0.5, min=0.3, max=0.7)
    g_1 = A_1 * BivariateGaussian(x=x, y=y, mu_x=x0_1, mu_y=y0_1,
                                  sig_x=sig_x_1, sig_y=sig_y_1, rho=rho_1)

    x0_2 = Parameter('x0_2', value=0.3, min=0.2, max=0.4)
    sig_x_2 = Parameter('sig_x_2', value=0.05, min=0.0, max=0.2)
    y0_2 = Parameter('y0_2', value=0.3, min=0.2, max=0.4)
    sig_y_2 = Parameter('sig_y_2', value=0.1, min=0.0, max=0.2)
    rho_2 = Parameter('rho_2', value=0.26, min=0.0, max=0.8)
    A_2 = Parameter('A_2', value=0.5, min=0.3, max=0.7)
    g_2 = A_2 * BivariateGaussian(x=x, y=y, mu_x=x0_2, mu_y=y0_2,
                                  sig_x=sig_x_2, sig_y=sig_y_2, rho=rho_2)

    model = g_1 + g_2
    fit = Fit(model, xx, yy, ydata)
    fit_result = fit.execute()

    assert fit_result.r_squared > 0.95
    for param in fit.model.params:
        try:
            assert fit_result.stdev(param)**2 == pytest.approx(fit_result.variance(param))
        except AssertionError:
            assert fit_result.variance(param) <= 0.0
            assert np.isnan(fit_result.stdev(param))

    # Covariance matrix should be symmetric
    for param_1 in fit.model.params:
        for param_2 in fit.model.params:
            assert fit_result.covariance(param_1, param_2) == pytest.approx(fit_result.covariance(param_2, param_1), rel=1e-3)


def test_minimizer_included_chained():
    xdata = np.linspace(1, 10, 10)
    ydata = 3 * xdata ** 2
    model = Model({y: a * x ** b})
    fit = Fit(model, x=xdata, y=ydata, minimizer=[BFGS, NelderMead])
    chained_result = fit.execute()
    assert isinstance(chained_result.minimizer, ChainedMinimizer)
    for minimizer, cls in zip(chained_result.minimizer.minimizers, [BFGS, NelderMead]):
        assert isinstance(minimizer, cls)


