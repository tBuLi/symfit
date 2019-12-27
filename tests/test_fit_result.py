from __future__ import division, print_function
import pytest
import pickle
from collections import OrderedDict

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

@pytest.fixture
def a():
    return Parameter('a')


@pytest.fixture
def b():
    return Parameter('b')


@pytest.fixture
def result_dict(a, b):

    result_dict = dict()

    z = Variable('z')
    x = Variable('x')
    y = Variable('y')
    xdata = np.linspace(1, 10, 10)
    ydata = 3 * xdata ** 2
    model = Model({y: a * x ** b})
    constraints = [
        Eq(a, b),
        CallableNumericalModel.as_constraint(
            {z: ge_constraint}, connectivity_mapping={z: {a}},
            constraint_type=Ge, model=model
        )
    ]
    
    fit = Fit(model, x=xdata, y=ydata)
    result_dict['fit_result'] = fit.execute()

    fit = Fit(model, x=xdata, y=ydata, minimizer=MINPACK)
    result_dict['minpack_result'] = fit.execute()

    fit = Fit(model, x=xdata, objective=LogLikelihood)
    result_dict['likelihood_result'] = fit.execute()

    fit = Fit(model, x=xdata, y=ydata, minimizer=[BFGS, NelderMead])
    result_dict['chained_result'] = fit.execute()

    fit = Fit(model, x=xdata, y=ydata, constraints=constraints)
    result_dict['constrained_result'] = fit.execute()

    fit = Fit(model, x=xdata, y=ydata, constraints=constraints,
              minimizer=BasinHopping)
    result_dict['constrained_basinhopping_result'] = fit.execute()

    return result_dict


def ge_constraint(a):  # Has to be in the global namespace for pickle.
    return a - 1


def test_params_type(result_dict):
    assert isinstance(result_dict['fit_result'].params, OrderedDict)


@pytest.mark.parametrize('result_name', ['fit_result', 'minpack_result',
    'likelihood_result'])
def test_minimizer_output_type(result_dict, result_name):
    assert isinstance(result_dict[result_name].minimizer_output, dict)


def test_fitting(result_dict, a, b):
    """
    Test if the fitting worked in the first place.
    """
    fit_result = result_dict['fit_result']

    assert isinstance(fit_result, FitResults)
    assert fit_result.value(a) == pytest.approx(3.0)
    assert fit_result.value(b) == pytest.approx(2.0)

    assert isinstance(fit_result.stdev(a), float)
    assert isinstance(fit_result.stdev(b), float)

    assert isinstance(fit_result.r_squared, float)
    # by definition since there's no fuzzyness
    assert fit_result.r_squared == 1.0

  
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


@pytest.mark.parametrize('result_name', ['fit_result',
    'likelihood_result','constrained_result', 'constrained_basinhopping_result'])
def test_minimizer_included(result_dict, result_name):
    """"The minimizer used should be included in the results."""
    assert isinstance(result_dict[result_name].minimizer, BaseMinimizer)


def test_minimizer_included_chained(result_dict):
    chained_result = result_dict['chained_result']
    assert isinstance(chained_result.minimizer, ChainedMinimizer)
    for minimizer, cls in zip(chained_result.minimizer.minimizers, [BFGS, NelderMead]):
        assert isinstance(minimizer, cls)


@pytest.mark.parametrize('result_name, objective_name',[
    ('fit_result', LeastSquares), 
    ('likelihood_result', LogLikelihood),
    ('minpack_result', VectorLeastSquares),
    ('constrained_result', LeastSquares),
    ('constrained_basinhopping_result', LeastSquares)
    ])
def test_objective_included(result_dict, result_name, objective_name):
    """"The objective used should be included in the results."""
    assert isinstance(result_dict[result_name].objective, objective_name)


@pytest.mark.parametrize('result_name', ['constrained_result',
    'constrained_basinhopping_result'])
def test_constraints_included(result_dict, result_name):
    """
    Test if the constraints have been properly fed to the results object so
    we can easily print their compliance.
    """
    # For a constrained fit we expect a list of MinimizeModel objectives.
    result_constraints = result_dict[result_name]
    assert isinstance(result_constraints.constraints, list)
    for constraint in result_constraints.constraints:
        assert isinstance(constraint, MinimizeModel)


@pytest.mark.parametrize('result_name', ['fit_result', 'likelihood_result',
    'minpack_result', 'chained_result', 'constrained_result',
    'constrained_basinhopping_result'])
def test_message_included(result_dict, result_name):
    """Status message should be included."""
    assert isinstance(result_dict[result_name].status_message, str)


@pytest.mark.parametrize('result_name', ['fit_result', 'likelihood_result',
    'chained_result', 'constrained_result', 'constrained_basinhopping_result'])
def test_pickle(result_dict, result_name):
    act_result = result_dict[result_name]
    dumped = pickle.dumps(act_result)
    new_result = pickle.loads(dumped)
    assert sorted(act_result.__dict__.keys()) == sorted(new_result.__dict__.keys())
    for k, v1 in act_result.__dict__.items():
        v2 = new_result.__dict__[k]
        if k == 'minimizer':
            assert type(v1) == type(v2)
        elif k != 'minimizer_output':  # Ignore minimizer_output
            if isinstance(v1, np.ndarray):
                assert v1 == pytest.approx(v2, nan_ok=True)


def test_gof_presence(result_dict):
    """
    Test if the expected goodness of fit estimators are present.
    """
    fit_result = result_dict['fit_result']
    assert hasattr(fit_result, 'objective_value')
    assert hasattr(fit_result, 'r_squared')
    assert hasattr(fit_result, 'chi_squared')
    assert not hasattr(fit_result, 'log_likelihood')
    assert not hasattr(fit_result, 'likelihood')

    minpack_result = result_dict['minpack_result']
    assert hasattr(minpack_result, 'objective_value')
    assert hasattr(minpack_result, 'r_squared')
    assert hasattr(minpack_result, 'chi_squared')
    assert not hasattr(minpack_result, 'log_likelihood')
    assert not hasattr(minpack_result, 'likelihood')

    likelihood_result = result_dict['likelihood_result']
    assert hasattr(likelihood_result, 'objective_value')
    assert not hasattr(likelihood_result, 'r_squared')
    assert not hasattr(likelihood_result, 'chi_squared')
    assert hasattr(likelihood_result, 'log_likelihood')
    assert hasattr(likelihood_result, 'likelihood')
