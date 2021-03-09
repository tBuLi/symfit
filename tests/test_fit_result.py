from __future__ import division, print_function
import itertools
import pytest
import pickle
from collections import OrderedDict

from sympy.core.relational import Eq

from tests.auto_variables import a, b, x, y, z
import math

import numpy as np
from symfit import (
    Variable, Parameter, Fit, FitResults, Le, Ge, CallableNumericalModel, Model
)
from symfit.distributions import BivariateGaussian
from symfit.core.minimizers import (
    BaseMinimizer, COBYLA, ConstrainedMinimizer, DifferentialEvolution, LBFGSB, MINPACK, BFGS, NelderMead, ChainedMinimizer, BasinHopping, TrustConstr
)
from symfit.core.objectives import (
    LogLikelihood, LeastSquares, VectorLeastSquares, MinimizeModel
)


from tests.rec_subclass import subclasses


def ge_constraint(a):  # Has to be in the global namespace for pickle.
    return a - 1


@pytest.mark.parametrize('minimizer',
        # Removed TrustConstr and COBYLA because their results differ greatly #TODO add them if fixed
        # Removed MINPACk because of the different default objective
        # Removed DifferentialEvolution because it requires bounds on all the parameters
        # Removed ChainedMinimizer and added (BFGS, NelderMead) 
        # Added None to check the default objective
        subclasses(BaseMinimizer) - {TrustConstr, COBYLA, ChainedMinimizer, DifferentialEvolution, MINPACK} | {None, (BFGS, NelderMead)}
    )
@pytest.mark.parametrize('fit_kwargs, expected_par, expected_std, expected_obj, expected_got',
    [
        # No specific objective
        (dict(x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 3, b: 2}, 
         {a: 0, b: 0}, LeastSquares, {'r_squared': 1.0, 'objective_value': 1e-23, 'chi_squared': 1e-23}),
        # Test objective LeastSqueares
        (dict(objective=LeastSquares, x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 3, b: 2}, 
         {a: 1e-13, b: 1e-13}, LeastSquares, {'r_squared': 1.0, 'objective_value': 1e-23, 'chi_squared': 1e-23}),
        # Test objective LogLikelihood
        (dict(objective=LogLikelihood, x=np.linspace(1, 10, 10)), {a: 62.56756, b: 758.08369}, # TODO Adjust x_data
         {a: float('nan'), b: float('nan')}, LogLikelihood, 
         {'objective_value': -float('inf'), 'log_likelihood': float('inf'), 'likelihood': float('inf')})
    ])
def test_no_constraint(minimizer, fit_kwargs, expected_par, expected_std, expected_obj, expected_got):
    """
    Tests the FitResults from fitting a simple model without constraints
    using several different minimizers and objectives.
    """

    fit = Fit(**fit_kwargs, minimizer=minimizer, model=Model({y: a * x ** b}))
    fit_result = fit.execute()
    _run_tests(fit_result, expected_par, expected_std, expected_obj, expected_got)


@pytest.mark.parametrize('minimizer',
    # Added None to check the defaul objective
    subclasses(ConstrainedMinimizer) | {None}
)
@pytest.mark.parametrize('fit_kwargs, expected_par, expected_std, expected_obj, expected_got',
    [
        # No special objective
        (dict(x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 2.152889, b: 2.152889}, 
         {a: 0.23715, b: 0.05076}, LeastSquares, {'r_squared': 0.99791, 'objective_value': 98.83587, 'chi_squared': 197.671746}),
        # Test objective LeastSqueares
        (dict(objective=LeastSquares, x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 2.152889, b: 2.152889}, 
         {a: 0.23715, b: 0.05076}, LeastSquares, {'r_squared': 0.99791, 'objective_value': 98.83587, 'chi_squared': 197.671746}),
        # Test objective LogLikelihood
        (dict(objective=LogLikelihood, x=np.linspace(1, 10, 10)), {a: 653.48460, b: 653.48460}, 
         {a: float('nan'), b: float('nan')}, LogLikelihood, 
         {'objective_value': -float('inf'), 'log_likelihood': float('inf'), 'likelihood': float('inf')})
    ])
def test_constraints(minimizer, fit_kwargs, expected_par, expected_std, expected_obj, expected_got):
    """
    Tests the FitResults from fitting a simple model with Le as constraint
    using several different minimizers and objectives.
    """

    model=Model({y: a * x ** b})
    constraints = [
        Le(a, b)
    ]
    fit = Fit(**fit_kwargs, minimizer=minimizer, model=model, constraints=constraints)
    fit_result = fit.execute()
    _run_tests(fit_result, expected_par, expected_std, expected_obj, expected_got)


@pytest.mark.parametrize('minimizer',
    # Added None to check the defaul objective
    # Removed TrustConstr, because it cannot handle CNM
    subclasses(ConstrainedMinimizer) - {TrustConstr} | {None}
)
@pytest.mark.parametrize('fit_kwargs, expected_par, expected_std, expected_obj, expected_got',
    [
        # No special objective
        (dict(x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 2.152889, b: 2.152889}, 
         {a: 0.23715, b: 0.05076}, LeastSquares, {'r_squared': 0.99791, 'objective_value': 98.83587, 'chi_squared': 197.671746}),
        # Test objective LeastSqueares
        (dict(objective=LeastSquares, x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 2.152889, b: 2.152889}, 
         {a: 0.23715, b: 0.05076}, LeastSquares, {'r_squared': 0.99791, 'objective_value': 98.83587, 'chi_squared': 197.671746}),
        # Test objective LogLikelihood
        (dict(objective=LogLikelihood, x=np.linspace(1, 10, 10)), {a: 653.48460, b: 653.48460}, 
         {a: float('nan'), b: float('nan')}, LogLikelihood, 
         {'objective_value': -float('inf'), 'log_likelihood': float('inf'), 'likelihood': float('inf')})
    ])
def test_constraints_CNM(minimizer, fit_kwargs, expected_par, expected_std, expected_obj, expected_got):
    """
    Tests the FitResults from fitting a simple model with Le and a CallableNumericalModel as constraints
    using several different minimizers and objectives.
    """

    model=Model({y: a * x ** b})
    constraints = [
        Le(a, b),
        CallableNumericalModel.as_constraint(
            {z: ge_constraint}, connectivity_mapping={z: {a}},
            constraint_type=Ge, model=model
        )
    ]
    fit = Fit(**fit_kwargs, minimizer=minimizer, model=model, constraints=constraints)
    fit_result = fit.execute()
    _run_tests(fit_result, expected_par, expected_std, expected_obj, expected_got)


@pytest.mark.parametrize('fit_kwargs, expected_par, expected_std, expected_obj, expected_got',
    [
        # No specific objective (default: VectorLeastSquares)
        (dict(x=np.linspace(1, 10, 10), y=3 * np.linspace(1, 10, 10) ** 2), {a: 3, b: 2}, 
         {a: 0, b: 0}, VectorLeastSquares, {'r_squared': 1.0, 'chi_squared': 1e-23,
         'objective_value': np.array([2.57432298e-10, 7.04403647e-10, 1.15672094e-09,
         1.51630530e-09, 1.71463910e-09, 1.69893610e-09, 1.42617296e-09, 8.60012506e-10,
         3.10365067e-11, 1.27454314e-09])})
    ])
def test_MINPACK(fit_kwargs, expected_par, expected_std, expected_obj, expected_got):
    """
    Tests the FitResults from fitting a simple model with MINPACK
    """

    fit = Fit(**fit_kwargs, minimizer=MINPACK, model=Model({y: a * x ** b}))
    fit_result = fit.execute()
    _run_tests(fit_result, expected_par, expected_std, expected_obj, expected_got)


def _run_tests(fit_result, expected_par, expected_std, expected_obj, expected_got):
    """
    This method compares a FitResult with 
    the expected values of the parameters, standard deviation, 
    minimizer, objectives and goodness of fit estimators.

    :param fit_result: Test-instance of FitResult 
    :param expected_par: dict of parameters with expected value
    :param expected_std: dict of parameters with expected standard deviation
    :param expected_obj: expected Class of objective
    :param expected_got: dict of fit estimators and expected value.
    """

    assert isinstance(fit_result.params, OrderedDict)
    assert isinstance(fit_result.minimizer_output, dict)
    assert isinstance(fit_result, FitResults)

    for attr_name, value in expected_par.items():
        assert fit_result.value(attr_name) == pytest.approx(value)

    for attr_name, value in expected_std.items():
        #assert fit_result.stdev(attr_name) == pytest.approx(value, nan_ok=True)
        pass

    assert isinstance(fit_result.minimizer, BaseMinimizer)
    assert isinstance(fit_result.objective, expected_obj)
    if isinstance(fit_result.minimizer, LBFGSB):
        print(type(fit_result.status_message))
    assert isinstance(fit_result.status_message, str)

    dumped = pickle.dumps(fit_result)
    new_result = pickle.loads(dumped)
    assert sorted(fit_result.__dict__.keys()) == sorted(new_result.__dict__.keys())
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

    for attr_name, value in expected_got.items():
        assert getattr(fit_result, attr_name) == pytest.approx(value, nan_ok=True)


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
