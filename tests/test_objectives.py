# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import pytest
import pickle

import numpy as np

from symfit import (
    Variable, Parameter, parameters, Fit,
    Model, FitResults, variables, Idx,
    symbols, Sum, log, exp, cos, pi, besseli
)
from symfit.core.objectives import (
    VectorLeastSquares, LeastSquares, LogLikelihood, MinimizeModel,
    BaseIndependentObjective
)
from symfit.distributions import Exp

# Overwrite the way Sum is printed by numpy just while testing. Is not
# general enough to be moved to symfit.core.printing, but has to be used
# in this test. This way of summing completely ignores the summation indices and
# the dimensions, and instead just flattens everything to a scalar. Only used
# in this test to build the analytical equivalents of our LeastSquares
# and LogLikelihood


class FlattenSum(Sum):
    """
    Just a sum which is printed differently: by flattening the whole array and
    summing it. Used in tests only.
    """

    def _numpycode(self, printer):
        return "%s(%s)" % (printer._module_format('numpy.sum'), printer.doprint(self.function))


def setup_module():
    np.random.seed(0)


def test_pickle():
    """
    Test the picklability of the built-in objectives.
    """
    # Create test data
    xdata = np.linspace(0, 100, 100)  # From 0 to 100 in 100 steps
    a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
    b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
    ydata = a_vec * xdata + b_vec  # Point scattered around the line 15 * x + 100

    # Normal symbolic fit
    a = Parameter('a', value=0, min=0.0, max=1000)
    b = Parameter('b', value=0, min=0.0, max=1000)
    x, y = variables('x, y')
    model = Model({y: a * x + b})

    for objective in [VectorLeastSquares, LeastSquares, LogLikelihood, MinimizeModel]:
        if issubclass(objective, BaseIndependentObjective):
            data = {x: xdata}
        else:
            data = {x: xdata, y: ydata, model.sigmas[y]: np.ones_like(ydata)}
        obj = objective(model, data=data)
        new_obj = pickle.loads(pickle.dumps(obj))
        assert FitResults._array_safe_dict_eq(obj.__dict__, new_obj.__dict__)


def test_LeastSquares():
    """
    Tests if the LeastSquares objective gives the right shapes of output by
    comparing with its analytical equivalent.
    """
    i = Idx('i', 100)
    x, y = symbols('x, y', cls=Variable)
    X2 = symbols('X2', cls=Variable)
    a, b = parameters('a, b')

    model = Model({y: a * x**2 + b * x})
    xdata = np.linspace(0, 10, 100)
    ydata = model(x=xdata, a=5, b=2).y + np.random.normal(0, 5, xdata.shape)

    # Construct a LeastSquares objective and its analytical equivalent
    chi2_numerical = LeastSquares(model, data={
        x: xdata, y: ydata, model.sigmas[y]: np.ones_like(xdata)
    })
    chi2_exact = Model({X2: FlattenSum(0.5 * ((a * x ** 2 + b * x) - y) ** 2, i)})

    eval_exact = chi2_exact(x=xdata, y=ydata, a=2, b=3)
    jac_exact = chi2_exact.eval_jacobian(x=xdata, y=ydata, a=2, b=3)
    hess_exact = chi2_exact.eval_hessian(x=xdata, y=ydata, a=2, b=3)
    eval_numerical = chi2_numerical(x=xdata, a=2, b=3)
    jac_numerical = chi2_numerical.eval_jacobian(x=xdata, a=2, b=3)
    hess_numerical = chi2_numerical.eval_hessian(x=xdata, a=2, b=3)

    # Test model jacobian and hessian shape
    assert model(x=xdata, a=2, b=3)[0].shape == ydata.shape
    assert model.eval_jacobian(x=xdata, a=2, b=3)[0].shape == (2, 100)
    assert model.eval_hessian(x=xdata, a=2, b=3)[0].shape == (2, 2, 100)
    # Test exact chi2 shape
    assert eval_exact[0].shape, (1,)
    assert jac_exact[0].shape, (2, 1)
    assert hess_exact[0].shape, (2, 2, 1)

    # Test if these two models have the same call, jacobian, and hessian
    assert eval_exact[0] == pytest.approx(eval_numerical)
    assert isinstance(eval_numerical, float)
    assert isinstance(eval_exact[0][0], float)
    assert np.squeeze(jac_exact[0], axis=-1) == pytest.approx(jac_numerical)
    assert isinstance(jac_numerical, np.ndarray)
    assert np.squeeze(hess_exact[0], axis=-1) == pytest.approx(hess_numerical)
    assert isinstance(hess_numerical, np.ndarray)

    fit = Fit(chi2_exact, x=xdata, y=ydata, objective=MinimizeModel)
    fit_exact_result = fit.execute()
    fit = Fit(model, x=xdata, y=ydata, absolute_sigma=True)
    fit_num_result = fit.execute()
    assert fit_exact_result.value(a) == fit_num_result.value(a)
    assert fit_exact_result.value(b) == fit_num_result.value(b)
    assert fit_exact_result.stdev(a) == pytest.approx(fit_num_result.stdev(a))
    assert fit_exact_result.stdev(b) == pytest.approx(fit_num_result.stdev(b))


def test_LogLikelihood():
    """
    Tests if the LeastSquares objective gives the right shapes of output by
    comparing with its analytical equivalent.
    """
    # TODO: update these tests to use indexed variables in the future
    a, b = parameters('a, b')
    i = Idx('i', 100)
    x, y = variables('x, y')
    pdf = Exp(x, 1 / a) * Exp(x, b)

    np.random.seed(10)
    xdata = np.random.exponential(3.5, 100)

    # We use minus loglikelihood for the model, because the objective was
    # designed to find the maximum when used with a *minimizer*, so it has
    # opposite sign. Also test MinimizeModel at the same time.
    logL_model = Model({y: pdf})
    logL_exact = Model({y: - FlattenSum(log(pdf), i)})
    logL_numerical = LogLikelihood(logL_model, {x: xdata, y: None})
    logL_minmodel = MinimizeModel(logL_exact, data={x: xdata, y: None})

    # Test model jacobian and hessian shape
    eval_exact = logL_exact(x=xdata, a=2, b=3)
    jac_exact = logL_exact.eval_jacobian(x=xdata, a=2, b=3)
    hess_exact = logL_exact.eval_hessian(x=xdata, a=2, b=3)
    eval_minimizemodel = logL_minmodel(a=2, b=3)
    jac_minimizemodel = logL_minmodel.eval_jacobian(a=2, b=3)
    hess_minimizemodel = logL_minmodel.eval_hessian(a=2, b=3)
    eval_numerical = logL_numerical(a=2, b=3)
    jac_numerical = logL_numerical.eval_jacobian(a=2, b=3)
    hess_numerical = logL_numerical.eval_hessian(a=2, b=3)

    # TODO: These shapes should not have the ones! This is due to the current
    # convention that scalars should be returned as a 1d array by Model's.
    assert eval_exact[0].shape == (1,)
    assert jac_exact[0].shape == (2, 1)
    assert hess_exact[0].shape == (2, 2, 1)
    # Test if identical to MinimizeModel
    assert eval_exact[0] == pytest.approx(eval_minimizemodel)
    assert jac_exact[0] == pytest.approx(jac_minimizemodel)
    assert hess_exact[0] == pytest.approx(hess_minimizemodel)

    # Test if these two models have the same call, jacobian, and hessian.
    # Since models always have components as their first dimension, we have
    # to slice that away.
    assert eval_exact.y == pytest.approx(eval_numerical)
    assert isinstance(eval_numerical, float)
    assert isinstance(eval_exact.y[0], float)
    assert np.squeeze(jac_exact[0], axis=-1) == pytest.approx(jac_numerical)
    assert isinstance(jac_numerical, np.ndarray)
    assert np.squeeze(hess_exact[0], axis=-1) == pytest.approx(hess_numerical)
    assert isinstance(hess_numerical, np.ndarray)

    fit = Fit(logL_exact, x=xdata, objective=MinimizeModel)
    fit_exact_result = fit.execute()
    fit = Fit(logL_model, x=xdata, objective=LogLikelihood)
    fit_num_result = fit.execute()
    assert fit_exact_result.value(a) == pytest.approx(fit_num_result.value(a))
    assert fit_exact_result.value(b) == pytest.approx(fit_num_result.value(b))
    assert fit_exact_result.stdev(a) == pytest.approx(fit_num_result.stdev(a))
    assert fit_exact_result.stdev(b) == pytest.approx(fit_num_result.stdev(b))


def test_data_sanity():
    """
    Tests very basicly the data sanity for different objective types.
    :return:
    """
    # Create test data
    xdata = np.linspace(0, 100, 25)  # From 0 to 100 in 25 steps
    a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
    b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
    ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

    # Normal symbolic fit
    a = Parameter('a', value=0, min=0.0, max=1000)
    b = Parameter('b', value=0, min=0.0, max=1000)
    x, y, z = variables('x, y, z')
    model = Model({y: a * x + b})

    for objective in [VectorLeastSquares, LeastSquares, LogLikelihood, MinimizeModel]:
        if issubclass(objective, BaseIndependentObjective):
            incomplete_data = {}
            data = {x: xdata}
            overcomplete_data = {x: xdata, z: ydata}
        else:
            incomplete_data = {x: xdata, y: ydata}
            data = {x: xdata, y: ydata, model.sigmas[y]: np.ones_like(ydata)}
            overcomplete_data = {x: xdata, y: ydata, z: ydata, model.sigmas[y]: np.ones_like(ydata)}
        with pytest.raises(KeyError):
            obj = objective(model, data=incomplete_data)

        obj = objective(model, data=data)
        # Overcomplete data has to be allowed, since constraints share their
        # data with models.
        obj = objective(model, data=overcomplete_data)


def test_LogLikelihood_global():
    """
    This is a test for global likelihood fitting to multiple data sets.
    Based on SO question 56006357.
    """
    # creating the data
    mu1, mu2 = .05, -.05
    sigma1, sigma2 = 3.5, 2.5
    n1, n2 = 80, 90
    np.random.seed(42)
    x1 = np.random.vonmises(mu1, sigma1, n1)
    x2 = np.random.vonmises(mu2, sigma2, n2)

    n = 2  # number of components
    xs = variables('x,' + ','.join('x_{}'.format(i) for i in range(1, n + 1)))
    x, xs = xs[0], xs[1:]
    ys = variables(','.join('y_{}'.format(i) for i in range(1, n + 1)))
    mu, kappa = parameters('mu, kappa')
    kappas = parameters(','.join('k_{}'.format(i) for i in range(1, n + 1)), min=0, max=10)
    mu.min, mu.max = - np.pi, np.pi

    template = exp(kappa * cos(x - mu)) / (2 * pi * besseli(0, kappa))

    model = Model(
        {y_i: template.subs({kappa: k_i, x: x_i}) for y_i, x_i, k_i in zip(ys, xs, kappas)}
    )

    all_data = {xs[0]: x1, xs[1]: x2, ys[0]: None, ys[1]: None}
    all_params = {'mu': 1}
    all_params.update({k_i.name: 1 for k_i in kappas})

    # Evaluate the loglikelihood and its jacobian and hessian
    logL = LogLikelihood(model, data=all_data)
    eval_numerical = logL(**all_params)
    jac_numerical = logL.eval_jacobian(**all_params)
    hess_numerical = logL.eval_hessian(**all_params)

    # Test the types and shapes of the components.
    assert isinstance(eval_numerical, float)
    assert isinstance(jac_numerical, np.ndarray)
    assert isinstance(hess_numerical, np.ndarray)

    assert eval_numerical.shape == tuple()  # Empty tuple -> scalar
    assert jac_numerical.shape == (3,)
    assert hess_numerical.shape == (3, 3,)
