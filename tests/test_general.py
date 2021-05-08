# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import sys

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import pytest

from symfit import (
    Variable, Parameter, Fit, FitResults, log, variables,
    parameters, Model, exp, integrate, oo, GradientModel
)
from symfit.core.minimizers import (
    MINPACK, LBFGSB, BoundedMinimizer, DifferentialEvolution, BaseMinimizer,
    ChainedMinimizer
)
from symfit.core.objectives import LogLikelihood, MinimizeModel, LeastSquares
from symfit.distributions import Gaussian, Exp, BivariateGaussian
from tests.test_minimizers import subclasses

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


def setup_module():
    np.random.seed(0)


def test_callable():
    """
    Make sure that symfit expressions are callable (with scalars and
    arrays), and produce the expected results.
    """
    a, b = parameters('a, b')
    x, y = variables('x, y')
    func = a*x**2 + b*y**2
    result = func(x=2, y=3, a=3, b=9)
    assert result == 3*2**2 + 9*3**2
    result = func(2, 3, a=3, b=9)
    assert result == 3*2**2 + 9*3**2

    xdata = np.arange(1, 10)
    ydata = np.arange(1, 10)
    result = func(x=ydata, y=ydata, a=3, b=9)
    assert np.array_equal(result, 3*xdata**2 + 9*ydata**2)


def test_named_fitting():
    """
    Make sure that fitting with NumericalLeastSquares works using a dict
    as model and that the resulting fit_result is of the right type.
    """
    xdata = np.linspace(1, 10, 10)
    ydata = 3*xdata**2

    a = Parameter('a', value=1.0)
    b = Parameter('b', value=2.5)
    x, y = variables('x, y')

    model = {y: a*x**b}

    fit = Fit(model, x=xdata, y=ydata, minimizer=MINPACK)
    fit_result = fit.execute()
    assert isinstance(fit_result, FitResults)
    assert fit_result.value(a) == pytest.approx(3.0)
    assert fit_result.value(b) == pytest.approx(2.0)


def test_backwards_compatible_fitting():
    """
    In 0.4.2 we replaced the usage of inspect by automatically generated
    names. This can cause problems for users using named variables to call
    fit.
    """
    xdata = np.linspace(1, 10, 10)
    ydata = 3*xdata**2

    a = Parameter('a', value=1.0)
    b = Parameter('b', value=2.5)

    y = Variable('y')

    with pytest.warns(DeprecationWarning):
        x = Variable()

    model = {y: a*x**b}

    with pytest.raises(TypeError):
        fit = Fit(model, x=xdata, y=ydata)


def test_vector_fitting():
    """
    Tests fitting to a 3 component vector valued function, without bounds
    or guesses.
    """
    a, b, c = parameters('a, b, c')
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = {a_i: a, b_i: b, c_i: c}

    xdata = np.array([
        [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
        [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
        [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
    ])

    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        minimizer=MINPACK
    )
    fit_result = fit.execute()

    assert fit_result.value(a) == pytest.approx(np.mean(xdata[0]), 1e-5)
    assert fit_result.value(b) == pytest.approx(np.mean(xdata[1]), 1e-4)
    assert fit_result.value(c) == pytest.approx(np.mean(xdata[2]), 1e-5)


def test_vector_none_fitting():
    """
    Fit to a 3 component vector valued function with one variables data set
    to None, without bounds or guesses.
    """
    a, b, c = parameters('a, b, c')
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = {a_i: a, b_i: b, c_i: c}

    xdata = np.array([
        [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
        [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
        [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
    ])

    fit_none = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=None,
        minimizer=MINPACK
    )
    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        minimizer=MINPACK
    )
    fit_none_result = fit_none.execute()
    fit_result = fit.execute()

    assert fit_none_result.value(b) == pytest.approx(fit_result.value(b), 1e-4)
    assert fit_none_result.value(a) == pytest.approx(fit_result.value(a), 1e-4)
    # the parameter without data should be unchanged.
    assert fit_none_result.value(c) == pytest.approx(1.0)


def test_vector_fitting_guess():
    """
    Tests fitting to a 3 component vector valued function, with guesses.
    """
    a, b, c = parameters('a, b, c')
    a.value = 10
    b.value = 100
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = {a_i: a, b_i: b, c_i: c}

    xdata = np.array([
        [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
        [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
        [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
    ])

    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        minimizer=MINPACK
    )
    fit_result = fit.execute()

    assert fit_result.value(a) == pytest.approx(np.mean(xdata[0]), 1e-4)
    assert fit_result.value(b) == pytest.approx(np.mean(xdata[1]), 1e-4)
    assert fit_result.value(c) == pytest.approx(np.mean(xdata[2]), 1e-4)


def test_fitting():
    """
    Tests fitting with NumericalLeastSquares. Makes sure that the resulting
    objects and values are of the right type, and that the fit_result does
    not have unexpected members.
    """
    xdata = np.linspace(1, 10, 10)
    ydata = 3*xdata**2

    a = Parameter('a')  # 3.1, min=2.5, max=3.5
    b = Parameter('b')
    x = Variable('x')
    new = a*x**b

    fit = Fit(new, xdata, ydata, minimizer=MINPACK)

    fit_result = fit.execute()
    assert isinstance(fit_result, FitResults)
    assert fit_result.value(a) == pytest.approx(3.0)
    assert fit_result.value(b) == pytest.approx(2.0)

    assert isinstance(fit_result.stdev(a), float)
    assert isinstance(fit_result.stdev(b), float)

    assert isinstance(fit_result.r_squared, float)
    assert fit_result.r_squared == 1.0  # by definition since there's no fuzzyness


def test_grid_fitting():
    """
    Tests fitting a scalar function with 2 independent variables.
    """
    xdata = np.arange(-5, 5, 1)
    ydata = np.arange(5, 15, 1)
    xx, yy = np.meshgrid(xdata, ydata, sparse=False)

    zdata = (2.5*xx**2 + 3.0*yy**2)

    a = Parameter('a', value=2.5, max=2.75)
    b = Parameter('b', value=3.0, min=2.75)
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    new = {z: a*x**2 + b*y**2}

    fit = Fit(new, x=xx, y=yy, z=zdata)
    results = fit.execute()

    assert isinstance(fit.minimizer, LBFGSB)

    assert results.value(a) == pytest.approx(2.5)
    assert results.value(b) == pytest.approx(3.)


def test_model_callable():
    """
    Tests if Model objects are callable in the way expected. Calling a
    model should evaluate it's expression(s) with the given values. The
    return value is a namedtuple.

    The signature should also work so inspection is saved.
    """
    a, b = parameters('a, b')
    x, y = variables('x, y')
    new = a*x**2 + b*y**2
    model = Model(new)
    ans = model(3, 3, 2, 2)
    assert isinstance(ans, tuple)
    z, = ans

    assert z == 36
    for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
        assert arg_name == name


@pytest.mark.skip(reason=("Test fails a proportion of the time because `z_1`, "
                          "`z_2` and `z_3` are not necessarily ordered as "
                          "expected so the assert statements fail. `z_1` is "
                          "frequenty equal to either 36 or 72."))
def test_model_callable_from_model_init():
    """
    Tests if Model objects are callable in the way expected. Calling a
    model should evaluate it's expression(s) with the given values. The
    return value is a namedtuple.

    The signature should also work so inspection is saved.

    Tests from Model __init__ directly
    """
    a, b = parameters('a, b')
    x, y = variables('x, y')
    model = Model([
        a*x**2,
        4*b*y**2,
        a*x**2 + b*y**2
    ])
    z_1, z_2, z_3 = model(3, 3, 2, 2)

    assert z_1 == 18
    assert z_2 == 72
    assert z_3 == 36
    for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
        assert arg_name == name


def test_model_callable_from_dict():
    """
    Tests if Model objects are callable in the way expected. Calling a
    model should evaluate it's expression(s) with the given values. The
    return value is a namedtuple.

    The signature should also work so inspection is saved.

    Tests from dict.
    """
    a, b = parameters('a, b')
    x, y = variables('x, y')
    z_1, z_2, z_3 = variables('z_1, z_2, z_3')
    model = Model({
        z_1: a*x**2,
        z_2: 4*b*y**2,
        z_3: a*x**2 + b*y**2
    })
    z_1, z_2, z_3 = model(3, 3, 2, 2)

    assert z_1 == 18
    assert z_2 == 72
    assert z_3 == 36
    for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
        assert arg_name == name


def test_2D_fitting():
    """
    Makes sure that a scalar model with 2 independent variables has the
    proper signature, and that the fit result is of the correct type.
    """
    xdata = np.random.randint(-10, 11, size=(2, 400))
    zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

    a = Parameter('a')
    b = Parameter('b')
    x = Variable('x')
    y = Variable('y')
    new = a*x**2 + b*y**2

    fit = Fit(new, xdata[0], xdata[1], zdata)

    result = fit.model(xdata[0], xdata[1], 2, 3)
    assert isinstance(result, tuple)

    for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(fit.model).parameters):
        assert arg_name == name

    fit_result = fit.execute()
    assert isinstance(fit_result, FitResults)


def test_gaussian_fitting():
    """
    Tests fitting to a gaussian function and fit_result.params unpacking.
    """
    xdata = 2*np.random.rand(10000) - 1  # random betwen [-1, 1]
    ydata = 5.0 * scipy.stats.norm.pdf(xdata, loc=0.0, scale=1.0)

    x0 = Parameter('x0')
    sig = Parameter('sig')
    A = Parameter('A')
    x = Variable('x')
    g = GradientModel(A * Gaussian(x, x0, sig))

    fit = Fit(g, xdata, ydata)
    assert isinstance(fit.objective, LeastSquares)
    fit_result = fit.execute()

    assert fit_result.value(A) == pytest.approx(5.0)
    assert np.abs(fit_result.value(sig)) == pytest.approx(1.0)
    assert fit_result.value(x0) == pytest.approx(0.0)
    # raise Exception([i for i in fit_result.params])
    sexy = g(x=2.0, **fit_result.params)
    ugly = g(
        x=2.0,
        x0=fit_result.value(x0),
        A=fit_result.value(A),
        sig=fit_result.value(sig),
    )
    assert sexy == ugly


def test_2_gaussian_2d_fitting():
    """
    Tests fitting to a scalar gaussian with 2 independent variables with
    tight bounds.
    """
    mean = (0.3, 0.4)  # x, y mean 0.6, 0.4
    cov = [[0.01**2, 0], [0, 0.01**2]]
    # TODO: evaluate gaussian at 100x100 points and add appropriate noise
    data = np.random.multivariate_normal(mean, cov, 3000000)
    mean = (0.7, 0.8)  # x, y mean 0.6, 0.4
    cov = [[0.01**2, 0], [0, 0.01**2]]
    data_2 = np.random.multivariate_normal(mean, cov, 3000000)
    data = np.vstack((data, data_2))

    # Insert them as y,x here as np fucks up cartesian conventions.
    ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=100,
                                           range=[[0.0, 1.0], [0.0, 1.0]])
    xcentres = (xedges[:-1] + xedges[1:]) / 2
    ycentres = (yedges[:-1] + yedges[1:]) / 2

    # Make a valid grid to match ydata
    xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
    # xdata = np.dstack((xx, yy)).T

    x = Variable('x')
    y = Variable('y')

    x0_1 = Parameter('x0_1', value=0.7, min=0.6, max=0.9)
    sig_x_1 = Parameter('sig_x_1', value=0.1, min=0.0, max=0.2)
    y0_1 = Parameter('y0_1', value=0.8, min=0.6, max=0.9)
    sig_y_1 = Parameter('sig_y_1', value=0.1, min=0.0, max=0.2)
    A_1 = Parameter('A_1')
    g_1 = A_1 * Gaussian(x, x0_1, sig_x_1) * Gaussian(y, y0_1, sig_y_1)

    x0_2 = Parameter('x0_2', value=0.3, min=0.2, max=0.5)
    sig_x_2 = Parameter('sig_x_2', value=0.1, min=0.0, max=0.2)
    y0_2 = Parameter('y0_2', value=0.4, min=0.2, max=0.5)
    sig_y_2 = Parameter('sig_y_2', value=0.1, min=0.0, max=0.2)
    A_2 = Parameter('A_2')
    g_2 = A_2 * Gaussian(x, x0_2, sig_x_2) * Gaussian(y, y0_2, sig_y_2)

    model = GradientModel(g_1 + g_2)
    fit = Fit(model, xx, yy, ydata)
    fit_result = fit.execute()

    assert isinstance(fit.minimizer, LBFGSB)

    img = model(x=xx, y=yy, **fit_result.params)[0]
    img_g_1 = g_1(x=xx, y=yy, **fit_result.params)
    img_g_2 = g_2(x=xx, y=yy, **fit_result.params)
    assert img == pytest.approx(img_g_1 + img_g_2)

    # Equal up to some precision. Not much obviously.
    assert fit_result.value(x0_1) == pytest.approx(0.7, 1e-3)
    assert fit_result.value(y0_1) == pytest.approx(0.8, 1e-3)
    assert fit_result.value(x0_2) == pytest.approx(0.3, 1e-3)
    assert fit_result.value(y0_2) == pytest.approx(0.4, 1e-3)


def test_gaussian_2d_fitting():
    """
    Tests fitting to a scalar gaussian function with 2 independent
    variables.
    """
    mean = (0.6, 0.4)  # x, y mean 0.6, 0.4
    cov = [[0.2**2, 0], [0, 0.1**2]]

    # TODO: evaluate gaussian at 100x100 points and add appropriate noise
    data = np.random.multivariate_normal(mean, cov, 1000000)

    # Insert them as y,x here as np fucks up cartesian conventions.
    ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100,
                                           range=[[0.0, 1.0], [0.0, 1.0]])
    xcentres = (xedges[:-1] + xedges[1:]) / 2
    ycentres = (yedges[:-1] + yedges[1:]) / 2

    # Make a valid grid to match ydata
    xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

    x0 = Parameter('x0', value=mean[0])
    sig_x = Parameter('sig_x', min=0.0)
    x = Variable('x')
    y0 = Parameter('y0', value=mean[1])
    sig_y = Parameter('sig_y', min=0.0)
    A = Parameter('A', min=1, value=100)
    y = Variable('y')
    g = Variable('g')
    # g = A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)
    model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
    fit = Fit(model, x=xx, y=yy, g=ydata, minimizer=MINPACK)
    fit_result = fit.execute()

    assert fit_result.value(x0) == pytest.approx(np.mean(data[:, 0]), 1e-1)
    assert fit_result.value(y0) == pytest.approx(np.mean(data[:, 1]), 1e-1)
    assert np.abs(fit_result.value(sig_x)) == pytest.approx(np.std(data[:, 0]), 1e-1)
    assert np.abs(fit_result.value(sig_y)) == pytest.approx(np.std(data[:, 1]), 1e-1)
    assert fit_result.r_squared >= 0.99


def test_jacobian_matrix():
    """
    The jacobian matrix of a model should be a 2D list (matrix) containing
    all the partial derivatives.
    """
    a, b, c = parameters('a, b, c')
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = Model({a_i: 2 * a + 3 * b, b_i: 5 * b, c_i: 7 * c})
    assert [[2, 3, 0], [0, 5, 0], [0, 0, 7]] == model.jacobian


def test_hessian_matrix():
    """
    The Hessian matrix of a model should be a 3D list (matrix) containing
    all the 2nd partial derivatives.
    """
    a, b, c = parameters('a, b, c')
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = Model({a_i: 2 * a**2 + 3 * b, b_i: 5 * b**2, c_i: 7 * c*b})
    assert [[[4, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 10, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 7], [0, 7, 0]]] == model.hessian


def test_likelihood_fitting_exponential():
    """
    Fit using the likelihood method.
    """
    b = Parameter('b', value=4, min=3.0)
    x, y = variables('x, y')
    pdf = {y: Exp(x, 1/b)}

    # Draw points from an Exp(5) exponential distribution.
    np.random.seed(100)
    # TODO: Do we *really* need 1m points?
    xdata = np.random.exponential(5, 1000000)

    # Expected parameter values
    mean = np.mean(xdata)
    stdev = np.std(xdata)
    mean_stdev = stdev / np.sqrt(len(xdata))

    with pytest.raises(TypeError):
        fit = Fit(pdf, x=xdata, sigma_y=2.0, objective=LogLikelihood)

    fit = Fit(pdf, xdata, objective=LogLikelihood)
    fit_result = fit.execute()
    pdf_i = fit.model(x=xdata, **fit_result.params).y  # probabilities
    likelihood = np.product(pdf_i)
    loglikelihood = np.sum(np.log(pdf_i))

    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(mean, 1e-3)
    assert fit_result.value(b) == pytest.approx(stdev, 1e-3)
    assert fit_result.stdev(b) == pytest.approx(mean_stdev, 1e-3)

    assert likelihood == pytest.approx(fit_result.likelihood)
    assert loglikelihood == pytest.approx(fit_result.log_likelihood)


def test_likelihood_fitting_gaussian():
    """
    Fit using the likelihood method.
    """
    mu, sig = parameters('mu, sig')
    sig.min = 0.01
    sig.value = 3.0
    mu.value = 50.
    x = Variable('x')
    pdf = GradientModel(Gaussian(x, mu, sig))

    np.random.seed(10)
    # TODO: Do we really need 1k points?
    xdata = np.random.normal(51., 3.5, 10000)

    # Expected parameter values
    mean = np.mean(xdata)
    stdev = np.std(xdata)
    mean_stdev = stdev/np.sqrt(len(xdata))

    fit = Fit(pdf, xdata, objective=LogLikelihood)
    fit_result = fit.execute()

    assert fit_result.value(mu) == pytest.approx(mean, 1e-6)
    assert fit_result.stdev(mu) == pytest.approx(mean_stdev, 1e-3)
    assert fit_result.value(sig) == pytest.approx(np.std(xdata), 1e-6)


def test_likelihood_fitting_bivariate_gaussian():
    """
    Fit using the likelihood method.
    """
    # Make variables and parameters
    x = Variable('x')
    y = Variable('y')
    x0 = Parameter('x0', value=0.6, min=0.5, max=0.7)
    sig_x = Parameter('sig_x', value=0.1, max=1.0)
    y0 = Parameter('y0', value=0.7, min=0.6, max=0.9)
    sig_y = Parameter('sig_y', value=0.05, max=1.0)
    rho = Parameter('rho', value=0.001, min=-1, max=1)

    pdf = BivariateGaussian(x=x, mu_x=x0, sig_x=sig_x, y=y, mu_y=y0,
                            sig_y=sig_y, rho=rho)

    # Draw 100000 samples from a bivariate distribution
    mean = [0.59, 0.8]
    r = 0.6
    cov = np.array([[0.11 ** 2, 0.11 * 0.23 * r],
                    [0.11 * 0.23 * r, 0.23 ** 2]])
    np.random.seed(42)
    # TODO: Do we really need 100k points?
    xdata, ydata = np.random.multivariate_normal(mean, cov, 100000).T

    fit = Fit(pdf, x=xdata, y=ydata, objective=LogLikelihood)
    fit_result = fit.execute()

    assert fit_result.value(x0) == pytest.approx(mean[0], 1e-2)
    assert fit_result.value(y0) == pytest.approx(mean[1], 1e-2)
    assert fit_result.value(sig_x) == pytest.approx(np.sqrt(cov[0, 0]), 1e-2)
    assert fit_result.value(sig_y) == pytest.approx(np.sqrt(cov[1, 1]), 1e-2)
    assert fit_result.value(rho) == pytest.approx(r, 1e-2)

    marginal = integrate(pdf, (y, -oo, oo), conds='none')
    fit = Fit(marginal, x=xdata, objective=LogLikelihood)

    with pytest.raises(NameError):    
        # Should raise a NameError, not a TypeError, see #219
        fit.execute()


def test_evaluate_model():
    """
    Makes sure that models are callable and give the expected answer.
    """
    A = Parameter('A')
    x = Variable('x')
    new = A * x ** 2

    assert new(x=2, A=2) == 8
    assert not new(x=2, A=3) == 8


def test_simple_sigma():
    """
    Make sure we produce the same results as scipy's curve_fit, with and
    without sigmas, and compare the results of both to a known value.
    """
    t_data = np.array([1.4, 2.1, 2.6, 3.0, 3.3])
    y_data = np.array([10, 20, 30, 40, 50])

    sigma = 0.2
    n = np.array([5, 3, 8, 15, 30])
    sigma_t = sigma / np.sqrt(n)

    # We now define our model
    y = Variable('x')
    g = Parameter('g')
    t_model = (2 * y / g)**0.5

    fit = Fit(t_model, y_data, t_data)  # , sigma=sigma_t)
    fit_result = fit.execute()

    # h_smooth = np.linspace(0,60,100)
    # t_smooth = t_model(y=h_smooth, **fit_result.params)

    # Lets with the results from curve_fit, no weights
    popt_noweights, pcov_noweights = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data)

    assert fit_result.value(g) == pytest.approx(popt_noweights[0])
    assert fit_result.stdev(g) == pytest.approx(np.sqrt(pcov_noweights[0, 0]), 1e-6)

    # Same sigma everywere
    fit = Fit(t_model, y_data, t_data, 0.0031, absolute_sigma=False)
    fit_result = fit.execute()
    popt_sameweights, pcov_sameweights = curve_fit(
        lambda y, p: (2 * y / p)**0.5,
        y_data,
        t_data,
        sigma=0.0031*np.ones(len(y_data)),
        absolute_sigma=False
    )
    assert fit_result.value(g) == pytest.approx(popt_sameweights[0], 1e-4)
    assert fit_result.stdev(g) == pytest.approx(np.sqrt(pcov_sameweights[0, 0]), 1e-4)
    # Same weight everywere should be the same as no weight when absolute_sigma=False
    assert fit_result.value(g) == pytest.approx(popt_noweights[0], 1e-4)
    assert fit_result.stdev(g) == pytest.approx(np.sqrt(pcov_noweights[0, 0]), 1e-4)

    # Different sigma for every point
    fit = Fit(t_model, y_data, t_data, 0.1*sigma_t, absolute_sigma=False)
    fit_result = fit.execute()
    popt, pcov = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data, sigma=.1*sigma_t)

    assert fit_result.value(g) == pytest.approx(popt[0])
    assert fit_result.stdev(g) == pytest.approx(np.sqrt(pcov[0, 0]), 1e-6)

    # according to Mathematica
    assert fit_result.value(g) == pytest.approx(9.095, 1e-3)
    assert fit_result.stdev(g) == pytest.approx(0.102, 1e-2)


def test_error_advanced():
    """
    Models an example from the mathematica docs and try's to replicate it
    using both symfit and scipy's curve_fit.
    http://reference.wolfram.com/language/howto/FitModelsWithMeasurementErrors.html
    """
    data = [
        [0.9, 6.1, 9.5], [3.9, 6., 9.7], [0.3, 2.8, 6.6],
        [1., 2.2, 5.9], [1.8, 2.4, 7.2], [9., 1.7, 7.],
        [7.9, 8., 10.4], [4.9, 3.9, 9.], [2.3, 2.6, 7.4],
        [4.7, 8.4, 10.]
    ]
    xdata, ydata, zdata = [np.array(data) for data in zip(*data)]
    xy = np.vstack((xdata, ydata))
    errors = np.array([.4, .4, .2, .4, .1, .3, .1, .2, .2, .2])

    # raise Exception(xy, z)
    a = Parameter('a', value=3.0)
    b = Parameter('b', value=0.9)
    c = Parameter('c', value=5)
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    model = {z: a * log(b * x + c * y)}

    # Use a gradient model because Mathematica uses the Hessian
    # approximation instead of the exact Hessian.
    model = GradientModel(model)
    fit = Fit(model, x=xdata, y=ydata, z=zdata, absolute_sigma=False)
    fit_result = fit.execute()

    # Same as Mathematica default behavior.
    assert fit_result.value(a) == pytest.approx(2.9956, 1e-4)
    assert fit_result.value(b) == pytest.approx(0.563212, 1e-4)
    assert fit_result.value(c) == pytest.approx(3.59732, 1e-4)
    assert fit_result.stdev(a) == pytest.approx(0.278304, 1e-4)
    assert fit_result.stdev(b) == pytest.approx(0.224107, 1e-4)
    assert fit_result.stdev(c) == pytest.approx(0.980352, 1e-4)

    fit = Fit(model, xdata, ydata, zdata, absolute_sigma=True)
    fit_result = fit.execute()
    # Same as Mathematica in Measurement error mode, but without suplying
    # any errors.
    assert fit_result.value(a) == pytest.approx(2.9956, 1e-4)
    assert fit_result.value(b) == pytest.approx(0.563212, 1e-4)
    assert fit_result.value(c) == pytest.approx(3.59732, 1e-4)
    assert fit_result.stdev(a) == pytest.approx(0.643259, 1e-4)
    assert fit_result.stdev(b) == pytest.approx(0.517992, 1e-4)
    assert fit_result.stdev(c) == pytest.approx(2.26594, 1e-4)

    fit = Fit(model, xdata, ydata, zdata, sigma_z=errors)
    fit_result = fit.execute()

    popt, pcov, infodict, errmsg, ier = curve_fit(
        lambda x_vec, a, b, c: a * np.log(b * x_vec[0] + c * x_vec[1]),
        xy, zdata, sigma=errors, absolute_sigma=True, full_output=True
    )

    # Same as curve_fit?
    assert fit_result.value(a) == pytest.approx(popt[0], 1e-4)
    assert fit_result.value(b) == pytest.approx(popt[1], 1e-4)
    assert fit_result.value(c) == pytest.approx(popt[2], 1e-4)
    assert fit_result.stdev(a) == pytest.approx(np.sqrt(pcov[0, 0]), 1e-4)
    assert fit_result.stdev(b) == pytest.approx(np.sqrt(pcov[1, 1]), 1e-4)
    assert fit_result.stdev(c) == pytest.approx(np.sqrt(pcov[2, 2]), 1e-4)

    # Same as Mathematica with MEASUREMENT ERROR
    assert fit_result.value(a) == pytest.approx(2.68807, 1e-4)
    assert fit_result.value(b) == pytest.approx(0.941344, 1e-4)
    assert fit_result.value(c) == pytest.approx(5.01541, 1e-4)
    assert fit_result.stdev(a) == pytest.approx(0.0974628, 1e-4)
    assert fit_result.stdev(b) == pytest.approx(0.247018, 1e-4)
    assert fit_result.stdev(c) == pytest.approx(0.597661, 1e-4)


def test_error_analytical():
    """
    Test using a case where the analytical answer is known. Uses both
    symfit and scipy's curve_fit.
    Modeled after:
    http://nbviewer.ipython.org/urls/gist.github.com/taldcroft/5014170/raw/31e29e235407e4913dc0ec403af7ed524372b612/curve_fit.ipynb
    """
    N = 10000
    sigma = 10.0 * np.ones(N)
    xn = np.arange(N, dtype=np.float)
    # yn = np.zeros_like(xn)
    np.random.seed(10)
    yn = np.random.normal(size=len(xn), scale=sigma)

    a = Parameter('a')
    y = Variable('y')
    model = {y: a}

    fit = Fit(model, y=yn, sigma_y=sigma)
    fit_result = fit.execute()

    popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x),
                           xn, yn, sigma=sigma, absolute_sigma=True)
    assert fit_result.value(a) == pytest.approx(popt[0], 1e-5)
    assert fit_result.stdev(a) == pytest.approx(np.sqrt(np.diag(pcov))[0], 1e-2)

    fit_no_sigma = Fit(model, yn)
    fit_result_no_sigma = fit_no_sigma.execute()

    popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn,)
    # With or without sigma, the bestfit params should be in agreement in case of equal weights
    assert fit_result.value(a) == pytest.approx(fit_result_no_sigma.value(a), 1e-5)
    # Since symfit is all about absolute errors, the sigma will not be in agreement
    assert not fit_result.stdev(a) == fit_result_no_sigma.stdev(a) == 5
    assert fit_result_no_sigma.stdev(a) == pytest.approx(pcov[0][0]**0.5, 1e-5)
    assert fit_result_no_sigma.value(a) == pytest.approx(popt[0], 1e-5)

    # Analytical answer for mean of N(0,1):
    mu = 0.0
    sigma_mu = sigma[0]/N**0.5

    assert fit_result.stdev(a) == pytest.approx(sigma_mu, 1e-5)

# TODO: redudant with test_error_analytical?
@pytest.mark.skip()
def test_straight_line_analytical():
     """
     Test symfit against a straight line, for which the parameters and their
     uncertainties are known analytically. Assuming equal weights.
     """
     data = [[0, 1], [1, 0], [3, 2], [5, 4]]
     x, y = (np.array(i, dtype='float64') for i in zip(*data))
     # x = np.arange(0, 100, 0.1)
     # np.random.seed(10)
     # y = 3.0*x + 105.0 + np.random.normal(size=x.shape)

     dx = x - x.mean()
     dy = y - y.mean()
     mean_squared_x = np.mean(x**2) - np.mean(x)**2
     mean_xy = np.mean(x * y) - np.mean(x)*np.mean(y)
     a = mean_xy/mean_squared_x
     b = y.mean() - a * x.mean()
     assert a == pytest.approx(0.694915, 1e-6) # values form Mathematica
     assert b == pytest.approx(0.186441, 1e-6)

     S = np.sum((y - (a*x + b))**2)
     var_a_exact = S/(len(x) * (len(x) - 2) * mean_squared_x)
     var_b_exact = var_a_exact*np.mean(x ** 2)
     a_exact = a
     b_exact = b

     # We will now compare these exact results with values from symfit
     a, b, x_var = Parameter(name='a', value=3.0), Parameter(name='b'), Variable(name='x')
     model = a*x_var + b
     fit = Fit(model, x, y, absolute_sigma=False)
     fit_result = fit.execute()

     popt, pcov = curve_fit(lambda z, c, d: c * z + d, x, y,
                            Dfun=lambda p, x, y, func: np.transpose([x, np.ones_like(x)]))
                             # Dfun=lambda p, x, y, func: print(p, func, x, y))

     # curve_fit
     assert a_exact == pytest.approx(popt[0], 1e-4)
     assert b_exact == pytest,approx(popt[1], 1e-4)
     assert var_a_exact == pytest.approx(pcov[0][0], 1e-6)
     assert var_b_exact == pytest.approx(pcov[1][1], 1e-6)

     assert a_exact == pytest.approx(fit_result.params.a, 1e-4)
     assert b_exact == pytest.approx(fit_result.params.b, 1e-4)
     assert var_a_exact**0.5 == pytest.approx(fit_result.params.a_stdev, 1e-6)
     assert var_b_exact**0.5 == pytest.approx(fit_result.params.b_stdev, 1e-6)


def test_fixed_parameters():
    """
    Make sure fixed parameters don't change on fitting
    """
    a, b, c, d = parameters('a, b, c, d')
    x, y = variables('x, y')

    c.value = 4.0
    a.min, a.max = 1.0, 5.0  # Bounds are needed for DifferentialEvolution
    b.min, b.max = 1.0, 5.0
    c.min, c.max = 1.0, 5.0
    d.min, d.max = 1.0, 5.0
    c.fixed = True

    model = Model({y: a * exp(-(x - b)**2 / (2 * c**2)) + d})
    # Generate data
    xdata = np.linspace(0, 100)
    ydata = model(xdata, a=2, b=3, c=2, d=2).y

    for minimizer in subclasses(BaseMinimizer):
        if minimizer is ChainedMinimizer:
            continue
        else:
            fit = Fit(model, x=xdata, y=ydata, minimizer=minimizer)
            fit_result = fit.execute()
            # Should still be 4.0, not 2.0!
            assert 4.0 == fit_result.params['c']


def test_fixed_parameters_2():
    """
    Make sure parameter boundaries are respected
    """
    x = Parameter('x', min=1)
    y = Variable('y')
    model = Model({y: x**2})

    bounded_minimizers = list(subclasses(BoundedMinimizer))
    for minimizer in bounded_minimizers:
        if minimizer is MINPACK:
            # Not a MINPACKable problem because it only has a param
            continue
        fit = Fit(model, minimizer=minimizer)
        assert isinstance(fit.objective, MinimizeModel)
        if minimizer is DifferentialEvolution:
            # Also needs a max
            x.max = 10
            fit_result = fit.execute()
            x.max = None
        else:
            fit_result = fit.execute()
            assert fit_result.value(x) >= 1.0
            assert fit_result.value(x) <= 2.0
        assert fit.minimizer.bounds == [(1, None)]


def test_non_boundaries():
    """
    Make sure parameter boundaries are not invented
    """
    x = Parameter('x')
    y = Variable('y')
    model = Model({y: x**2})

    bounded_minimizers = list(subclasses(BoundedMinimizer))
    bounded_minimizers = [minimizer for minimizer in bounded_minimizers
                          if minimizer is not DifferentialEvolution]
    for minimizer in bounded_minimizers:
        # Not a MINPACKable problem because it only has a param
        if minimizer is MINPACK:
            continue
        fit = Fit(model, minimizer=minimizer)
        fit_result = fit.execute()
        assert fit_result.value(x) == pytest.approx(0.0)
        assert fit.minimizer.bounds == [(None, None)]


def test_single_param_model():
    """
    Added after #161, this tests if models with a single additive parameter
    are fitted properly. The problem with these models is that their
    jacobian is in principle just int 1, which is not the correct shape.

    No news is good news.
    :return:
    """
    T = Variable('T')
    l = Variable('l')
    s = Parameter('s', value=300)
    a = Parameter('a', value=300)
    model = {l: s + a + 1 / (1 + exp(- T))}

    temp_data = [270, 280, 285, 290, 295, 300, 310, 320]
    length_data = [8.33, 8.41, 8.45, 8.5, 8.54, 9.13, 9.27, 9.4]
    fit = Fit(model, l=length_data, T=temp_data)
    fit_result = fit.execute()

    # Raise the stakes by increasing the dimensionality of the data
    TT, LL = np.meshgrid(temp_data, length_data)
    fit = Fit(model, l=LL, T=TT)
    fit_result = fit.execute()


def test_model_from_dict():
    """
    Tries to create a model from a dictionary.
    """
    x, y_1, y_2 = variables('x, y_1, y_2')
    a, b = parameters('a, b')
    # This way the test fails rather than errors.
    try:
        Model({
            y_1: 2 * a * x,
            y_2: b * x**2
        })
    except Exception as error:
        pytest.fail('test_model_from_dict raised {}'.format(error))


def test_version():
    """
    Test if __version__ is availabe
    :return:
    """
    import symfit
    symfit.__version__
