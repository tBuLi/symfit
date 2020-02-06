from __future__ import division, print_function
import pytest

import numpy as np
import sympy
from scipy.integrate import simps
from scipy.optimize import NonlinearConstraint, minimize

from symfit import (
    variables, Variable, parameters, Parameter, ODEModel,
    Fit, Equality, D, Model, log, FitResults, GreaterThan, Eq, Ge, Le,
    CallableNumericalModel, HadamardProduct, GradientModel
)
from symfit.distributions import Gaussian
from symfit.core.minimizers import (
    SLSQP, MINPACK, TrustConstr, ScipyConstrainedMinimize, COBYLA
)
from symfit.core.support import key2str
from symfit.core.objectives import MinimizeModel, LogLikelihood, LeastSquares
from symfit.core.models import ModelError
from symfit import (
    Symbol, MatrixSymbol, Inverse, CallableModel, sqrt, Sum, Idx, symbols
)
from tests.test_minimizers import subclasses


"""
Tests for the `Fit` object. This object does
everything the normal `NumericalLeastSquares` does and more. Tests should
therefore cover the full range of scenarios `symfit` currently handles.
"""


def test_simple_kinetics():
    """
    Simple kinetics data to test fitting
    """
    tdata = np.array([10, 26, 44, 70, 120])
    adata = 10e-4 * np.array([44, 34, 27, 20, 14])
    a, b, t = variables('a, b, t')
    k, a0 = parameters('k, a0')
    k.value = 0.01
    # a0.value, a0.min, a0.max = 54 * 10e-4, 40e-4, 60e-4
    a0 = 54 * 10e-4

    model_dict = {
        D(a, t): - k * a**2,
        D(b, t): k * a**2,
    }

    ode_model = ODEModel(model_dict, initial={t: 0.0, a: a0, b: 0.0})

    fit = Fit(ode_model, t=tdata, a=adata, b=None)
    fit_result = fit.execute(tol=1e-9)

    assert fit_result.value(k) == pytest.approx(4.302875e-01, 1e-5)
    assert fit_result.stdev(k) == pytest.approx(6.447068e-03, 1e-5)


def test_global_fitting():
    """
    Test a global fitting scenario with datasets of unequal length. In this
    scenario, a quartic equation is fitted where the constant term is shared
    between the datasets. (e.g. identical background noise)
    """
    x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')
    y0, a_1, a_2, b_1, b_2 = parameters('y0, a_1, a_2, b_1, b_2')

    # The following vector valued function links all the equations together
    # as stated in the intro.
    model = Model({
        y_1: a_1 * x_1**2 + b_1 * x_1 + y0,
        y_2: a_2 * x_2**2 + b_2 * x_2 + y0,
    })

    # Generate data from this model
    # xdata = np.linspace(0, 10)
    xdata1 = np.linspace(0, 10)
    xdata2 = xdata1[::2]  # Make the sets of unequal size

    ydata1, ydata2 = model(x_1=xdata1, x_2=xdata2, a_1=101.3,
                           b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
    # Add some noise to make it appear like real data
    np.random.seed(1)
    ydata1 += np.random.normal(0, 2, size=ydata1.shape)
    ydata2 += np.random.normal(0, 2, size=ydata2.shape)

    xdata = [xdata1, xdata2]
    ydata = [ydata1, ydata2]

    # Guesses
    a_1.value = 100
    a_2.value = 50
    b_1.value = 1
    b_2.value = 1
    y0.value = 10

    eval_jac = model.eval_jacobian(x_1=xdata1, x_2=xdata2, a_1=101.3,
                                   b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
    assert len(eval_jac) == 2
    for comp in eval_jac:
        assert len(comp) == len(model.params)

    sigma_y = np.concatenate((np.ones(20), [2., 4., 5, 7, 3]))

    fit = Fit(model, x_1=xdata[0], x_2=xdata[1],
              y_1=ydata[0], y_2=ydata[1], sigma_y_2=sigma_y)
    fit_result = fit.execute()

    # fit_curves = model(x_1=xdata[0], x_2=xdata[1], **fit_result.params)
    assert fit_result.value(y0) == pytest.approx(1.061892e+01, 1e-03)
    assert fit_result.value(a_1) == pytest.approx(1.013269e+02, 1e-03)
    assert fit_result.value(a_2) == pytest.approx(5.625694e+01, 1e-03)
    assert fit_result.value(b_1) == pytest.approx(3.362240e-01, 1e-03)
    assert fit_result.value(b_2) == pytest.approx(1.565253e+00, 1e-03)


def test_named_fitting():
    xdata = np.linspace(1, 10, 10)
    ydata = 3*xdata**2

    a = Parameter('a', 1.0)
    b = Parameter('b', 2.5)
    x, y = variables('x, y')
    model = {y: a*x**b}

    fit = Fit(model, x=xdata, y=ydata)
    fit_result = fit.execute()
    assert isinstance(fit_result, FitResults)
    assert fit_result.value(a) == pytest.approx(3.0, 1e-3)
    assert fit_result.value(b) == pytest.approx(2.0, 1e-4)


def test_param_error_analytical():
    """
    Take an example in which the parameter errors are known and see if
    `Fit` reproduces them.

    It also needs to support the absolute_sigma argument.
    """
    N = 10000
    sigma = 25.0
    xn = np.arange(N, dtype=np.float)
    np.random.seed(110)
    yn = np.random.normal(size=xn.shape, scale=sigma)

    a = Parameter()
    y = Variable('y')
    model = {y: a}

    constr_fit = Fit(model, y=yn, sigma_y=sigma)
    constr_result = constr_fit.execute()

    fit = Fit(model, y=yn, sigma_y=sigma, minimizer=MINPACK)
    fit_result = fit.execute()

    assert fit_result.value(a) == pytest.approx(constr_result.value(a), 1e-5)
    assert fit_result.stdev(a) == pytest.approx(constr_result.stdev(a), 1e-5)

    # Analytical answer for mean of N(0,sigma):
    sigma_mu = sigma/N**0.5

    assert fit_result.value(a) == pytest.approx(np.mean(yn), 1e-5)
    assert fit_result.stdev(a) == pytest.approx(sigma_mu, 1e-5)

    # Compare for absolute_sigma = False.
    constr_fit = Fit(model, y=yn, sigma_y=sigma, absolute_sigma=False)
    constr_result = constr_fit.execute()

    fit = Fit(model, y=yn, sigma_y=sigma,
              minimizer=MINPACK, absolute_sigma=False)
    fit_result = fit.execute()

    assert fit_result.value(a) == pytest.approx(constr_result.value(a), 1e-5)
    assert fit_result.stdev(a) == pytest.approx(constr_result.stdev(a), 1e-5)


def test_grid_fitting():
    xdata = np.arange(-5, 5, 1)
    ydata = np.arange(5, 15, 1)
    xx, yy = np.meshgrid(xdata, ydata, sparse=False)

    zdata = (2.5*xx**2 + 3.0*yy**2)

    a = Parameter(value=2.4, max=2.75)
    b = Parameter(value=3.1, min=2.75)
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    new = {z: a*x**2 + b*y**2}

    fit = Fit(new, x=xx, y=yy, z=zdata)
    # results = fit.execute(options={'maxiter': 10})
    results = fit.execute()

    assert results.value(a) == pytest.approx(2.5, 1e-4)
    assert results.value(b) == pytest.approx(3.0, 1e-4)


@pytest.mark.skip(reason='Fit fails to compute the covariance matrix for a sparse grid.')
def test_grid_fitting_sparse():
    xdata = np.arange(-5, 5, 1)
    ydata = np.arange(5, 15, 1)
    xx, yy = np.meshgrid(xdata, ydata, sparse=True)

    zdata = (2.5*xx**2 + 3.0*yy**2)

    a = Parameter(value=2.4, max=2.75)
    b = Parameter(value=3.1, min=2.75)
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    new = {z: a*x**2 + b*y**2}

    fit = Fit(new, x=xx, y=yy, z=zdata)
    results = fit.execute()

    assert results.value(a) == pytest.approx(2.5, 1e-4)
    assert results.value(b) == pytest.approx(3.0, 1e-4)


def test_vector_constrained_fitting():
    """
    Tests `Fit` with vector models. The
    classical example of fitting measurements of the angles of a triangle is
    taken. In this case we know they should add up to 180 degrees, so this
    can be added as a constraint. Additionally, not even all three angles
    have to be provided with measurement data since the constrained means
    the angles are not independent.
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
    )
    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
    )
    fit_std = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        minimizer=MINPACK
    )
    fit_constrained = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        constraints=[Equality(a + b + c, 180)]
    )
    fit_none_result = fit_none.execute()
    fit_new_result = fit.execute()
    std_result = fit_std.execute()
    constr_result = fit_constrained.execute()

    # The total of averages should equal the total of the params by definition
    mean_total = np.mean(np.sum(xdata, axis=0))
    params_tot = std_result.value(a) + std_result.value(b) + std_result.value(c)
    assert mean_total / params_tot == pytest.approx(1.0, 1e-4)

    # The total after constraining to 180 should be exactly 180.
    params_tot = constr_result.value(a) + constr_result.value(b) + constr_result.value(c)
    assert isinstance(fit_constrained.minimizer, SLSQP)
    assert 180.0 == pytest.approx(params_tot, 1e-4)

    # The standard method and the Constrained object called without constraints
    # should behave roughly the same.
    assert fit_new_result.value(b) == pytest.approx(std_result.value(b), 1e-4)
    assert fit_new_result.value(a) == pytest.approx(std_result.value(a), 1e-4)
    assert fit_new_result.value(c) == pytest.approx(std_result.value(c), 1e-4)

    # When fitting with a dataset set to None, for this example the value of c
    # should be unaffected.
    assert fit_none_result.value(a) == pytest.approx(std_result.value(a), 1e-4)
    assert fit_none_result.value(b) == pytest.approx(std_result.value(b), 1e-4)
    assert fit_none_result.value(c) == pytest.approx(c.value)

    fit_none_constr = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=None,
        constraints=[Equality(a + b + c, 180)]
    )
    none_constr_result = fit_none_constr.execute()
    params_tot = none_constr_result.value(a) + none_constr_result.value(b) + none_constr_result.value(c)
    assert 180.0 == pytest.approx(params_tot, 1e-4)


def test_vector_parameter_error():
    """
    Tests `Fit` parameter error estimation with
    vector models. This is done by using the typical angles of a triangle
    example. For completeness, we throw in covariance between the angles.

    As per 0.5.0 this test has been updated in an important way. Previously
    the covariance matrix was estimated on a per component basis for global
    fitting problems. This was incorrect, but no solution was possible at
    the time. Now, we calculate the covariance matrix from the Hessian of
    the function being optimized, and so now the covariance is calculated
    correctly in those scenarios.

    As a result for this particular test however, it means we lose
    sensitivity to the error of each parameter separately. This makes sense,
    since the uncertainty is now being spread out over the components. To
    regain this, the user should just fit the components separately.
    """
    N = 10000
    a, b, c = parameters('a, b, c')
    a_i, b_i, c_i = variables('a_i, b_i, c_i')

    model = {a_i: a, b_i: b, c_i: c}

    np.random.seed(1)
    # Sample from a multivariate normal with correlation.
    pcov = np.array([[0.4, 0.3, 0.5], [0.3, 0.8, 0.4], [0.5, 0.4, 1.2]])
    xdata = np.random.multivariate_normal([10, 100, 70], pcov, N).T

    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
    )
    fit_std = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        minimizer=MINPACK
    )
    fit_new_result = fit.execute()
    std_result = fit_std.execute()

    # When no errors are given, we default to `absolute_sigma=False`, since
    # that is the best we can do.
    assert not fit.absolute_sigma
    assert not fit_std.absolute_sigma

    # The standard method and the Constrained object called without constraints
    # should give roughly the same parameter values.
    assert fit_new_result.value(a) == pytest.approx(std_result.value(a), 1e-3)
    assert fit_new_result.value(b) == pytest.approx(std_result.value(b), 1e-3)
    assert fit_new_result.value(c) == pytest.approx(std_result.value(c), 1e-3)

    # in this toy model, fitting is identical to simply taking the average
    assert fit_new_result.value(a) == pytest.approx(np.mean(xdata[0]), 1e-4)
    assert fit_new_result.value(b) == pytest.approx(np.mean(xdata[1]), 1e-4)
    assert fit_new_result.value(c) == pytest.approx(np.mean(xdata[2]), 1e-4)

    # All stdev's must be equal
    assert fit_new_result.stdev(a) == pytest.approx(fit_new_result.stdev(b), 1e-3)
    assert fit_new_result.stdev(a) == pytest.approx(fit_new_result.stdev(c), 1e-3)
    # Test for a miss on the exact value
    assert not fit_new_result.stdev(a) == pytest.approx(np.sqrt(pcov[0, 0]/N), 1e-3)
    assert not fit_new_result.stdev(b) == pytest.approx(np.sqrt(pcov[1, 1]/N), 1e-3)
    assert not fit_new_result.stdev(c) == pytest.approx(np.sqrt(pcov[2, 2]/N), 1e-3)

    # The standard object actually does not predict the right values for
    # stdev, because its method for computing them apparently does not allow
    # for vector valued functions.
    # So actually, for vector valued functions its better to use
    # Fit, though this does not give covariances.

    # With the correct values of sigma, absolute_sigma=True should be in
    # agreement with analytical.
    sigmadata = np.array([
        np.sqrt(pcov[0, 0]),
        np.sqrt(pcov[1, 1]),
        np.sqrt(pcov[2, 2])
    ])
    fit = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        sigma_a_i=sigmadata[0],
        sigma_b_i=sigmadata[1],
        sigma_c_i=sigmadata[2],
    )
    assert fit.absolute_sigma
    fit_result = fit.execute()
    # The standard deviation in the mean is stdev/sqrt(N),
    # see test_param_error_analytical
    assert fit_result.stdev(a)/np.sqrt(pcov[0, 0]/N) == pytest.approx(1.0, 1e-4)
    assert fit_result.stdev(b)/np.sqrt(pcov[1, 1]/N) == pytest.approx(1.0, 1e-4)
    assert fit_result.stdev(c)/np.sqrt(pcov[2, 2]/N) == pytest.approx(1.0, 1e-4)

    # Finally, we should confirm that with unrealistic sigma and
    # absolute_sigma=True, we are no longer in agreement with the analytical result
    # Let's take everything to be 1 to point out the dangers of doing so.
    sigmadata = np.array([1, 1, 1])
    fit2 = Fit(
        model=model,
        a_i=xdata[0],
        b_i=xdata[1],
        c_i=xdata[2],
        sigma_a_i=sigmadata[0],
        sigma_b_i=sigmadata[1],
        sigma_c_i=sigmadata[2],
        absolute_sigma=True
    )
    fit_result = fit2.execute()
    # Should be off bigly
    assert not fit_result.stdev(a)/np.sqrt(pcov[0, 0]/N) == pytest.approx(1.0, 1e-1)
    assert not fit_result.stdev(b)/np.sqrt(pcov[1, 1]/N) == pytest.approx(1.0, 1e-1)
    assert not fit_result.stdev(c)/np.sqrt(pcov[2, 2]/N) == pytest.approx(1.0, 1e-5)


def test_error_advanced():
    """
    Compare the error propagation of Fit against
    NumericalLeastSquares.
    Models an example from the mathematica docs and tries to replicate it:
    http://reference.wolfram.com/language/howto/FitModelsWithMeasurementErrors.html
    """
    data = [
        [0.9, 6.1, 9.5], [3.9, 6., 9.7], [0.3, 2.8, 6.6],
        [1., 2.2, 5.9], [1.8, 2.4, 7.2], [9., 1.7, 7.],
        [7.9, 8., 10.4], [4.9, 3.9, 9.], [2.3, 2.6, 7.4],
        [4.7, 8.4, 10.]
    ]
    xdata, ydata, zdata = [np.array(data) for data in zip(*data)]
    # errors = np.array([.4, .4, .2, .4, .1, .3, .1, .2, .2, .2])

    a = Parameter('a', 3.0)
    b = Parameter('b', 0.9)
    c = Parameter('c', 5.0)
    x = Variable('x')
    y = Variable('y')
    z = Variable('z')
    model = {z: a * log(b * x + c * y)}

    const_fit = Fit(model, xdata, ydata, zdata, absolute_sigma=False)
    assert len(const_fit.model(x=xdata, y=ydata, a=2, b=2, c=5)) == 1
    assert const_fit.model(x=xdata, y=ydata, a=2, b=2, c=5)[0].shape == (10,)

    assert len(const_fit.model.eval_jacobian(x=xdata, y=ydata, a=2, b=2, c=5)) == 1
    assert const_fit.model.eval_jacobian(x=xdata, y=ydata, a=2, b=2, c=5)[0].shape == (3, 10)

    assert len(const_fit.model.eval_hessian(x=xdata, y=ydata, a=2, b=2, c=5)) == 1
    assert const_fit.model.eval_hessian(x=xdata, y=ydata, a=2, b=2, c=5)[0].shape == (3, 3, 10)

    assert const_fit.objective(a=2, b=2, c=5).shape == tuple()
    assert const_fit.objective.eval_jacobian(a=2, b=2, c=5).shape == (3,)

    assert const_fit.objective.eval_hessian(a=2, b=2, c=5).shape == (3, 3)
    assert const_fit.objective.eval_hessian(a=2, b=2, c=5).dtype != object

    const_result = const_fit.execute()
    fit = Fit(model, xdata, ydata, zdata,
              absolute_sigma=False, minimizer=MINPACK)
    std_result = fit.execute()

    assert const_fit.absolute_sigma == fit.absolute_sigma

    assert const_result.value(a) == pytest.approx(std_result.value(a), 1e-4)
    assert const_result.value(b) == pytest.approx(std_result.value(b), 1e-4)
    assert const_result.value(c) == pytest.approx(std_result.value(c), 1e-4)

    # This used to be a tighter equality test, but since we now use the
    # Hessian we actually get a more accurate value from the standard fit
    # then for MINPACK. Hence we check if it is roughly equal, and if our
    # stdev is greater than that of minpack.
    assert const_result.stdev(a) / std_result.stdev(a) == pytest.approx(1, 1e-2)
    assert const_result.stdev(b) / std_result.stdev(b) == pytest.approx(1, 1e-1)
    assert const_result.stdev(c) / std_result.stdev(c) == pytest.approx(1, 1e-2)

    assert const_result.stdev(a) >= std_result.stdev(a)
    assert const_result.stdev(b) >= std_result.stdev(b)
    assert const_result.stdev(c) >= std_result.stdev(c)


def test_gaussian_2d_fitting():
    """
    Tests fitting to a scalar gaussian function with 2 independent
    variables. Very sensitive to initial guesses, and if they are chosen too
    restrictive Fit actually throws a tantrum.
    It therefore appears to be more sensitive than NumericalLeastSquares.
    """
    mean = (0.6, 0.4)  # x, y mean 0.6, 0.4
    cov = [[0.2**2, 0], [0, 0.1**2]]

    np.random.seed(0)
    data = np.random.multivariate_normal(mean, cov, 100000)

    # Insert them as y,x here as np fucks up cartesian conventions.
    ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100,
                                           range=[[0.0, 1.0], [0.0, 1.0]])
    xcentres = (xedges[:-1] + xedges[1:]) / 2
    ycentres = (yedges[:-1] + yedges[1:]) / 2

    # Make a valid grid to match ydata
    xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

    x0 = Parameter(value=mean[0], min=0.0, max=1.0)
    sig_x = Parameter(value=0.2, min=0.0, max=0.3)
    y0 = Parameter(value=mean[1], min=0.0, max=1.0)
    sig_y = Parameter(value=0.1, min=0.0, max=0.3)
    A = Parameter(value=np.mean(ydata), min=0.0)
    x = Variable('x')
    y = Variable('y')
    g = Variable('g')
    model = GradientModel({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
    fit = Fit(model, x=xx, y=yy, g=ydata)
    fit_result = fit.execute()

    assert fit_result.value(x0) == pytest.approx(np.mean(data[:, 0]), 1e-3)
    assert fit_result.value(y0) == pytest.approx(np.mean(data[:, 1]), 1e-3)
    assert np.abs(fit_result.value(sig_x)) == pytest.approx(np.std(data[:, 0]), 1e-2)
    assert np.abs(fit_result.value(sig_y)) == pytest.approx(np.std(data[:, 1]), 1e-2)
    assert (fit_result.r_squared, 0.96)

    # Compare with industry standard MINPACK
    fit_std = Fit(model, x=xx, y=yy, g=ydata, minimizer=MINPACK)
    fit_std_result = fit_std.execute()

    assert fit_std_result.value(x0) == pytest.approx(fit_result.value(x0), 1e-4)
    assert fit_std_result.value(y0) == pytest.approx(fit_result.value(y0), 1e-4)
    assert fit_std_result.value(sig_x) == pytest.approx(fit_result.value(sig_x), 1e-4)
    assert fit_std_result.value(sig_y) == pytest.approx(fit_result.value(sig_y), 1e-4)
    assert fit_std_result.r_squared == pytest.approx(fit_result.r_squared, 1e-4)


def test_fixed_and_constrained():
    """
    Taken from #165. Fixing parameters and constraining others caused a
    TypeError: missing a required argument: 'theta1', which was caused by a
    mismatch in the shape of the initial guesses given and the number of
    parameters constraints expected. The initial_guesses no longer contained
    those corresponding to fixed parameters.
    """
    phi1, phi2, theta1, theta2 = parameters('phi1, phi2, theta1, theta2')
    x, y = variables('x, y')

    model_dict = {y: (1 + x * theta1 + theta2 * x ** 2) /
                     (1 + phi1 * x * theta1 + phi2 * theta2 * x ** 2)}
    constraints = [GreaterThan(theta1, theta2)]

    xdata = np.array([0., 0.000376, 0.000752, 0.0015, 0.00301, 0.00601, 0.00902])
    ydata = np.array([1., 1.07968041, 1.08990638, 1.12151629, 1.13068452, 1.15484109, 1.19883952])

    phi1.value = 0.845251484373516
    phi1.fixed = True

    phi2.value = 0.7105427053026403
    phi2.fixed = True

    fit = Fit(model_dict, x=xdata, y=ydata,
              constraints=constraints, minimizer=SLSQP)
    fit_result_slsqp = fit.execute()
    # The data and fixed parameters should be partialed away.
    objective_kwargs = {
        phi2.name: phi2.value,
        phi1.name: phi1.value,
        x.name: xdata,
    }
    constraint_kwargs = {
        phi2.name: phi2.value,
        phi1.name: phi1.value,
    }
    for index, constraint in enumerate(fit.minimizer.constraints):
        assert isinstance(constraint, MinimizeModel)
        assert constraint.model == fit.constraints[index]
        assert constraint.data == fit.data
        assert constraint.data == fit.objective.data

        # Data should be the same memory location so they can share state.
        assert id(fit.objective.data) == id(constraint.data)

        # Test if the fixed params have been partialed away
        assert key2str(constraint._invariant_kwargs).keys() == constraint_kwargs.keys()
        assert key2str(fit.objective._invariant_kwargs).keys() == objective_kwargs.keys()

    # Compare the shapes. The constraint shape should now be the same as
    # that of the objective
    obj_val = fit.minimizer.objective(fit.minimizer.initial_guesses)
    obj_jac = fit.minimizer.wrapped_jacobian(fit.minimizer.initial_guesses)

    # scalars don't have lengths
    with pytest.raises(TypeError):
        len(obj_val)
    assert len(obj_jac) == 2

    for index, constraint in enumerate(fit.minimizer.wrapped_constraints):
        assert constraint['type'] == 'ineq'
        assert 'args' not in constraint
        assert callable(constraint['fun'])
        assert callable(constraint['jac'])

        # The argument should be the partialed Constraint object
        assert constraint['fun'] == fit.minimizer.constraints[index]
        assert isinstance(constraint['fun'], MinimizeModel)
        assert 'jac' in constraint

        # Test the shapes
        cons_val = constraint['fun'](fit.minimizer.initial_guesses)
        cons_jac = constraint['jac'](fit.minimizer.initial_guesses)
        assert cons_val.shape == (1,)
        assert isinstance(cons_val[0], float)
        assert obj_jac.shape == cons_jac.shape
        assert obj_jac.shape == (2,)


def test_interdependency_constrained():
    """
    Test a model with interdependent components, and with constraints which
    depend on the Model's output.
    This is done in the MatrixSymbol formalism, using a Tikhonov
    regularization as an example. In this, a matrix inverse has to be
    calculated and is used multiple times. Therefore we split that term of
    into a seperate component, so the inverse only has to be computed once
    per model call.

    See https://arxiv.org/abs/1901.05348 for a more detailed background.
    """
    N = Symbol('N', integer=True)
    M = MatrixSymbol('M', N, N)
    W = MatrixSymbol('W', N, N)
    I = MatrixSymbol('I', N, N)
    y = MatrixSymbol('y', N, 1)
    c = MatrixSymbol('c', N, 1)
    a, = parameters('a')
    z, = variables('z')
    i = Idx('i')

    model_dict = {
        W: Inverse(I + M / a ** 2),
        c: - W * y,
        z: sqrt(c.T * c)
    }
    # Sympy currently does not support derivatives of matrix expressions,
    # so we use CallableModel instead of Model.
    model = CallableModel(model_dict)

    # Generate data
    iden = np.eye(2)
    M_mat = np.array([[2, 1], [3, 4]])
    y_vec = np.array([[3], [5]])
    eval_model = model(I=iden, M=M_mat, y=y_vec, a=0.1)
    # Calculate the answers 'manually' so I know it was done properly
    W_manual = np.linalg.inv(iden + M_mat / 0.1 ** 2)
    c_manual = - np.atleast_2d(W_manual.dot(y_vec))
    z_manual = np.atleast_1d(np.sqrt(c_manual.T.dot(c_manual)))

    assert y_vec.shape == (2, 1)
    assert M_mat.shape == (2, 2)
    assert iden.shape == (2, 2)
    assert W_manual.shape == (2, 2)
    assert c_manual.shape == (2, 1)
    assert z_manual.shape == (1, 1)
    assert W_manual == pytest.approx(eval_model.W)
    assert c_manual == pytest.approx(eval_model.c)
    assert z_manual == pytest.approx(eval_model.z)
    fit = Fit(model, z=z_manual, I=iden, M=M_mat, y=y_vec)
    fit_result = fit.execute()

    # See if a == 0.1 was reconstructed properly. Since only a**2 features
    # in the equations, we check for the absolute value. Setting a.min = 0.0
    # is not appreciated by the Minimizer, it seems.
    assert np.abs(fit_result.value(a)) == pytest.approx(0.1)


def test_data_for_constraint():
    """
    Test the signature handling when constraints are at play. Constraints
    should take seperate data, but still kwargs that are not found in either
    the model nor the constraints should raise an error.
    """
    A, mu, sig = parameters('A, mu, sig')
    x, y, Y = variables('x, y, Y')

    model = Model({y: A * Gaussian(x, mu=mu, sig=sig)})
    constraint = Model.as_constraint(Y, model, constraint_type=Eq)

    np.random.seed(2)
    xdata = np.random.normal(1.2, 2, 10)
    ydata, xedges = np.histogram(xdata, bins=int(np.sqrt(len(xdata))), density=True)

    # Allowed
    fit = Fit(model, x=xdata, y=ydata, Y=2, constraints=[constraint])
    assert isinstance(fit.objective, LeastSquares)
    assert isinstance(fit.minimizer.constraints[0], MinimizeModel)
    fit = Fit(model, x=xdata, y=ydata)
    assert isinstance(fit.objective, LeastSquares)
    fit = Fit(model, x=xdata, objective=LogLikelihood)
    assert isinstance(fit.objective, LogLikelihood)

    # Not allowed
    with pytest.raises(TypeError):
        fit = Fit(model, x=xdata, y=ydata, Y=2)

    with pytest.raises(TypeError):
        fit = Fit(model, x=xdata, y=ydata, Y=2, Z=3, constraints=[constraint])

    with pytest.raises(TypeError):
        fit = Fit(model, x=xdata, y=ydata, objective=LogLikelihood)


def test_constrained_dependent_on_model():
    """
    For a simple Gaussian distribution, we test if Models of various types
    can be used as constraints. Of particular interest are NumericalModels,
    which can be used to fix the integral of the model during the fit to 1,
    as it should be for a probability distribution.
    :return:
    """
    A, mu, sig = parameters('A, mu, sig')
    x, y, Y = variables('x, y, Y')
    i = Idx('i', (0, 1000))
    sig.min = 0.0

    model = GradientModel({y: A * Gaussian(x, mu=mu, sig=sig)})

    # Generate data, 100 samples from a N(1.2, 2) distribution
    np.random.seed(2)
    xdata = np.random.normal(1.2, 2, 1000)
    ydata, xedges = np.histogram(xdata, bins=int(np.sqrt(len(xdata))), density=True)
    xcentres = (xedges[1:] + xedges[:-1]) / 2

    # Unconstrained fit
    fit = Fit(model, x=xcentres, y=ydata)
    unconstr_result = fit.execute()

    # Constraints must be scalar models.
    with pytest.raises(ModelError):
        Model.as_constraint([A - 1, sig - 1], model, constraint_type=Eq)

    constraint_exact = Model.as_constraint(A * sqrt(2 * sympy.pi) * sig - 1,
                                           model, constraint_type=Eq)
    # Only when explicitly asked, do models behave as constraints.
    assert hasattr(constraint_exact, 'constraint_type')
    assert constraint_exact.constraint_type == Eq
    assert not hasattr(model, 'constraint_type')

    # Now lets make some valid constraints and see if they are respected!
    # FIXME These first two should be symbolical integrals over `y` instead,
    # but currently this is not converted into a numpy/scipy function. So
    # instead the first two are not valid constraints.
    constraint_model = Model.as_constraint(A - 1, model, constraint_type=Eq)
    constraint_exact = Eq(A, 1)
    constraint_num = CallableNumericalModel.as_constraint(
        {Y: lambda x, y: simps(y, x) - 1},  # Integrate using simps
        model=model,
        connectivity_mapping={Y: {x, y}},
        constraint_type=Eq
    )

    # Test for all these different types of constraint.
    for constraint in [constraint_model, constraint_exact, constraint_num]:
        if not isinstance(constraint, Eq):
            assert constraint.constraint_type == Eq

        xcentres = (xedges[1:] + xedges[:-1]) / 2
        fit = Fit(model, x=xcentres, y=ydata, constraints=[constraint])
        # Test if conversion into a constraint was done properly
        fit_constraint = fit.constraints[0]
        assert fit.model.params == fit_constraint.params
        assert fit_constraint.constraint_type == Eq

        con_map = fit_constraint.connectivity_mapping
        if isinstance(constraint, CallableNumericalModel):
            assert con_map == {Y: {x, y}, y: {x, mu, sig, A}}
            assert fit_constraint.independent_vars == [x]
            assert fit_constraint.dependent_vars == [Y]
            assert fit_constraint.interdependent_vars == [y]
            assert fit_constraint.params == [A, mu, sig]
        else:
            # TODO if these constraints can somehow be written as integrals
            # depending on y and x this if/else should be removed.
            assert con_map == {fit_constraint.dependent_vars[0]: {A}}
            assert fit_constraint.independent_vars == []
            assert len(fit_constraint.dependent_vars) == 1
            assert fit_constraint.interdependent_vars == []
            assert fit_constraint.params == [A, mu, sig]

        # Finally, test if the constraint worked
        fit_result = fit.execute(options={'eps': 1e-15, 'ftol': 1e-10})
        unconstr_value = fit.minimizer.wrapped_constraints[0]['fun'](**unconstr_result.params)
        constr_value = fit.minimizer.wrapped_constraints[0]['fun'](**fit_result.params)

        # TODO because of a bug by pytest we have to solve it like this
        assert constr_value[0] == pytest.approx(0, abs=1e-10)
    # And if it was very poorly met before
    assert not unconstr_value[0] == pytest.approx(0.0, 1e-1)


def test_constrained_dependent_on_matrixmodel():
    """
    Similar to test_constrained_dependent_on_model, but now using
    MatrixSymbols. This is much more powerful, since now the constraint can
    really be written down as a symbolical one as well.
    """
    A, mu, sig = parameters('A, mu, sig')
    M = symbols('M', integer=True)  # Number of measurements

    # Create vectors for all the quantities
    x = MatrixSymbol('x', M, 1)
    dx = MatrixSymbol('dx', M, 1)
    y = MatrixSymbol('y', M, 1)
    I = MatrixSymbol('I', M, 1)  # 'identity' vector
    Y = MatrixSymbol('Y', 1, 1)
    B = MatrixSymbol('B', M, 1)
    i = Idx('i', M)

    # Looks overly complicated, but it's just a simple Gaussian
    model = CallableModel(
        {y: A * sympy.exp(- HadamardProduct(B, B) / (2 * sig**2)) / sympy.sqrt(2*sympy.pi*sig**2),
         B: (x - mu * I)}
    )
    assert model.independent_vars == [I, x]
    assert model.dependent_vars == [y]
    assert model.interdependent_vars == [B]
    assert model.params == [A, mu, sig]

    # Generate data, sample from a N(1.2, 2) distribution. Has to be 2D.
    np.random.seed(2)
    # TODO: sample points on a Guassian and add appropriate noise.
    xdata = np.random.normal(1.2, 2, size=10000)
    ydata, xedges = np.histogram(xdata, bins=int(np.sqrt(len(xdata))), density=True)
    xcentres = np.atleast_2d((xedges[1:] + xedges[:-1]) / 2).T
    xdiff = np.atleast_2d((xedges[1:] - xedges[:-1])).T
    ydata = np.atleast_2d(ydata).T
    Idata = np.ones_like(xcentres)

    assert xcentres.shape == (int(np.sqrt(len(xdata))), 1)
    assert xdiff.shape == (int(np.sqrt(len(xdata))), 1)
    assert ydata.shape == (int(np.sqrt(len(xdata))), 1)

    fit = Fit(model, x=xcentres, y=ydata, I=Idata)
    unconstr_result = fit.execute()

    constraint = CallableModel({Y: Sum(y[i, 0] * dx[i, 0], i) - 1})

    with pytest.raises(ModelError):
        fit = Fit(model, x=xcentres, y=ydata, dx=xdiff, M=len(xcentres),
                  I=Idata, constraints=[constraint])

    constraint = CallableModel.as_constraint(
        {Y: Sum(y[i, 0] * dx[i, 0], i) - 1},
        model=model,
        constraint_type=Eq
    )
    assert constraint.independent_vars == [I, M, dx, x]
    assert constraint.dependent_vars == [Y]
    assert constraint.interdependent_vars == [B, y]
    assert constraint.params == [A, mu, sig]
    assert constraint.constraint_type == Eq

    # Provide the extra data needed for the constraints as well
    fit = Fit(model, x=xcentres, y=ydata, dx=xdiff, M=len(xcentres),
              I=Idata, constraints=[constraint])

    # After treatment, our constraint should have `y` & `b` dependencies
    assert fit.constraints[0].independent_vars == [I, M, dx, x]
    assert fit.constraints[0].dependent_vars == [Y]
    assert fit.constraints[0].interdependent_vars == [B, y]
    assert fit.constraints[0].params == [A, mu, sig]
    assert fit.constraints[0].constraint_type == Eq
    assert isinstance(fit.objective, LeastSquares)
    assert isinstance(fit.minimizer.constraints[0], MinimizeModel)

    assert {k for k, v in fit.data.items() if v is not None} == {x, y, dx, M, I, fit.model.sigmas[y]}
    # These belong to internal variables
    assert {k for k, v in fit.data.items() if v is None} == {constraint.sigmas[Y], Y}

    constr_result = fit.execute()
    # The constraint should not be met for the unconstrained fit
    assert not fit.minimizer.wrapped_constraints[0]['fun'](**unconstr_result.params)[0] == pytest.approx(0, 1e-3)
    # And at high precision with constraint
    # TODO Change after resolve bug at pytest
    assert fit.minimizer.wrapped_constraints[0]['fun'](**constr_result.params)[0] == pytest.approx(0, abs=1e-8)

    # Constraining will negatively effect the R^2 value, but...
    assert constr_result.r_squared < unconstr_result.r_squared
    # both should be pretty good
    assert constr_result.r_squared > 0.99


def test_fixed_and_constrained_tc():
    """
    Taken from #165. Make sure the TrustConstr minimizer can deal with
    constraints and fixed parameters.
    """
    phi1, phi2, theta1, theta2 = parameters('phi1, phi2, theta1, theta2')
    x, y = variables('x, y')

    model_dict = {y: (1 + x * theta1 + theta2 * x ** 2) / (1 + phi1 * x * theta1 + phi2 * theta2 * x ** 2)}
    constraints = [GreaterThan(theta1, theta2)]

    xdata = np.array([0., 0.000376, 0.000752, 0.0015, 0.00301, 0.00601, 0.00902])
    ydata = np.array([1., 1.07968041, 1.08990638, 1.12151629, 1.13068452, 1.15484109, 1.19883952])

    phi1.value = 0.845251484373516
    phi1.fixed = True

    phi2.value = 0.7105427053026403
    phi2.fixed = True

    fit = Fit(model_dict, x=xdata, y=ydata,
              constraints=constraints, minimizer=TrustConstr)
    fit_result_tc = fit.execute()
    # The data and fixed parameters should be partialed away.
    objective_kwargs = {
        phi2.name: phi2.value,
        phi1.name: phi1.value,
        x.name: xdata,
    }
    constraint_kwargs = {
        phi2.name: phi2.value,
        phi1.name: phi1.value,
    }
    for index, constraint in enumerate(fit.minimizer.constraints):
        assert isinstance(constraint, MinimizeModel)
        assert constraint.model == fit.constraints[index]
        assert constraint.data == fit.data
        assert constraint.data == fit.objective.data

        # Data should be the same memory location so they can share state.
        assert id(fit.objective.data) == id(constraint.data)

        # Test if the data and fixed params have been partialed away
        assert key2str(constraint._invariant_kwargs).keys() == constraint_kwargs.keys()
        assert key2str(fit.objective._invariant_kwargs).keys() == objective_kwargs.keys()

    # Compare the shapes. The constraint shape should now be the same as
    # that of the objective
    obj_val = fit.minimizer.objective(fit.minimizer.initial_guesses)
    obj_jac = fit.minimizer.wrapped_jacobian(fit.minimizer.initial_guesses)
    with pytest.raises(TypeError):
        len(obj_val)  # scalars don't have lengths
    assert len(obj_jac) == 2

    for index, constraint in enumerate(fit.minimizer.wrapped_constraints):
        assert callable(constraint.fun)
        assert callable(constraint.jac)

        # The argument should be the partialed Constraint object
        assert constraint.fun == fit.minimizer.constraints[index]
        assert isinstance(constraint.fun, MinimizeModel)

        # Test the shapes
        cons_val = constraint.fun(fit.minimizer.initial_guesses)
        cons_jac = constraint.jac(fit.minimizer.initial_guesses)
        assert cons_val.shape == (1,)
        assert isinstance(cons_val[0], float)
        assert obj_jac.shape == cons_jac.shape
        assert obj_jac.shape == (2,)


def test_constrainedminimizers():
    """
    Compare the different constrained minimizers, to make sure all support
    constraints, and converge to the same answer.
    """
    minimizers = list(subclasses(ScipyConstrainedMinimize))
    x = Parameter('x', value=-1.0)
    y = Parameter('y', value=1.0)
    z = Variable('z')
    model = Model({z: 2 * x * y + 2 * x - x ** 2 - 2 * y ** 2})

    # First we try an unconstrained fit
    results = []
    for minimizer in minimizers:
        fit = Fit(- model, minimizer=minimizer)
        assert isinstance(fit.objective, MinimizeModel)
        fit_result = fit.execute(tol=1e-15)
        results.append(fit_result)

    # Compare the parameter values.
    for r1, r2 in zip(results[:-1], results[1:]):
        assert r1.value(x) == pytest.approx(r2.value(x), 1e-6)
        assert r1.value(y) == pytest.approx(r2.value(y), 1e-6)
        assert r1.covariance_matrix == pytest.approx(r2.covariance_matrix)

    constraints = [
        Ge(y - 1, 0),  # y - 1 >= 0,
        Eq(x ** 3 - y, 0),  # x**3 - y == 0,
    ]

    # Constrained fit.
    results = []
    for minimizer in minimizers:
        if minimizer is COBYLA:
            # COBYLA only supports inequality.
            continue
        fit = Fit(- model, constraints=constraints, minimizer=minimizer)
        fit_result = fit.execute(tol=1e-15)
        results.append(fit_result)

    for r1, r2 in zip(results[:-1], results[1:]):
        assert r1.value(x) == pytest.approx(r2.value(x), 1e-6)
        assert r1.value(y) == pytest.approx(r2.value(y), 1e-6)
        assert r1.covariance_matrix == pytest.approx(r2.covariance_matrix)


def test_trustconstr():
    """
    Solve the standard constrained example from
    https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
    using the trust-constr method.
    """
    def func(x, sign=1.0):
        """ Objective function """
        return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

    def func_jac(x, sign=1.0):
        """ Derivative of objective function """
        dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
        dfdx1 = sign*(2*x[0] - 4*x[1])
        return np.array([dfdx0, dfdx1])

    def func_hess(x, sign=1.0):
        """ Hessian of objective function """
        dfdx2 = sign*(-2)
        dfdxdy = sign * 2
        dfdy2 = sign * (-4)
        return np.array([[dfdx2, dfdxdy], [dfdxdy, dfdy2]])

    def cons_f(x):
        return [x[1] - 1, x[0]**3 - x[1]]

    def cons_J(x):
        return [[0, 1], [3 * x[0] ** 2, -1]]

    def cons_H(x, v):
        return v[0] * np.zeros((2, 2)) + v[1] * np.array([[6 * x[0], 0], [0, 0]])

    # Unconstrained fit
    res = minimize(func, [-1.0, 1.0], args=(-1.0,),
                   jac=func_jac, hess=func_hess, method='trust-constr')
    assert res.x == pytest.approx([2, 1])

    # Constrained fit
    nonlinear_constraint = NonlinearConstraint(cons_f, 0, [np.inf, 0],
                                               jac=cons_J, hess=cons_H)
    res_constr = minimize(func, [-1.0, 1.0], args=(-1.0,), tol=1e-15,
                          jac=func_jac, hess=func_hess, method='trust-constr',
                          constraints=[nonlinear_constraint])
    assert res_constr.x == pytest.approx([1, 1])

    # Symfit equivalent code
    x = Parameter('x', value=-1.0)
    y = Parameter('y', value=1.0)
    z = Variable('z')
    model = Model({z: 2 * x * y + 2 * x - x ** 2 - 2 * y ** 2})

    # Unconstrained fit first, see if we get the known result.
    fit = Fit(-model, minimizer=TrustConstr)
    fit_result = fit.execute()
    assert list(fit_result.params.values()) == pytest.approx([2, 1])

    # Now we are ready for the constrained fit.
    constraints = [
        Le(- y + 1, 0),  # y - 1 >= 0,
        Eq(x ** 3 - y, 0),  # x**3 - y == 0,
    ]
    fit = Fit(-model, constraints=constraints, minimizer=TrustConstr)
    fit_result = fit.execute(tol=1e-15)

    # Test if the constrained results are equal
    assert list(fit_result.params.values()) == pytest.approx(res_constr.x)
