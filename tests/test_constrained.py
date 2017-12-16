from __future__ import division, print_function
import unittest

import numpy as np
from symfit import (
    variables, Variable, parameters, Parameter, ODEModel,
    Fit, Equality, D, Model, log, FitResults
)
from symfit.distributions import Gaussian
from symfit.core.minimizers import SLSQP, MINPACK

class TestConstrained(unittest.TestCase):
    """
    Tests for the `Fit` object. This object does
    everything the normal `NumericalLeastSquares` does and more. Tests should
    therefore cover the full range of scenarios `symfit` currently handles.
    """
    def test_simple_kinetics(self):
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

        self.assertAlmostEqual(fit_result.value(k) / 4.302875e-01, 1.0, 5)
        self.assertAlmostEqual(fit_result.stdev(k) / 6.447068e-03, 1.0, 5)

    def test_global_fitting(self):
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

        ydata1, ydata2 = model(x_1=xdata1, x_2=xdata2, a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
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

        sigma_y = np.concatenate((np.ones(20), [2., 4., 5, 7, 3]))

        fit = Fit(model, x_1=xdata[0], x_2=xdata[1],
                                               y_1=ydata[0], y_2=ydata[1], sigma_y_2=sigma_y)
        fit_result = fit.execute()

        # fit_curves = model(x_1=xdata[0], x_2=xdata[1], **fit_result.params)
        self.assertAlmostEqual(fit_result.value(y0), 1.061892e+01, 3)
        self.assertAlmostEqual(fit_result.value(a_1), 1.013269e+02, 3)
        self.assertAlmostEqual(fit_result.value(a_2), 5.625694e+01, 3)
        self.assertAlmostEqual(fit_result.value(b_1), 3.362240e-01, 3)
        self.assertAlmostEqual(fit_result.value(b_2), 1.565253e+00, 3)

    def test_named_fitting(self):
        xdata = np.linspace(1, 10, 10)
        ydata = 3*xdata**2

        a = Parameter(1.0)
        b = Parameter(2.5)
        x, y = variables('x, y')
        model = {y: a*x**b}

        fit = Fit(model, x=xdata, y=ydata)
        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.value(a), 3.0, 3)
        self.assertAlmostEqual(fit_result.value(b), 2.0, 4)

    def test_param_error_analytical(self):
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
        y = Variable()
        model = {y: a}

        constr_fit = Fit(model, y=yn, sigma_y=sigma)
        constr_result = constr_fit.execute()

        fit = Fit(model, y=yn, sigma_y=sigma, minimizer=MINPACK)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), constr_result.value(a), 5)
        self.assertAlmostEqual(fit_result.stdev(a), constr_result.stdev(a), 5)

        # Analytical answer for mean of N(0,sigma):
        sigma_mu = sigma/N**0.5

        # self.assertAlmostEqual(fit_result.value(a), mu, 5)
        self.assertAlmostEqual(fit_result.value(a), np.mean(yn), 5)
        self.assertAlmostEqual(fit_result.stdev(a), sigma_mu, 5)

        # Compare for absolute_sigma = False.
        constr_fit = Fit(model, y=yn, sigma_y=sigma, absolute_sigma=False)
        constr_result = constr_fit.execute()

        fit = Fit(model, y=yn, sigma_y=sigma, minimizer=MINPACK, absolute_sigma=False)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), constr_result.value(a), 5)
        self.assertAlmostEqual(fit_result.stdev(a), constr_result.stdev(a), 5)

    def test_grid_fitting(self):
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(5, 15, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter(2.4, max=2.75)
        b = Parameter(3.1, min=2.75)
        x = Variable()
        y = Variable()
        z = Variable()
        new = {z: a*x**2 + b*y**2}

        fit = Fit(new, x=xx, y=yy, z=zdata)
        # results = fit.execute(options={'maxiter': 10})
        results = fit.execute()

        self.assertAlmostEqual(results.value(a), 2.5, 4)
        self.assertAlmostEqual(results.value(b), 3.0, 4)

    @unittest.skip('Fit fails to compute the '
                   'covariance matrix for a sparse grid.')
    def test_grid_fitting_sparse(self):
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(5, 15, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=True)

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter(2.4, max=2.75)
        b = Parameter(3.1, min=2.75)
        x = Variable()
        y = Variable()
        z = Variable()
        new = {z: a*x**2 + b*y**2}

        fit = Fit(new, x=xx, y=yy, z=zdata)
        results = fit.execute()

        self.assertAlmostEqual(results.value(a), 2.5, 4)
        self.assertAlmostEqual(results.value(b), 3.0, 4)

    def test_vector_constrained_fitting(self):
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
            minimizer = MINPACK
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
        self.assertAlmostEqual(mean_total / params_tot, 1.0, 4)

        # The total after constraining to 180 should be exactly 180.
        params_tot = constr_result.value(a) + constr_result.value(b) + constr_result.value(c)
        self.assertIsInstance(fit_constrained.minimizer, SLSQP)
        self.assertAlmostEqual(180.0, params_tot, 4)

        # The standard method and the Constrained object called without constraints
        # should behave roughly the same.
        self.assertAlmostEqual(fit_new_result.value(a), std_result.value(a), 4)
        self.assertAlmostEqual(fit_new_result.value(b), std_result.value(b), 4)
        self.assertAlmostEqual(fit_new_result.value(c), std_result.value(c), 4)

        # When fitting with a dataset set to None, for this example the value of c
        # should be unaffected.
        self.assertAlmostEqual(fit_none_result.value(a), std_result.value(a), 4)
        self.assertAlmostEqual(fit_none_result.value(b), std_result.value(b), 4)
        self.assertAlmostEqual(fit_none_result.value(c), c.value)

        fit_none_constr = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=None,
            constraints=[Equality(a + b + c, 180)]
        )
        none_constr_result = fit_none_constr.execute()
        params_tot = none_constr_result.value(a) + none_constr_result.value(b) + none_constr_result.value(c)
        self.assertAlmostEqual(180.0, params_tot, 4)

    def test_vector_parameter_error(self):
        """
        Tests `Fit` parameter error estimation with
        vector models. This is done by using the typical angles of a triangle
        example. For completeness, we throw in covariance between the angles.

        As it stands now, `Fit` is able to correctly
        predict the values of the parameters an their standard deviations, but
        it is not able to give the covariances. Those are therefore returned as
        nan, to prevent people from citing them as 0.0.
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
            # absolute_sigma=False
        )
        fit_std = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
            minimizer = MINPACK
            # sigma_a_i=np.ones_like(xdata[0]),
            # sigma_b_i=np.ones_like(xdata[0]),
            # sigma_c_i=np.ones_like(xdata[0]),
            # absolute_sigma=False
        )
        fit_new_result = fit.execute()
        std_result = fit_std.execute()

        # When no errors are given, we default to `absolute_sigma=False`, since
        # that is the best we can do.
        self.assertFalse(fit.absolute_sigma)
        self.assertFalse(fit_std.absolute_sigma)

        # The standard method and the Constrained object called without constraints
        # should give roughly the same parameter values.
        self.assertAlmostEqual(fit_new_result.value(a), std_result.value(a), 3)
        self.assertAlmostEqual(fit_new_result.value(b), std_result.value(b), 3)
        self.assertAlmostEqual(fit_new_result.value(c), std_result.value(c), 3)

        # in this toy model, fitting is identical to simply taking the average
        self.assertAlmostEqual(fit_new_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_new_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_new_result.value(c), np.mean(xdata[2]), 4)

        # Since no sigma were provided, absolute_sigma=False. Therefore the
        # standard deviation doesn't match the expected value, but it does match the emperical value
        self.assertAlmostEqual(fit_new_result.stdev(a)/(np.std(xdata[0], ddof=1)/np.sqrt(N)), 1.0, 3)
        self.assertAlmostEqual(fit_new_result.stdev(b)/(np.std(xdata[1], ddof=1)/np.sqrt(N)), 1.0, 3)
        self.assertAlmostEqual(fit_new_result.stdev(c)/(np.std(xdata[2], ddof=1)/np.sqrt(N)), 1.0, 3)
        # Test for a miss on the exact value
        self.assertNotAlmostEqual(fit_new_result.stdev(a)/np.sqrt(pcov[0, 0]/N), 1.0, 3)
        self.assertNotAlmostEqual(fit_new_result.stdev(b)/np.sqrt(pcov[1, 1]/N), 1.0, 3)
        self.assertNotAlmostEqual(fit_new_result.stdev(c)/np.sqrt(pcov[2, 2]/N), 1.0, 3)

        # The standard object actually does not predict the right values for
        # stdev, because its method for computing them apperantly does not allow
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
        self.assertTrue(fit.absolute_sigma)
        fit_result = fit.execute()
        # The standard deviation in the mean is stdev/sqrt(N),
        # see test_param_error_analytical
        self.assertAlmostEqual(fit_result.stdev(a)/np.sqrt(pcov[0, 0]/N), 1.0, 4)
        self.assertAlmostEqual(fit_result.stdev(b)/np.sqrt(pcov[1, 1]/N), 1.0, 4)
        self.assertAlmostEqual(fit_result.stdev(c)/np.sqrt(pcov[2, 2]/N), 1.0, 4)


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
        self.assertNotAlmostEqual(fit_result.stdev(a)/np.sqrt(pcov[0, 0]/N), 1.0, 1)
        self.assertNotAlmostEqual(fit_result.stdev(b)/np.sqrt(pcov[1, 1]/N), 1.0, 1)
        self.assertNotAlmostEqual(fit_result.stdev(c)/np.sqrt(pcov[2, 2]/N), 1.0, 1)

    def test_covariances(self):
        """
        Compare the equal and unequal length handeling of `HasCovarianceMatrix`.
        If it works properly, the unequal length method should reduce to the
        equal length one if called qith equal length data. Computing unequal
        dataset length covariances remains something to be careful with, but
        this backwards compatibility provides some validation.
        """
        N = 10000
        a, b, c = parameters('a, b, c')
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        np.random.seed(1)
        # Sample from a multivariate normal with correlation.
        pcov = 1e-1 * np.array([[0.4, 0.3, 0.5], [0.3, 0.8, 0.4], [0.5, 0.4, 1.2]])
        xdata = np.random.multivariate_normal([10, 100, 70], pcov, N).T

        fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
            absolute_sigma=False
        )
        fit_result = fit.execute()

        cov_equal = fit._cov_mat_equal_lenghts(fit_result.params)
        cov_unequal = fit._cov_mat_unequal_lenghts(fit_result.params)
        np.testing.assert_array_almost_equal(cov_equal, cov_unequal)

        # Try with absolute_sigma=True
        fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
            sigma_a_i=np.sqrt(pcov[0, 0]),
            sigma_b_i=np.sqrt(pcov[1, 1]),
            sigma_c_i=np.sqrt(pcov[2, 2]),
            absolute_sigma=True
        )
        fit_result = fit.execute()

        cov_equal = fit._cov_mat_equal_lenghts(fit_result.params)
        cov_unequal = fit._cov_mat_unequal_lenghts(fit_result.params)
        np.testing.assert_array_almost_equal(cov_equal, cov_unequal)

    def test_error_advanced(self):
        """
        Compare the error propagation of Fit against
        NumericalLeastSquares.
        Models an example from the mathematica docs and try's to replicate it:
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

        a = Parameter(3.0)
        b = Parameter(0.9)
        c = Parameter(5.0)
        x = Variable()
        y = Variable()
        z = Variable()
        model = {z: a * log(b * x + c * y)}

        const_fit = Fit(model, xdata, ydata, zdata, absolute_sigma=False)
        const_result = const_fit.execute()
        fit = Fit(model, xdata, ydata, zdata, absolute_sigma=False, minimizer=MINPACK)
        std_result = fit.execute()

        self.assertEqual(const_fit.absolute_sigma, fit.absolute_sigma)

        self.assertAlmostEqual(const_result.value(a), std_result.value(a), 4)
        self.assertAlmostEqual(const_result.value(b), std_result.value(b), 4)
        self.assertAlmostEqual(const_result.value(c), std_result.value(c), 4)

        self.assertAlmostEqual(const_result.stdev(a), std_result.stdev(a), 4)
        self.assertAlmostEqual(const_result.stdev(b), std_result.stdev(b), 4)
        self.assertAlmostEqual(const_result.stdev(c), std_result.stdev(c), 4)

    def test_gaussian_2d_fitting(self):
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
        sig_x = Parameter(0.2, min=0.0, max=0.3)
        y0 = Parameter(value=mean[1], min=0.0, max=1.0)
        sig_y = Parameter(0.1, min=0.0, max=0.3)
        A = Parameter(value=np.mean(ydata), min=0.0)
        x = Variable()
        y = Variable()
        g = Variable()
        model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
        fit = Fit(model, x=xx, y=yy, g=ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(x0), np.mean(data[:, 0]), 3)
        self.assertAlmostEqual(fit_result.value(y0), np.mean(data[:, 1]), 3)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_x)), np.std(data[:, 0]), 2)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_y)), np.std(data[:, 1]), 2)
        self.assertGreaterEqual(fit_result.r_squared, 0.96)

        # Compare with industry standard MINPACK
        fit_std = Fit(model, x=xx, y=yy, g=ydata, minimizer=MINPACK)
        fit_std_result = fit_std.execute()

        self.assertAlmostEqual(fit_std_result.value(x0), fit_result.value(x0), 4)
        self.assertAlmostEqual(fit_std_result.value(y0), fit_result.value(y0), 4)
        self.assertAlmostEqual(fit_std_result.value(sig_x), fit_result.value(sig_x), 4)
        self.assertAlmostEqual(fit_std_result.value(sig_y), fit_result.value(sig_y), 4)
        self.assertAlmostEqual(fit_std_result.r_squared, fit_result.r_squared, 4)

if __name__ == '__main__':
    unittest.main()
