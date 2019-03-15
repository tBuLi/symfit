from __future__ import division, print_function
import unittest
import pickle
from collections import OrderedDict

import numpy as np
from scipy.optimize import OptimizeResult
from symfit import Variable, Parameter, Fit, FitResults
from symfit.distributions import BivariateGaussian
from symfit.core.minimizers import BaseMinimizer, MINPACK
from symfit.core.objectives import (
    LogLikelihood, LeastSquares, VectorLeastSquares
)

class TestFitResults(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    def setUp(self):
        xdata = np.linspace(1, 10, 10)
        ydata = 3 * xdata ** 2

        a = Parameter('a')
        b = Parameter('b')
        x = Variable('x')
        y = Variable('y')
        new = {y: a * x ** b}
        self.params = [a, b]

        fit = Fit(new, x=xdata, y=ydata)
        self.fit_result = fit.execute()
        fit = Fit(new, x=xdata, y=ydata, minimizer=MINPACK)
        self.minpack_result = fit.execute()
        fit = Fit(new, x=xdata, objective=LogLikelihood)
        self.likelihood_result = fit.execute()

    def test_params_type(self):
        self.assertIsInstance(self.fit_result.params, OrderedDict)

    def test_minimizer_output_type(self):
        self.assertIsInstance(self.fit_result.minimizer_output, dict)
        self.assertIsInstance(self.minpack_result.minimizer_output, dict)
        self.assertIsInstance(self.likelihood_result.minimizer_output, dict)

    def test_fitting(self):
        """
        Test if the fitting worked in the first place.
        """
        a, b = self.params
        fit_result = self.fit_result
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.value(a), 3.0)
        self.assertAlmostEqual(fit_result.value(b), 2.0)

        self.assertIsInstance(fit_result.stdev(a), float)
        self.assertIsInstance(fit_result.stdev(b), float)

        self.assertIsInstance(fit_result.r_squared, float)
        # by definition since there's no fuzzyness
        self.assertEqual(fit_result.r_squared, 1.0)

    def test_fitting_2(self):
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

        self.assertGreater(fit_result.r_squared, 0.95)
        for param in fit.model.params:
            try:
                self.assertAlmostEqual(fit_result.stdev(param)**2 / fit_result.variance(param), 1.0)
            except AssertionError:
                self.assertLessEqual(fit_result.variance(param), 0.0)
                self.assertTrue(np.isnan(fit_result.stdev(param)))

        # Covariance matrix should be symmetric
        for param_1 in fit.model.params:
            for param_2 in fit.model.params:
                self.assertAlmostEqual(fit_result.covariance(param_1, param_2) / fit_result.covariance(param_2, param_1), 1.0, 3)

    def test_minimizer_included(self):
        """"The minimizer used should be included in the results."""
        self.assertIsInstance(self.fit_result.minimizer, BaseMinimizer)

    def test_objective_included(self):
        """"The objective used should be included in the results."""
        self.assertIsInstance(self.fit_result.objective, LeastSquares)
        self.assertIsInstance(self.minpack_result.objective, VectorLeastSquares)
        self.assertIsInstance(self.likelihood_result.objective, LogLikelihood)

    def test_pickle(self):
        dumped = pickle.dumps(self.fit_result)
        new_result = pickle.loads(dumped)
        self.assertEqual(sorted(self.fit_result.__dict__.keys()),
                         sorted(new_result.__dict__.keys()))
        for k, v1 in self.fit_result.__dict__.items():
            v2 = new_result.__dict__[k]
            if k == 'objective' or k == 'minimizer':
                self.assertEqual(type(v1), type(v2))
            elif k != 'minimizer_output':  # Ignore minimizer_output
                print(k)
                if isinstance(v1, np.ndarray):
                    np.testing.assert_almost_equal(v1, v2)
                else:
                    self.assertEqual(v1, v2)

    def test_gof_presence(self):
        """
        Test if the expected goodness of fit estimators are present.
        """
        self.assertTrue(hasattr(self.fit_result, 'objective_value'))
        self.assertTrue(hasattr(self.fit_result, 'r_squared'))
        self.assertTrue(hasattr(self.fit_result, 'chi_squared'))
        self.assertFalse(hasattr(self.fit_result, 'log_likelihood'))
        self.assertFalse(hasattr(self.fit_result, 'likelihood'))

        self.assertTrue(hasattr(self.minpack_result, 'objective_value'))
        self.assertTrue(hasattr(self.minpack_result, 'r_squared'))
        self.assertTrue(hasattr(self.minpack_result, 'chi_squared'))
        self.assertFalse(hasattr(self.minpack_result, 'log_likelihood'))
        self.assertFalse(hasattr(self.minpack_result, 'likelihood'))

        self.assertTrue(hasattr(self.likelihood_result, 'objective_value'))
        self.assertFalse(hasattr(self.likelihood_result, 'r_squared'))
        self.assertFalse(hasattr(self.likelihood_result, 'chi_squared'))
        self.assertTrue(hasattr(self.likelihood_result, 'log_likelihood'))
        self.assertTrue(hasattr(self.likelihood_result, 'likelihood'))

if __name__ == '__main__':
    unittest.main()
