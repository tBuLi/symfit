from __future__ import division, print_function
import unittest
import warnings
import sympy
import types

import numpy as np
from symfit import Variable, Parameter, Fit, FitResults, NumericalLeastSquares, Model
from symfit.distributions import Gaussian


class TestFitResults(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    def test_read_only_results(self):
        """
        Fit results should be read-only. Let's try to break this!
        """
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter(3.0, min=2.75)
        b = Parameter(2.0, max=2.75)
        x = Variable('x')
        new = a*x**b

        fit = Fit(new, xdata, ydata)
        # raise Exception(fit.partial_chi(3, 2), [component(3, 2) for component in fit.partial_chi_jacobian])
        # raise Exception(fit.model.chi_jacobian)
        fit_result = fit.execute()

        # Break it!
        try:
            fit_result.params = 'hello'
        except AttributeError:
            self.assertTrue(True) # desired result
        finally:
            self.assertNotEqual(fit_result.params, 'hello')

        try:
            # Bypass the property getter. This will work, as it set's the instance value of __params.
            fit_result.__params = 'hello'
        except AttributeError:
            self.assertTrue(False) # undesired result
        finally:
            self.assertNotEqual(fit_result.params, 'hello')
            # The assginment will have succeeded on the instance because we set it from the outside.
            # I must admit I don't fully understand why this is allowed and I don't like it.
            # However, the tests below show that it did not influence the class method itself so
            # fitting still works fine.
            # assinging to __params makes *new* instance attribute, the "real"
            # __params instance is called _FitResult__params. See dir(fit_results) and
            # https://www.python.org/dev/peps/pep-0008/#designing-for-inheritance
            self.assertEqual(fit_result.__params, 'hello')

        # Do a second fit and dubble check that we do not overwrtie something crusial.
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(-5, 5, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)
        xdata_coor = np.dstack((xx, yy))

        zdata = 2.5*xx**2 + 3.0*yy**2

        a = Parameter(1., max=2.75)
        b = Parameter(5., min=2.75)
        x = Variable()
        y = Variable()
        new = Variable()
        new_model = Model({new: a*x**2 + b*y**2 })

        fit_2 = Fit(new_model, x=xx, y=yy, new=zdata)
        fit_result_2 = fit_2.execute()
        self.assertNotAlmostEqual(fit_result.value(a), fit_result_2.value(a))
        self.assertAlmostEqual(fit_result.value(a), 3.0)
        self.assertAlmostEqual(fit_result_2.value(a), 2.5)
        self.assertNotAlmostEqual(fit_result.value(b), fit_result_2.value(b))
        self.assertAlmostEqual(fit_result.value(b), 2.0)
        self.assertAlmostEqual(fit_result_2.value(b), 3.0)

    def test_fitting(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter() #3.1, min=2.5, max=3.5
        b = Parameter()
        x = Variable()
        new = a*x**b

        fit = NumericalLeastSquares(new, xdata, ydata)

        self.assertTrue(issubclass(fit.model.chi_squared.__class__, sympy.Expr))
        self.assertTrue(issubclass(fit.model.chi.__class__, sympy.Expr))
        self.assertTrue(type(fit.model.numerical_chi_squared) is types.LambdaType)
        self.assertTrue(type(fit.model.numerical_chi) is types.LambdaType)

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.value(a), 3.0)
        self.assertAlmostEqual(fit_result.value(b), 2.0)

        self.assertIsInstance(fit_result.stdev(a), float)
        self.assertIsInstance(fit_result.stdev(b), float)

        self.assertIsInstance(fit_result.r_squared, float)
        self.assertEqual(fit_result.r_squared, 1.0)  # by definition since there's no fuzzyness

        # Test several illegal ways to access the data.
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_fdska'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'c'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_stdev'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a__stdev'])

    def test_fitting_2(self):
        np.random.seed(4242)
        mean = (0.3, 0.3) # x, y mean 0.6, 0.4
        cov = [
            [0.01**2, 0.4],
            [0.4, 0.01**2]
        ]
        data = np.random.multivariate_normal(mean, cov, 1000000)
        mean = (0.7,0.7) # x, y mean 0.6, 0.4
        cov = [[0.01**2,0],[0,0.01**2]]
        data_2 = np.random.multivariate_normal(mean, cov, 1000000)
        data = np.vstack((data, data_2))

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:,1], data[:,0], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
        # xdata = np.dstack((xx, yy)).T

        x = Variable()
        y = Variable()

        x0_1 = Parameter(0.7, min=0.6, max=0.8)
        sig_x_1 = Parameter(0.1, min=0.0, max=0.2)
        y0_1 = Parameter(0.7, min=0.6, max=0.8)
        sig_y_1 = Parameter(0.1, min=0.0, max=0.2)
        A_1 = Parameter()
        g_1 = A_1 * Gaussian(x, x0_1, sig_x_1) * Gaussian(y, y0_1, sig_y_1)

        x0_2 = Parameter(0.3, min=0.2, max=0.4)
        sig_x_2 = Parameter(0.1, min=0.0, max=0.2)
        y0_2 = Parameter(0.3, min=0.2, max=0.4)
        sig_y_2 = Parameter(0.1, min=0.0, max=0.2)
        A_2 = Parameter()
        g_2 = A_2 * Gaussian(x, x0_2, sig_x_2) * Gaussian(y, y0_2, sig_y_2)

        model = g_1 + g_2
        fit = Fit(model, xx, yy, ydata)
        fit_result = fit.execute()

        for param in fit_result.params:
            self.assertAlmostEqual(fit_result.stdev(param)**2, fit_result.variance(param))
            self.assertEqual(fit_result.stdev(param), fit_result.params.stdev(param))
            self.assertEqual(fit_result.value(param), fit_result.params.value(param))

        # Covariance matrix should be symmetric
        for param_1 in fit_result.params:
            for param_2 in fit_result.params:
                self.assertAlmostEqual(fit_result.covariance(param_1, param_2), fit_result.covariance(param_2, param_1))
#        print(fit_result.params.covariance_matrix)
#        print(fit_result.covariance(x0_1, x0_2))

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger DeprecationWarning
            fit_result.params.get_stdev(x0_1)
            fit_result.params.get_value(x0_1)
            self.assertTrue(len(w) == 2)
            for warning in w:
                self.assertTrue(issubclass(warning.category, DeprecationWarning))


if __name__ == '__main__':
    unittest.main()
