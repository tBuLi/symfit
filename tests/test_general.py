from __future__ import division, print_function
import unittest
import sys
import sympy
import warnings

import numpy as np
import scipy.stats
from scipy.optimize import curve_fit, minimize

from symfit import (
    Variable, Parameter, Fit, FitResults, log, variables,
    parameters, Model, Eq, Ge
)
from symfit.core.minimizers import BFGS, MINPACK, SLSQP, LBFGSB
from symfit.core.objectives import LogLikelihood
from symfit.distributions import Gaussian, Exp

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_callable(self):
        """
        Make sure that symfit expressions are callable (with scalars and
        arrays), and produce the expected results.
        """
        a, b = parameters('a, b')
        x, y = variables('x, y')
        func = a*x**2 + b*y**2
        result = func(x=2, y=3, a=3, b=9)
        self.assertEqual(result, 3*2**2 + 9*3**2)
        result = func(2, 3, a=3, b=9)
        self.assertEqual(result, 3*2**2 + 9*3**2)

        xdata = np.arange(1, 10)
        ydata = np.arange(1, 10)
        result = func(x=ydata, y=ydata, a=3, b=9)
        self.assertTrue(np.array_equal(result, 3*xdata**2 + 9*ydata**2))

    def test_named_fitting(self):
        """
        Make sure that fitting with NumericalLeastSquares works using a dict
        as model and that the resulting fit_result is of the right type.
        """
        xdata = np.linspace(1, 10, 10)
        ydata = 3*xdata**2

        a = Parameter(1.0)
        b = Parameter(2.5)
        x, y = variables('x, y')

        model = {y: a*x**b}

        fit = Fit(model, x=xdata, y=ydata, minimizer=MINPACK)
        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.value(a), 3.0)
        self.assertAlmostEqual(fit_result.value(b), 2.0)

    def test_vector_fitting(self):
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
            minimizer = MINPACK
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a) / 9.985691, 1.0, 5)
        self.assertAlmostEqual(fit_result.value(b) / 1.006143e+02, 1.0, 4)
        self.assertAlmostEqual(fit_result.value(c) / 7.085713e+01, 1.0, 5)

    def test_vector_none_fitting(self):
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

        self.assertAlmostEqual(fit_none_result.value(a), fit_result.value(a), 4)
        self.assertAlmostEqual(fit_none_result.value(b), fit_result.value(b), 4)
        # the parameter without data should be unchanged.
        self.assertAlmostEqual(fit_none_result.value(c), 1.0)

    @unittest.skip('Vector models fail in NumericalLeastSquares with bounds. '
                   'However, this object is no longer used by Fit by default.')
    def test_vector_fitting_bounds_guess(self):
        """
        Tests fitting to a 3 component vector valued function, with bounds and
        guesses.
        """
        a, b, c = parameters('a, b, c')
        a.min = 0
        a.value = 10
        a.max = 25
        b.min = 0
        b.max = 500
        b.value = 100
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = NumericalLeastSquares(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 4)

    @unittest.skip('Vector models fail in NumericalLeastSquares with bounds.'
                   'However, this object is no longer used by Fit by default.')
    def test_vector_fitting_bounds(self):
        """
        Tests fitting to a 3 component vector valued function, with bounds.
        """
        a, b, c = parameters('a, b, c')
        a.min = 0
        a.max = 25
        b.min = 0
        b.max = 500
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = NumericalLeastSquares(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 4)

    def test_vector_fitting_guess(self):
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
            minimizer = MINPACK
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 4)

    def test_fitting(self):
        """
        Tests fitting with NumericalLeastSquares. Makes sure that the resulting
        objects and values are of the right type, and that the fit_result does
        not have unexpected members.
        """
        xdata = np.linspace(1, 10, 10)
        ydata = 3*xdata**2

        a = Parameter()  # 3.1, min=2.5, max=3.5
        b = Parameter()
        x = Variable()
        new = a*x**b

        fit = Fit(new, xdata, ydata, minimizer=MINPACK)

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.value(a), 3.0)
        self.assertAlmostEqual(fit_result.value(b), 2.0)

        self.assertIsInstance(fit_result.stdev(a), float)
        self.assertIsInstance(fit_result.stdev(b), float)

        self.assertIsInstance(fit_result.r_squared, float)
        self.assertEqual(fit_result.r_squared, 1.0)  # by definition since there's no fuzzyness

    # def test_analytical_fitting(self):
    #     """
    #     Tests fitting using AnalyticalFit. Makes sure that the resulting
    #     objects and values are of the right type, and that the fit_result does
    #     not have unexpected members.
    #     """
    #     xdata = np.linspace(1, 10, 10)
    #     ydata = 3*xdata + 2
    #
    #     a = Parameter()
    #     b = Parameter()
    #     x = Variable('x')
    #     new = b*x + a
    #
    #     fit = AnalyticalFit(new, xdata, ydata)
    #     fit_result = fit.execute()
    #     print(fit_result)
    #
    #
    #     self.assertIsInstance(fit_result, FitResults)
    #     self.assertAlmostEqual(fit_result.params.a, 3.0)
    #     self.assertAlmostEqual(fit_result.params.b, 2.0)
    #
    #     self.assertIsInstance(fit_result.params.a_stdev, float)
    #     self.assertIsInstance(fit_result.params.b_stdev, float)
    #
    #     self.assertIsInstance(fit_result.r_squared, float)
    #
    #     # Test several false ways to access the data.
    #     self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_fdska'])
    #     self.assertRaises(AttributeError, getattr, *[fit_result.params, 'c'])
    #     self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_stdev'])
    #     self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_'])
    #     self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a__stdev'])

    def test_grid_fitting(self):
        """
        Tests fitting a scalar function with 2 independent variables.
        """
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(5, 15, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter(2.5, max=2.75)
        b = Parameter(3.0, min=2.75)
        x = Variable()
        y = Variable()
        z = Variable()
        new = {z: a*x**2 + b*y**2}

        fit = Fit(new, x=xx, y=yy, z=zdata)
        results = fit.execute()

        self.assertIsInstance(fit.minimizer, LBFGSB)

        self.assertAlmostEqual(results.value(a), 2.5)
        self.assertAlmostEqual(results.value(b), 3.)

    # TODO: Should be 3 tests?
    def test_model_callable(self):
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
        self.assertIsInstance(ans, tuple)
        z, = ans

        self.assertEqual(z, 36)
        for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
            self.assertEqual(arg_name, name)

        # From Model __init__ directly
        model = Model([
            a*x**2,
            4*b*y**2,
            a*x**2 + b*y**2
        ])
        z_1, z_2, z_3 = model(3, 3, 2, 2)

        self.assertEqual(z_1, 18)
        self.assertEqual(z_2, 72)
        self.assertEqual(z_3, 36)
        for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
            self.assertEqual(arg_name, name)

        # From dict
        z_1, z_2, z_3 = variables('z_1, z_2, z_3')
        model = Model({
            z_1: a*x**2,
            z_2: 4*b*y**2,
            z_3: a*x**2 + b*y**2
        })
        z_1, z_2, z_3 = model(3, 3, 2, 2)

        self.assertEqual(z_1, 18)
        self.assertEqual(z_2, 72)
        self.assertEqual(z_3, 36)
        for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(model).parameters):
            self.assertEqual(arg_name, name)

    def test_2D_fitting(self):
        """
        Makes sure that a scalar model with 2 independent variables has the
        proper signature, and that the fit result is of the correct type.
        """
        xdata = np.random.randint(-10, 11, size=(2, 400))
        zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

        a = Parameter()
        b = Parameter()
        x = Variable()
        y = Variable()
        new = a*x**2 + b*y**2

        fit = Fit(new, xdata[0], xdata[1], zdata)

        result = fit.model(xdata[0], xdata[1], 2, 3)
        self.assertIsInstance(result, tuple)

        for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(fit.model).parameters):
            self.assertEqual(arg_name, name)

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)

    def test_gaussian_fitting(self):
        """
        Tests fitting to a gaussian function and fit_result.params unpacking.
        """
        xdata = 2*np.random.rand(10000) - 1  # random betwen [-1, 1]
        ydata = 5.0 * scipy.stats.norm.pdf(xdata, loc=0.0, scale=1.0)

        x0 = Parameter()
        sig = Parameter()
        A = Parameter()
        x = Variable()
        g = A * Gaussian(x, x0, sig)

        fit = Fit(g, xdata, ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(A), 5.0)
        self.assertAlmostEqual(np.abs(fit_result.value(sig)), 1.0)
        self.assertAlmostEqual(fit_result.value(x0), 0.0)
        # raise Exception([i for i in fit_result.params])
        sexy = g(x=2.0, **fit_result.params)
        ugly = g(
            x=2.0,
            x0=fit_result.value(x0),
            A=fit_result.value(A),
            sig=fit_result.value(sig),
        )
        self.assertEqual(sexy, ugly)

    def test_2_gaussian_2d_fitting(self):
        """
        Tests fitting to a scalar gaussian with 2 independent variables with
        tight bounds.
        """
        mean = (0.3, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.01**2, 0], [0, 0.01**2]]
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

        x = Variable()
        y = Variable()

        x0_1 = Parameter(0.7, min=0.6, max=0.9)
        sig_x_1 = Parameter(0.1, min=0.0, max=0.2)
        y0_1 = Parameter(0.8, min=0.6, max=0.9)
        sig_y_1 = Parameter(0.1, min=0.0, max=0.2)
        A_1 = Parameter()
        g_1 = A_1 * Gaussian(x, x0_1, sig_x_1) * Gaussian(y, y0_1, sig_y_1)

        x0_2 = Parameter(0.3, min=0.2, max=0.5)
        sig_x_2 = Parameter(0.1, min=0.0, max=0.2)
        y0_2 = Parameter(0.4, min=0.2, max=0.5)
        sig_y_2 = Parameter(0.1, min=0.0, max=0.2)
        A_2 = Parameter()
        g_2 = A_2 * Gaussian(x, x0_2, sig_x_2) * Gaussian(y, y0_2, sig_y_2)

        model = g_1 + g_2
        fit = Fit(model, xx, yy, ydata)
        fit_result = fit.execute()

        self.assertIsInstance(fit.minimizer, LBFGSB)

        img = model(x=xx, y=yy, **fit_result.params)
        img_g_1 = g_1(x=xx, y=yy, **fit_result.params)
        img_g_2 = g_2(x=xx, y=yy, **fit_result.params)
        np.testing.assert_array_equal(img, img_g_1 + img_g_2)

        # Equal up to some precision. Not much obviously.
        self.assertAlmostEqual(fit_result.value(x0_1), 0.7, 3)
        self.assertAlmostEqual(fit_result.value(y0_1), 0.8, 3)
        self.assertAlmostEqual(fit_result.value(x0_2), 0.3, 3)
        self.assertAlmostEqual(fit_result.value(y0_2), 0.4, 3)

    def test_gaussian_2d_fitting(self):
        """
        Tests fitting to a scalar gaussian function with 2 independent
        variables.
        """
        mean = (0.6, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.2**2, 0], [0, 0.1**2]]

        data = np.random.multivariate_normal(mean, cov, 1000000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100,
                                               range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

        x0 = Parameter(value=mean[0])
        sig_x = Parameter(min=0.0)
        x = Variable()
        y0 = Parameter(value=mean[1])
        sig_y = Parameter(min=0.0)
        A = Parameter(min=1, value=100)
        y = Variable()
        g = Variable()
#        g = A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)
        model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
        fit = Fit(model, x=xx, y=yy, g=ydata, minimizer=MINPACK)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(x0), np.mean(data[:, 0]), 1)
        self.assertAlmostEqual(fit_result.value(y0), np.mean(data[:, 1]), 1)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_x)), np.std(data[:, 0]), 1)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_y)), np.std(data[:, 1]), 1)
        self.assertGreaterEqual(fit_result.r_squared, 0.99)

    def test_jacobian_matrix(self):
        """
        The jacobian matrix of a model should be a 2D list (matrix) containing
        all the partial derivatives.
        """
        a, b, c = parameters('a, b, c')
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = Model({a_i: 2 * a + 3 * b, b_i: 5 * b, c_i: 7 * c})
        self.assertEqual([[2, 3, 0], [0, 5, 0], [0, 0, 7]], model.jacobian)

    def test_likelihood_fitting_exponential(self):
        """
        Fit using the likelihood method.
        """
        b = Parameter(4, min=3.0)
        x, y = variables('x, y')
        pdf = {y: Exp(x, 1/b)}

        # Draw points from an Exp(5) exponential distribution.
        np.random.seed(100)
        xdata = np.random.exponential(5, 1000000)

        # Expected parameter values
        mean = np.mean(xdata)
        stdev = np.std(xdata)
        mean_stdev = stdev / np.sqrt(len(xdata))

        with self.assertRaises(NotImplementedError):
            fit = Fit(pdf, x=xdata, sigma_y=2.0, objective=LogLikelihood)
        fit = Fit(pdf, xdata, objective=LogLikelihood)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(b) / mean, 1, 3)
        self.assertAlmostEqual(fit_result.value(b) / stdev, 1, 3)
        self.assertAlmostEqual(fit_result.stdev(b) / mean_stdev, 1, 3)

    def test_likelihood_fitting_gaussian(self):
        """
        Fit using the likelihood method.
        """
        mu, sig = parameters('mu, sig')
        sig.min = 0.01
        sig.value = 3.0
        mu.value = 50.
        x = Variable()
        pdf = Gaussian(x, mu, sig)

        np.random.seed(10)
        xdata = np.random.normal(51., 3.5, 10000)

        # Expected parameter values
        mean = np.mean(xdata)
        stdev = np.std(xdata)
        mean_stdev = stdev/np.sqrt(len(xdata))

        fit = Fit(pdf, xdata, objective=LogLikelihood)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(mu) / mean, 1, 6)
        self.assertAlmostEqual(fit_result.stdev(mu) / mean_stdev, 1, 3)
        self.assertAlmostEqual(fit_result.value(sig) / np.std(xdata), 1, 6)

    def test_parameter_add(self):
        """
        Makes sure the __add__ method of Parameters behaves as expected.
        """
        a = Parameter(value=1.0, min=0.5, max=1.5)
        b = Parameter(1.0, min=0.0)
        new = a + b
        self.assertIsInstance(new, sympy.Add)

    def test_argument_name(self):
        """
        Make sure that Parameters have a name attribute with the expected
        value.
        """
        a = Parameter()
        b = Parameter(name='b')
        c = Parameter(name='d')
        self.assertEqual(a.name, 'a')
        self.assertEqual(b.name, 'b')
        self.assertEqual(c.name, 'd')

    def test_symbol_add(self):
        """
        Makes sure the __add__ method of symbols behaves as expected.
        """
        x, y = sympy.symbols('x y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

    def test_evaluate_model(self):
        """
        Makes sure that models are callable and give the expected answer.
        """
        A = Parameter()
        x = Variable()
        new = A * x ** 2

        self.assertEqual(new(x=2, A=2), 8)
        self.assertNotEqual(new(x=2, A=3), 8)

    # TODO: Do we really need to test this?
    def test_symbol_object_add(self):
        """
        Makes sure the __add__ method of sympy's Symbol behaves as expected.
        """
        from sympy.core.symbol import Symbol
        x = Symbol('x')
        y = Symbol('y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

    def test_simple_sigma(self):
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
        y = Variable()
        g = Parameter()
        t_model = (2 * y / g)**0.5

        fit = Fit(t_model, y_data, t_data)  # , sigma=sigma_t)
        fit_result = fit.execute()

        # h_smooth = np.linspace(0,60,100)
        # t_smooth = t_model(y=h_smooth, **fit_result.params)

        # Lets with the results from curve_fit, no weights
        popt_noweights, pcov_noweights = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data)

        self.assertAlmostEqual(fit_result.value(g), popt_noweights[0])
        self.assertAlmostEqual(fit_result.stdev(g), np.sqrt(pcov_noweights[0, 0]), 6)

        # Same sigma everywere
        fit = Fit(t_model, y_data, t_data, 0.0031, absolute_sigma=False)
        fit_result = fit.execute()
        popt_sameweights, pcov_sameweights = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data, sigma=0.0031*np.ones(len(y_data)), absolute_sigma=False)
        self.assertAlmostEqual(fit_result.value(g), popt_sameweights[0], 4)
        self.assertAlmostEqual(fit_result.stdev(g), np.sqrt(pcov_sameweights[0, 0]), 4)
        # Same weight everywere should be the same as no weight when absolute_sigma=False
        self.assertAlmostEqual(fit_result.value(g), popt_noweights[0], 4)
        self.assertAlmostEqual(fit_result.stdev(g), np.sqrt(pcov_noweights[0, 0]), 4)

        # Different sigma for every point
        fit = Fit(t_model, y_data, t_data, 0.1*sigma_t, absolute_sigma=False)
        fit_result = fit.execute()
        popt, pcov = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data, sigma=.1*sigma_t)

        self.assertAlmostEqual(fit_result.value(g), popt[0])
        self.assertAlmostEqual(fit_result.stdev(g), np.sqrt(pcov[0, 0]), 6)

        # according to Mathematica
        self.assertAlmostEqual(fit_result.value(g), 9.095, 3)
        self.assertAlmostEqual(fit_result.stdev(g), 0.102, 3)

    def test_error_advanced(self):
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
        # z = np.array(z)
        errors = np.array([.4, .4, .2, .4, .1, .3, .1, .2, .2, .2])

        # raise Exception(xy, z)
        a = Parameter(3.0)
        b = Parameter(0.9)
        c = Parameter(5)
        x = Variable()
        y = Variable()
        z = Variable()
        model = {z: a * log(b * x + c * y)}

        # fit = Fit(model, xy, z, absolute_sigma=False)
        fit = Fit(model, xdata, ydata, zdata, absolute_sigma=False)
        # fit = Fit(model, x=xdata, y=ydata, z=zdata, absolute_sigma=False)
        fit_result = fit.execute()

        # Same as Mathematica default behavior.
        self.assertAlmostEqual(fit_result.value(a), 2.9956, 4)
        self.assertAlmostEqual(fit_result.value(b), 0.563212, 4)
        self.assertAlmostEqual(fit_result.value(c), 3.59732, 4)
        self.assertAlmostEqual(fit_result.stdev(a), 0.278304, 4)
        self.assertAlmostEqual(fit_result.stdev(b), 0.224107, 4)
        self.assertAlmostEqual(fit_result.stdev(c), 0.980352, 4)

        fit = Fit(model, xdata, ydata, zdata, absolute_sigma=True)
        fit_result = fit.execute()
        # Same as Mathematica in Measurement error mode, but without suplying
        # any errors.
        self.assertAlmostEqual(fit_result.value(a), 2.9956, 4)
        self.assertAlmostEqual(fit_result.value(b), 0.563212, 4)
        self.assertAlmostEqual(fit_result.value(c), 3.59732, 4)
        self.assertAlmostEqual(fit_result.stdev(a), 0.643259, 4)
        self.assertAlmostEqual(fit_result.stdev(b), 0.517992, 4)
        self.assertAlmostEqual(fit_result.stdev(c), 2.26594, 4)

        fit = Fit(model, xdata, ydata, zdata, sigma_z=errors)
        fit_result = fit.execute()

        popt, pcov, infodict, errmsg, ier = curve_fit(lambda x_vec, a, b, c: a * np.log(b * x_vec[0] + c * x_vec[1]), xy, zdata, sigma=errors, absolute_sigma=True, full_output=True)

        # Same as curve_fit?
        self.assertAlmostEqual(fit_result.value(a), popt[0], 4)
        self.assertAlmostEqual(fit_result.value(b), popt[1], 4)
        self.assertAlmostEqual(fit_result.value(c), popt[2], 4)
        self.assertAlmostEqual(fit_result.stdev(a), np.sqrt(pcov[0,0]), 4)
        self.assertAlmostEqual(fit_result.stdev(b), np.sqrt(pcov[1,1]), 4)
        self.assertAlmostEqual(fit_result.stdev(c), np.sqrt(pcov[2,2]), 4)

        # Same as Mathematica with MEASUREMENT ERROR
        self.assertAlmostEqual(fit_result.value(a), 2.68807, 4)
        self.assertAlmostEqual(fit_result.value(b), 0.941344, 4)
        self.assertAlmostEqual(fit_result.value(c), 5.01541, 4)
        self.assertAlmostEqual(fit_result.stdev(a), 0.0974628, 4)
        self.assertAlmostEqual(fit_result.stdev(b), 0.247018, 4)
        self.assertAlmostEqual(fit_result.stdev(c), 0.597661, 4)

    def test_error_analytical(self):
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

        a = Parameter()
        y = Variable()
        model = {y: a}

        fit = Fit(model, y=yn, sigma_y=sigma)
        fit_result = fit.execute()

        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn, sigma=sigma, absolute_sigma=True)
        self.assertAlmostEqual(fit_result.value(a), popt[0], 5)
        self.assertAlmostEqual(fit_result.stdev(a), np.sqrt(np.diag(pcov))[0], 2)

        fit_no_sigma = Fit(model, yn)
        fit_result_no_sigma = fit_no_sigma.execute()

        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn,)
        # With or without sigma, the bestfit params should be in agreement in case of equal weights
        self.assertAlmostEqual(fit_result.value(a), fit_result_no_sigma.value(a), 5)
        # Since symfit is all about absolute errors, the sigma will not be in agreement
        self.assertNotEqual(fit_result.stdev(a), fit_result_no_sigma.stdev(a), 5)
        self.assertAlmostEqual(fit_result_no_sigma.value(a), popt[0], 5)
        self.assertAlmostEqual(fit_result_no_sigma.stdev(a), pcov[0][0]**0.5, 5)

        # Analytical answer for mean of N(0,1):
        mu = 0.0
        sigma_mu = sigma[0]/N**0.5

        self.assertAlmostEqual(fit_result.stdev(a), sigma_mu, 5)

    # TODO: redudant with test_error_analytical?
    # def test_straight_line_analytical(self):
    #     """
    #     Test symfit against a straight line, for which the parameters and their
    #     uncertainties are known analytically. Assuming equal weights.
    #     """
    #     data = [[0, 1], [1, 0], [3, 2], [5, 4]]
    #     x, y = (np.array(i, dtype='float64') for i in zip(*data))
    #     # x = np.arange(0, 100, 0.1)
    #     # np.random.seed(10)
    #     # y = 3.0*x + 105.0 + np.random.normal(size=x.shape)
    #
    #     dx = x - x.mean()
    #     dy = y - y.mean()
    #     mean_squared_x = np.mean(x**2) - np.mean(x)**2
    #     mean_xy = np.mean(x * y) - np.mean(x)*np.mean(y)
    #     a = mean_xy/mean_squared_x
    #     b = y.mean() - a * x.mean()
    #     self.assertAlmostEqual(a, 0.694915, 6) # values from Mathematica
    #     self.assertAlmostEqual(b, 0.186441, 6)
    #     print(a, b)
    #
    #     S = np.sum((y - (a*x + b))**2)
    #     var_a_exact = S/(len(x) * (len(x) - 2) * mean_squared_x)
    #     var_b_exact = var_a_exact*np.mean(x ** 2)
    #     a_exact = a
    #     b_exact = b
    #
    #     # We will now compare these exact results with values from symfit
    #     a, b, x_var = Parameter(name='a', value=3.0), Parameter(name='b'), Variable(name='x')
    #     model = a*x_var + b
    #     fit = Fit(model, x, y, absolute_sigma=False)
    #     fit_result = fit.execute()
    #
    #     popt, pcov = curve_fit(lambda z, c, d: c * z + d, x, y,
    #                            Dfun=lambda p, x, y, func: np.transpose([x, np.ones_like(x)]))
    #                             # Dfun=lambda p, x, y, func: print(p, func, x, y))
    #
    #     # curve_fit
    #     self.assertAlmostEqual(a_exact, popt[0], 4)
    #     self.assertAlmostEqual(b_exact, popt[1], 4)
    #     self.assertAlmostEqual(var_a_exact, pcov[0][0], 6)
    #     self.assertAlmostEqual(var_b_exact, pcov[1][1], 6)
    #
    #     self.assertAlmostEqual(a_exact, fit_result.params.a, 4)
    #     self.assertAlmostEqual(b_exact, fit_result.params.b, 4)
    #     self.assertAlmostEqual(var_a_exact**0.5, fit_result.params.a_stdev, 6)
    #     self.assertAlmostEqual(var_b_exact**0.5, fit_result.params.b_stdev, 6)

    def test_model_from_dict(self):
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
            self.fail('test_model_from_dict raised {}'.format(error))


if __name__ == '__main__':
    try:
        unittest.main(warnings='ignore')
        # Note that unittest will catch and handle exceptions raised by tests.
        # So this line will *only* deal with exceptions raised by the line
        # above.
    except TypeError:
        # In Py2, unittest.main doesn't take a warnings argument
        warnings.simplefilter('ignore')
        unittest.main()
