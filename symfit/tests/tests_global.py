from __future__ import division, print_function
import unittest
import sys
import sympy
import types

from sympy import symbols
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

from symfit import Variable, Parameter, Fit, FitResults, Maximize, Likelihood, log, variables, parameters, Model, NumericalLeastSquares, GlobalLeastSquares
from symfit.distributions import Gaussian, Exp

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


class TestsGlobal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_named_fitting(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter(1.0, min=0, max=10)
        b = Parameter(2.5, min=0, max=10)
        x, y = variables('x, y')


        model = {y: a*x**b}

        fit = GlobalLeastSquares(model, x=xdata, y=ydata)
        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.params.a, 3.0)
        self.assertAlmostEqual(fit_result.params.b, 2.0)

    def test_vector_fitting(self):
        a, b, c = parameters('a, b, c')
        a.min = 5
        a.max = 15
        b.min = 50
        b.max = 150
        c.min = 0
        c.max = 100
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = GlobalLeastSquares(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.params.a, 9.985691, 4)
        self.assertAlmostEqual(fit_result.params.b, 1.006143e+02, 4)
        self.assertAlmostEqual(fit_result.params.c, 7.085713e+01, 5)

    def test_vector_none_fitting(self):
        """
        Fit to a vector model with one var's data set to None
        """
        a, b, c = parameters('a, b, c')
        a.min = 0
        a.max = 10
        b.min = 0
        b.max = 10
        c.min = 0
        c.max = 10
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit_none = GlobalLeastSquares(
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
        fit_none_result = fit_none.execute()
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_none_result.params.a, fit_result.params.a, 4)
        self.assertAlmostEqual(fit_none_result.params.b, fit_result.params.b, 4)
        self.assertAlmostEqual(fit_none_result.params.c, 1.0)


    def test_fitting(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter(min=0, max=5) #3.1, min=2.5, max=3.5
        b = Parameter(min=0, max=5)
        x = Variable()
        new = a*x**b

        fit = GlobalLeastSquares(new, xdata, ydata)

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.params.a, 3.0)
        self.assertAlmostEqual(fit_result.params.b, 2.0)

        self.assertIsInstance(fit_result.r_squared, float)
        self.assertEqual(fit_result.r_squared, 1.0)  # by definition since there's no fuzzyness

        # Test several false ways to access the data.
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_fdska'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'c'])

    def test_2D_fitting(self):
        xdata = np.random.randint(-10, 11, size=(2, 400))
        zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

        a = Parameter(min=0, max=5)
        b = Parameter(min=5, max=10)
        x = Variable()
        y = Variable()
        new = a*x**2 + b*y**2

        fit = GlobalLeastSquares(new, xdata[0], xdata[1], zdata)

        # result = fit.scipy_func(fit.xdata, [2, 3])
        result = fit.model(xdata[0], xdata[1], 2, 3)

        for arg_name, name in zip(('x', 'y', 'a', 'b'), inspect_sig.signature(fit.model).parameters):
            self.assertEqual(arg_name, name)

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertEqual(fit_result.params.a, 2.5)
        self.assertEqual(fit_result.params.b, 7.0)

    def test_gaussian_fitting(self):
        xdata = 2*np.random.rand(10000) - 1 # random betwen [-1, 1]
        ydata = 5.0 * scipy.stats.norm.pdf(xdata, loc=0.0, scale=1.0)

        x0 = Parameter(min=-1, max=1)
        sig = Parameter(min=0, max=10)
        A = Parameter(min=0, max=10)
        x = Variable()
        g = A * Gaussian(x, x0, sig)

        fit = GlobalLeastSquares(g, xdata, ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.params.A, 5.0)
        self.assertAlmostEqual(np.abs(fit_result.params.sig), 1.0)
        self.assertAlmostEqual(fit_result.params.x0, 0.0)
        # raise Exception([i for i in fit_result.params])
        sexy = g(x=2.0, **fit_result.params)
        ugly = g(
            x=2.0,
            x0=fit_result.params.x0,
            A=fit_result.params.A,
            sig=fit_result.params.sig,
        )
        self.assertEqual(sexy, ugly)

    @unittest.skip
    def test_2_gaussian_2d_fitting(self):
        np.random.seed(4242)
        mean = (0.3, 0.3) # x, y mean 0.6, 0.4
        cov = [[0.01**2, 0], [0, 0.01**2]]
        data = np.random.multivariate_normal(mean, cov, 1000000)
        mean = (0.7, 0.7) # x, y mean 0.6, 0.4
        cov = [[0.01**2, 0],[0, 0.01**2]]
        data_2 = np.random.multivariate_normal(mean, cov, 1000000)
        data = np.vstack((data, data_2))

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
        # xdata = np.dstack((xx, yy)).T

        x = Variable()
        y = Variable()

        x0_1 = Parameter(min=0.6, max=0.8)
        sig_x_1 = Parameter(min=0.0, max=0.05)
        y0_1 = Parameter(min=0.6, max=0.8)
        sig_y_1 = Parameter(min=0.0, max=0.05)
        A_1 = Parameter(min=5000, max=15000)
        g_1 = A_1 * Gaussian(x, x0_1, sig_x_1) * Gaussian(y, y0_1, sig_y_1)

        x0_2 = Parameter(min=0.2, max=0.4)
        sig_x_2 = Parameter(min=0.0, max=0.05)
        y0_2 = Parameter(min=0.2, max=0.4)
        sig_y_2 = Parameter(min=0.0, max=0.05)
        A_2 = Parameter(min=5000, max=15000)
        g_2 = A_2 * Gaussian(x, x0_2, sig_x_2) * Gaussian(y, y0_2, sig_y_2)

        model = g_1 + g_2
        fit = GlobalLeastSquares(model, xx, yy, ydata)
        fit_result = fit.execute()

        # Equal up to some precision. Not much obviously.
        self.assertAlmostEqual(fit_result.params.x0_1, 0.7, 3)
        self.assertAlmostEqual(fit_result.params.y0_1, 0.7, 3)
        self.assertAlmostEqual(fit_result.params.x0_2, 0.3, 3)
        self.assertAlmostEqual(fit_result.params.y0_2, 0.3, 3)

    def test_gaussian_2d_fitting(self):
        mean = (0.6, 0.4) # x, y mean 0.6, 0.4
        cov = [[0.2**2, 0], [0, 0.1**2]]

        data = np.random.multivariate_normal(mean, cov, 1000000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

        x0 = Parameter(value=mean[0], min=0, max=1)
        sig_x = Parameter(min=0.0, max=0.3)
        x = Variable()
        y0 = Parameter(value=mean[1], min=0, max=1)
        sig_y = Parameter(min=0.0, max=0.3)
        A = Parameter(min=1, value=100, max=1000)
        y = Variable()
        g = Variable()
#        g = A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)
        model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
        fit = GlobalLeastSquares(model, x=xx, y=yy, g=ydata)
        fit_result = fit.execute()
        
        # Again, the order seems to be swapped for py3k
        self.assertAlmostEqual(fit_result.params.x0, np.mean(data[:, 0]), 1)
#        self.assertAlmostEqual(fit_result.params.x0, mean[0], 1)
        self.assertAlmostEqual(fit_result.params.y0, np.mean(data[:, 1]), 1)
#        self.assertAlmostEqual(fit_result.params.y0, mean[1], 1)
        self.assertAlmostEqual(np.abs(fit_result.params.sig_x), np.std(data[:, 0]), 1)
        self.assertAlmostEqual(np.abs(fit_result.params.sig_y), np.std(data[:, 1]), 1)
        self.assertGreaterEqual(fit_result.r_squared, 0.99)

    def test_meta_parameters(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter(min=0, max=5) #3.1, min=2.5, max=3.5
        b = Parameter(min=0, max=5)
        x = Variable()
        new = a*x**b

        fit = GlobalLeastSquares(new, xdata, ydata)
        try:
            # Stupid settings for (super)fast convergence
            fit.execute(tol=1, popsize=5, strategy='best1bin')
        except Exception as error:
            self.fail('test_meta_parameters raised {}'.format(str(error)))

if __name__ == '__main__':
    unittest.main()
