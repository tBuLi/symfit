from __future__ import division, print_function
import unittest
import inspect
import sympy
from sympy import symbols
import numpy as np
from symfit.api import Variable, Parameter, Fit, FitResults, Maximize, Minimize, exp, Likelihood, ln, log, variables, parameters
from symfit.functions import Gaussian, Exp
import scipy.stats
from scipy.optimize import curve_fit
from symfit.core.support import sympy_to_scipy, sympy_to_py
import matplotlib.pyplot as plt
import seaborn

class TddInPythonExample(unittest.TestCase):
    def test_gaussian(self):
        x0, sig = parameters('x0, sig')
        x = Variable()

        new = sympy.exp(-(x - x0)**2/(2*sig**2))
        self.assertIsInstance(new, sympy.exp)
        g = Gaussian(x, x0, sig)
        self.assertTrue(issubclass(g.__class__, sympy.exp))

    def test_callable(self):
        a, b = parameters('a, b')
        x, y = variables('x, y')
        func = a*x**2 + b*y**2
        result = func(x=2, y=3, a=3, b=9)
        self.assertEqual(result, 3*2**2 + 9*3**2)

        xdata = np.arange(1,10)
        ydata = np.arange(1,10)
        result = func(x=ydata, y=ydata, a=3, b=9)
        self.assertTrue(np.array_equal(result, 3*xdata**2 + 9*ydata**2))

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
        fit_result = fit.execute()

        # Break it!
        try:
            fit_result.params = 'hello'
        except AttributeError:
            self.assertTrue(True) # desired result
        else:
            self.assertNotEqual(fit_result.params, 'hello')

        try:
            # Bypass the property getter. This will work, as it set's the instance value of __params.
            fit_result.__params = 'hello'
        except AttributeError as foo:
            self.assertTrue(False) # undesired result
        else:
            self.assertNotEqual(fit_result.params, 'hello')
            # The assginment will have succeeded on the instance because we set it from the outside.
            # I must admit I don't fully understand why this is allowed and I don't like it.
            # However, the tests below show that it did not influence the class method itself so
            # fitting still works fine.
            self.assertEqual(fit_result.__params, 'hello')

        # Do a second fit and dubble check that we do not overwrtie something crusial.
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(-5, 5, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)
        xdata_coor = np.dstack((xx, yy))

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter(2.5, max=2.75)
        b = Parameter(3.0, min=2.75)
        x = Variable()
        y = Variable()
        new = (a*x**2 + b*y**2)

        fit_2 = Fit(new, xdata_coor, zdata)
        fit_result_2 = fit_2.execute()
        self.assertNotAlmostEqual(fit_result.params.a, fit_result_2.params.a)
        self.assertAlmostEqual(fit_result.params.a, 3.0)
        self.assertAlmostEqual(fit_result_2.params.a, 2.5)
        self.assertNotAlmostEqual(fit_result.params.b, fit_result_2.params.b)
        self.assertAlmostEqual(fit_result.params.b, 2.0)
        self.assertAlmostEqual(fit_result_2.params.b, 3.0)

    def test_fitting(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter(3.0)
        b = Parameter(2.0)
        x = Variable('x')
        new = a*x**b

        fit = Fit(new, xdata, ydata)

        func = sympy_to_py(new, [x], [a, b])
        result = func(xdata, 3, 2)
        self.assertTrue(np.array_equal(result, ydata))

        result = fit.scipy_func(fit.xdata, [3, 2])
        self.assertTrue(np.array_equal(result, ydata))

        args, varargs, keywords, defaults = inspect.getargspec(func)

        # self.assertEqual(args, ['x', 'a', 'b'])
        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        self.assertAlmostEqual(fit_result.params.a, 3.0)
        self.assertAlmostEqual(fit_result.params.b, 2.0)

        self.assertIsInstance(fit_result.params.a_stdev, float)
        self.assertIsInstance(fit_result.params.b_stdev, float)

        self.assertIsInstance(fit_result.r_squared, float)

        # Test several false ways to access the data.
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_fdska'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'c'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_stdev'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a__stdev'])

    def test_numpy_functions(self):
        xdata = np.linspace(1,10,10)
        ydata = 45*np.log(xdata*2)

        a = Parameter()
        b = Parameter(value=2.1, fixed=True)
        x = Variable()
        new = a*sympy.log(x*b)


    def test_grid_fitting(self):
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(-5, 5, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)
        xdata_coor = np.dstack((xx, yy))

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter(2.5, max=2.75)
        b = Parameter(3.0, min=2.75)
        x = Variable()
        y = Variable()
        new = (a*x**2 + b*y**2)

        fit = Fit(new, xdata_coor, zdata)

        # Test the flatten function for consistency.
        xdata_coor_flat, zdata_flat = fit._flatten(xdata_coor, zdata)
        # _flatten transposes such arrays because the variables are in the deepest dimension instead of the first.
        # This is normally not a problem because all we want from the fit is the correct parameters.
        self.assertFalse(np.array_equal(zdata, zdata_flat.reshape((10,10))))
        self.assertTrue(np.array_equal(zdata, zdata_flat.reshape((10,10)).T))
        self.assertFalse(np.array_equal(xdata_coor, xdata_coor_flat.reshape((10,10,2))))
        new_xdata = xdata_coor_flat.reshape((2,10,10)).T
        self.assertTrue(np.array_equal(xdata_coor, new_xdata))


        results = fit.execute()
        self.assertAlmostEqual(results.params.a, 2.5)
        self.assertAlmostEqual(results.params.b, 3.)


    def test_2D_fitting(self):
        xdata = np.random.randint(-10, 11, size=(2, 400))
        zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

        a = Parameter()
        b = Parameter()
        x = Variable()
        y = Variable()
        new = a*x**2 + b*y**2

        fit = Fit(new, xdata, zdata)

        result = fit.scipy_func(fit.xdata, [2, 3])

        import inspect
        args, varargs, keywords, defaults = inspect.getargspec(fit.scipy_func)
        self.assertEqual(args, ['x', 'p'])

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)

    def test_gaussian_fitting(self):
        xdata = 2*np.random.rand(10000) - 1 # random betwen [-1, 1]
        ydata = scipy.stats.norm.pdf(xdata, loc=0.0, scale=1.0)

        x0 = Parameter()
        sig = Parameter()
        A = Parameter()
        x = Variable()
        g = A * Gaussian(x, x0, sig)

        fit = Fit(g, xdata, ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.params.A, 0.3989423)
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

    def test_2_gaussian_2d_fitting(self):
        np.random.seed(4242)
        mean = (0.3, 0.3) # x, y mean 0.6, 0.4
        cov = [[0.01**2,0],[0,0.01**2]]
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
        xdata = np.dstack((xx, yy)).T

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
        fit = Fit(model, xdata, ydata)
        fit_result = fit.execute()

        img = model(x=xx, y=yy, **fit_result.params)
        img_g_1 = g_1(x=xx, y=yy, **fit_result.params)

        # Equal up to some precision. Not much obviously.
        self.assertAlmostEqual(fit_result.params.x0_1, 0.7, 2)
        self.assertAlmostEqual(fit_result.params.y0_1, 0.7, 2)
        self.assertAlmostEqual(fit_result.params.x0_2, 0.3, 2)
        self.assertAlmostEqual(fit_result.params.y0_2, 0.3, 2)

    def test_gaussian_2d_fitting(self):
        mean = (0.6,0.4) # x, y mean 0.6, 0.4
        cov = [[0.2**2,0],[0,0.1**2]]
        data = np.random.multivariate_normal(mean, cov, 1000000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
        xdata = np.dstack((xx, yy)).T # T because np fucks up conventions.

        x0 = Parameter(0.6)
        sig_x = Parameter(0.2, min=0.0)
        x = Variable()
        y0 = Parameter(0.4)
        sig_y = Parameter(0.1, min=0.0)
        A = Parameter()
        y = Variable()
        g = A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)

        fit = Fit(g, xdata, ydata)
        fit_result = fit.execute()

        # Again, the order seems to be swapped for py3k
        self.assertAlmostEqual(fit_result.params.x0, np.mean(data[:,0]), 3)
        self.assertAlmostEqual(fit_result.params.y0, np.mean(data[:,1]), 3)
        self.assertAlmostEqual(np.abs(fit_result.params.sig_x), np.std(data[:,0]), 3)
        self.assertAlmostEqual(np.abs(fit_result.params.sig_y), np.std(data[:,1]), 3)
        self.assertGreaterEqual(fit_result.r_squared, 0.99)

    def test_minimize(self):
        x = Parameter(-1.)
        y = Parameter()
        model = 2*x*y + 2*x - x**2 - 2*y**2
        from sympy import Eq, Ge
        constraints = [
            Ge(y - 1, 0),  #y - 1 >= 0,
            Eq(x**3 - y, 0),  # x**3 - y == 0,
        ]

        # raise Exception(model.atoms(), model.as_ordered_terms())
        # self.assertIsInstance(constraints[0], Eq)

        # Unbounded
        fit = Maximize(model)
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.params.y, 1.)
        self.assertAlmostEqual(fit_result.params.x, 2.)

        fit = Maximize(model, constraints=constraints)
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.params.x, 1.00000009)
        self.assertAlmostEqual(fit_result.params.y, 1.)

    def test_scipy_style(self):
        def func(x, sign=1.0):
            """ Objective function """
            return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

        def func_deriv(x, sign=1.0):
            """ Derivative of objective function """
            dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
            dfdx1 = sign*(2*x[0] - 4*x[1])
            return np.array([ dfdx0, dfdx1 ])

        cons = (
            {'type': 'eq',
             'fun' : lambda x: np.array([x[0]**3 - x[1]]),
             'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
            {'type': 'ineq',
             'fun' : lambda x: np.array([x[1] - 1]),
             'jac' : lambda x: np.array([0.0, 1.0])})

        from scipy.optimize import minimize
        res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
               method='SLSQP', options={'disp': True})

        res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
               constraints=cons, method='SLSQP', options={'disp': True})

    def test_likelihood_fitting(self):
        """
        Fit using the likelihood method.
        """
        b = Parameter(4, min=3.0)
        x = Variable()
        pdf = (1/b) * exp(- x / b)

        # Draw 100 points from an exponential distribution.
        # np.random.seed(100)
        xdata = np.random.exponential(5, 100000)

        fit = Likelihood(pdf, xdata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.params.b, 5., 1)


    def test_parameter_add(self):
        a = Parameter(value=1.0, min=0.5, max=1.5)
        b = Parameter(1.0, min=0.0)
        new = a + b
        self.assertIsInstance(new, sympy.Add)

    def test_argument_name(self):
        a = Parameter()
        b = Parameter(name='b')
        c = Parameter(name='d')
        self.assertEqual(a.name, 'a')
        self.assertEqual(b.name, 'b')
        self.assertEqual(c.name, 'd')

    def test_symbol_add(self):
        x, y = symbols('x y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

    def test_evaluate_model(self):
        A = Parameter()
        x = Variable()
        new = A * x ** 2

        self.assertEqual(new(x=2, A=2), 8)
        self.assertNotEqual(new(x=2, A=3), 8)

    def test_symbol_object_add(self):
        from sympy.core.symbol import Symbol
        x = Symbol('x')
        y = Symbol('y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

    def test_simple_sigma(self):
        from symfit.api import Variable, Parameter, Fit

        t_data = np.array([1.4, 2.1, 2.6, 3.0, 3.3])
        y_data = np.array([10, 20, 30, 40, 50])

        sigma = 0.2
        n = np.array([5, 3, 8, 15, 30])
        sigma_t = sigma / np.sqrt(n)

        # We now define our model
        y = Variable()
        g = Parameter()
        t_model = (2 * y / g)**0.5

        fit = Fit(t_model, y_data, t_data)#, sigma=sigma_t)
        fit_result = fit.execute()

        # h_smooth = np.linspace(0,60,100)
        # t_smooth = t_model(y=h_smooth, **fit_result.params)

        # Lets with the results from curve_fit, no weights
        popt_noweights, pcov_noweights = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data)

        self.assertAlmostEqual(fit_result.params.g, popt_noweights[0])
        self.assertAlmostEqual(fit_result.params.g_stdev, np.sqrt(pcov_noweights[0, 0]))

        # Same sigma everywere
        fit = Fit(t_model, y_data, t_data, sigma=0.0031, absolute_sigma=False)
        fit_result = fit.execute()
        popt_sameweights, pcov_sameweights = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data, sigma=0.0031, absolute_sigma=False)
        self.assertAlmostEqual(fit_result.params.g, popt_sameweights[0], 4)
        self.assertAlmostEqual(fit_result.params.g_stdev, np.sqrt(pcov_sameweights[0, 0]), 4)
        # Same weight everywere should be the same as no weight.
        self.assertAlmostEqual(fit_result.params.g, popt_noweights[0], 4)
        self.assertAlmostEqual(fit_result.params.g_stdev, np.sqrt(pcov_noweights[0, 0]), 4)

        # Different sigma for every point
        fit = Fit(t_model, y_data, t_data, sigma=0.1*sigma_t, absolute_sigma=False)
        fit_result = fit.execute()
        popt, pcov = curve_fit(lambda y, p: (2 * y / p)**0.5, y_data, t_data, sigma=.1*sigma_t)

        self.assertAlmostEqual(fit_result.params.g, popt[0])
        self.assertAlmostEqual(fit_result.params.g_stdev, np.sqrt(pcov[0, 0]))

        self.assertAlmostEqual(fit_result.params.g, 9.095, 3)
        self.assertAlmostEqual(fit_result.params.g_stdev, 0.102, 3) # according to Mathematica

    def test_error_advanced(self):
        """
        Models an example from the mathematica docs and try's to replicate it:
        http://reference.wolfram.com/language/howto/FitModelsWithMeasurementErrors.html
        """
        data = [
            [0.9, 6.1, 9.5], [3.9, 6., 9.7], [0.3, 2.8, 6.6],
            [1., 2.2, 5.9], [1.8, 2.4, 7.2], [9., 1.7, 7.],
            [7.9, 8., 10.4], [4.9, 3.9, 9.], [2.3, 2.6, 7.4],
            [4.7, 8.4, 10.]
        ]
        x, y, z = zip(*data)
        xy = np.vstack((x, y))
        z = np.array(z)
        errors = np.array([.4, .4, .2, .4, .1, .3, .1, .2, .2, .2])

        # raise Exception(xy, z)
        a = Parameter()
        b = Parameter(0.9)
        c = Parameter(5)
        x = Variable()
        y = Variable()
        model = a * log(b * x + c * y)

        fit = Fit(model, xy, z, absolute_sigma=False)
        fit_result = fit.execute()
        print(fit_result)

        # Same as Mathematica default behavior.
        self.assertAlmostEqual(fit_result.params.a, 2.9956, 4)
        self.assertAlmostEqual(fit_result.params.b, 0.563212, 4)
        self.assertAlmostEqual(fit_result.params.c, 3.59732, 4)
        self.assertAlmostEqual(fit_result.params.a_stdev, 0.278304, 4)
        self.assertAlmostEqual(fit_result.params.b_stdev, 0.224107, 4)
        self.assertAlmostEqual(fit_result.params.c_stdev, 0.980352, 4)

        fit = Fit(model, xy, z, absolute_sigma=True)
        fit_result = fit.execute()
        # Same as Mathematica in Measurement error mode, but without suplying
        # any errors.
        self.assertAlmostEqual(fit_result.params.a, 2.9956, 4)
        self.assertAlmostEqual(fit_result.params.b, 0.563212, 4)
        self.assertAlmostEqual(fit_result.params.c, 3.59732, 4)
        self.assertAlmostEqual(fit_result.params.a_stdev, 0.643259, 4)
        self.assertAlmostEqual(fit_result.params.b_stdev, 0.517992, 4)
        self.assertAlmostEqual(fit_result.params.c_stdev, 2.26594, 4)

        fit = Fit(model, xy, z, sigma=errors)
        fit_result = fit.execute()

        popt, pcov, infodict, errmsg, ier = curve_fit(lambda x_vec, a, b, c: a * np.log(b * x_vec[0] + c * x_vec[1]), xy, z, sigma=errors, absolute_sigma=True, full_output=True)

        # Same as curve_fit?
        self.assertAlmostEqual(fit_result.params.a, popt[0], 4)
        self.assertAlmostEqual(fit_result.params.b, popt[1], 4)
        self.assertAlmostEqual(fit_result.params.c, popt[2], 4)
        self.assertAlmostEqual(fit_result.params.a_stdev, np.sqrt(pcov[0,0]), 4)
        self.assertAlmostEqual(fit_result.params.b_stdev, np.sqrt(pcov[1,1]), 4)
        self.assertAlmostEqual(fit_result.params.c_stdev, np.sqrt(pcov[2,2]), 4)

        # Same as Mathematica with MEASUREMENT ERROR
        self.assertAlmostEqual(fit_result.params.a, 2.68807, 4)
        self.assertAlmostEqual(fit_result.params.b, 0.941344, 4)
        self.assertAlmostEqual(fit_result.params.c, 5.01541, 4)
        self.assertAlmostEqual(fit_result.params.a_stdev, 0.0974628, 4)
        self.assertAlmostEqual(fit_result.params.b_stdev, 0.247018, 4)
        self.assertAlmostEqual(fit_result.params.c_stdev, 0.597661, 4)

    def test_error_analytical(self):
        """
        Test using a case where the analytical answer is known.
        Modeled after:
        http://nbviewer.ipython.org/urls/gist.github.com/taldcroft/5014170/raw/31e29e235407e4913dc0ec403af7ed524372b612/curve_fit.ipynb
        """
        N = 10000
        sigma = 10
        xn = np.arange(N, dtype=np.float)
        yn = np.zeros_like(xn)
        yn = yn + np.random.normal(size=len(yn), scale=sigma)

        a = Parameter()
        model = a

        fit = Fit(model, xn, yn, sigma=sigma)
        fit_result = fit.execute()
        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn, sigma=sigma, absolute_sigma=True)
        self.assertAlmostEqual(fit_result.params.a, popt[0], 5)
        self.assertAlmostEqual(fit_result.params.a_stdev, np.sqrt(np.diag(pcov))[0], 2)

        fit_no_sigma = Fit(model, xn, yn)
        fit_result_no_sigma = fit_no_sigma.execute()
        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn,)
        # With or without sigma, the bestfit params should be in agreement in case of equal weights
        self.assertAlmostEqual(fit_result.params.a, fit_result_no_sigma.params.a, 5)
        # Since symfit is all about absolute errors, the sigma will not be in agreement
        self.assertNotEqual(fit_result.params.a_stdev, fit_result_no_sigma.params.a_stdev, 5)
        self.assertAlmostEqual(fit_result_no_sigma.params.a, popt[0], 5)
        self.assertAlmostEqual(fit_result_no_sigma.params.a_stdev, pcov[0][0]**0.5, 5)

        # Analytical answer for mean of N(0,1):
        mu = 0.0
        sigma_mu = sigma/N**0.5
        # self.assertAlmostEqual(fit_result.params.a, mu, 5)
        self.assertAlmostEqual(fit_result.params.a_stdev, sigma_mu, 5)

    def test_straight_line_analytical(self):
        """
        Test symfit against a straight line, for which the parameters and their
        uncertainties are known analytically. Assuming equal weights.
        :return:
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
        self.assertAlmostEqual(a, 0.694915, 6) # values from Mathematica
        self.assertAlmostEqual(b, 0.186441, 6)
        print(a, b)

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
        self.assertAlmostEqual(a_exact, popt[0], 4)
        self.assertAlmostEqual(b_exact, popt[1], 4)
        self.assertAlmostEqual(var_a_exact, pcov[0][0], 6)
        self.assertAlmostEqual(var_b_exact, pcov[1][1], 6)

        self.assertAlmostEqual(a_exact, fit_result.params.a, 4)
        self.assertAlmostEqual(b_exact, fit_result.params.b, 4)
        self.assertAlmostEqual(var_a_exact**0.5, fit_result.params.a_stdev, 6)
        self.assertAlmostEqual(var_b_exact**0.5, fit_result.params.b_stdev, 6)








if __name__ == '__main__':
    unittest.main()
