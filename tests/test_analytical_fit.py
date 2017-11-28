from __future__ import division, print_function
import unittest
import warnings
import types

import sympy
import numpy as np
from scipy.optimize import curve_fit

from symfit import Variable, Parameter, Fit, FitResults, LinearLeastSquares, parameters, variables, NonLinearLeastSquares, Model, TaylorModel
from symfit.core.minimizers import MINPACK
from symfit.core.support import seperate_symbols, sympy_to_py
from symfit.distributions import Gaussian


class TestAnalyticalFit(unittest.TestCase):
    """
    Tests for Analytical fitting objects.
    """
    def test_linear_analytical_fit(self):
        a, b = parameters('a, b')
        x, y = variables('x, y')
        model = {y: a * x + b}

        data = [[0, 1], [1, 0], [3, 2], [5, 4]]
        xdata, ydata = (np.array(i, dtype='float64') for i in zip(*data))

        fit = LinearLeastSquares(model, x=xdata, y=ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), 0.694915, 6) # values from Mathematica
        self.assertAlmostEqual(fit_result.value(b), 0.186441, 6)

    def test_taylor_model(self):
        a, b = parameters('a, b')
        x, y, z = variables('x, y, z')

        model = Model({y: a * x + b})
        appr = TaylorModel(model)
        self.assertEqual(set([a, b]), set(appr.params))
        appr.p0 = {a: 2.0, b: 5.0}
        self.assertEqual(set(appr.p0.keys()), set(appr.params_0[p] for p in appr.params))
        self.assertTrue(LinearLeastSquares.is_linear(appr))

        model = Model({z: a * x**2 + b * y**2})
        appr = TaylorModel(model)
        appr.p0 = {a: 2, b: 5}
        model = Model({z: a * x**2 + b * y**2})
        appr_2 = TaylorModel(model)
        appr_2.p0 = {a: 1, b: 1}
        self.assertTrue(appr == appr_2)

        model = Model({y: a * sympy.exp(x * b)})
        appr = TaylorModel(model)
        appr.p0 = {a: 2.0, b: 5.0}
        self.assertTrue(LinearLeastSquares.is_linear(appr))

        model = Model({y: sympy.sin(a * x)})
        appr = TaylorModel(model)
        appr.p0 = {a: 0.0}
        self.assertTrue(LinearLeastSquares.is_linear(appr))


    def test_straight_line_analytical(self):
        """
        Test symfit against a straight line, for which the parameters and their
        uncertainties are known analytically. Assuming equal weights.
        """
        data = [[0, 1], [1, 0], [3, 2], [5, 4]]
        xdata, ydata = (np.array(i, dtype='float64') for i in zip(*data))
        # x = np.arange(0, 100, 0.1)
        # np.random.seed(10)
        # y = 3.0*x + 105.0 + np.random.normal(size=x.shape)

        dx = xdata - xdata.mean()
        dy = ydata - ydata.mean()
        mean_squared_x = np.mean(xdata**2) - np.mean(xdata)**2
        mean_xy = np.mean(xdata * ydata) - np.mean(xdata)*np.mean(ydata)
        a = mean_xy/mean_squared_x
        b = ydata.mean() - a * xdata.mean()
        self.assertAlmostEqual(a, 0.694915, 6) # values from Mathematica
        self.assertAlmostEqual(b, 0.186441, 6)

        S = np.sum((ydata - (a*xdata + b))**2)
        var_a_exact = S/(len(xdata) * (len(xdata) - 2) * mean_squared_x)
        var_b_exact = var_a_exact*np.mean(xdata**2)
        a_exact = a
        b_exact = b

        # We will now compare these exact results with values from symfit, numerically
        a, b = parameters('a, b')
        x, y = variables('x, y')
        model = {y: a*x + b}
        fit = Fit(model, x=xdata, y=ydata, minimizer=MINPACK)#, absolute_sigma=False)
        fit_result = fit.execute()

        popt, pcov = curve_fit(lambda z, c, d: c * z + d, xdata, ydata,
                               jac=lambda z, c, d: np.transpose([xdata, np.ones_like(xdata)]))
#                               jac=lambda p, x, y, func: np.transpose([x, np.ones_like(x)]))
                                # Dfun=lambda p, x, y, func: print(p, func, x, y))

        # curve_fit
        self.assertAlmostEqual(a_exact, popt[0], 4)
        self.assertAlmostEqual(b_exact, popt[1], 4)
        self.assertAlmostEqual(var_a_exact, pcov[0][0], 6)
        self.assertAlmostEqual(var_b_exact, pcov[1][1], 6)

        self.assertAlmostEqual(a_exact, fit_result.value(a), 4)
        self.assertAlmostEqual(b_exact, fit_result.value(b), 4)
        self.assertAlmostEqual(var_a_exact, fit_result.variance(a), 6)
        self.assertAlmostEqual(var_b_exact, fit_result.variance(b), 6)

        # Do the fit with the LinearLeastSquares object
        fit = LinearLeastSquares(model, x=xdata, y=ydata)
        fit_result = fit.execute()
        self.assertAlmostEqual(a_exact, fit_result.value(a), 4)
        self.assertAlmostEqual(b_exact, fit_result.value(b), 4)
        self.assertAlmostEqual(var_a_exact, fit_result.variance(a), 6)
        self.assertAlmostEqual(var_b_exact, fit_result.variance(b), 6)

        # Lets also make sure the entire covariance matrix is the same
        for cov1, cov2 in zip(fit_result.covariance_matrix.flatten(), pcov.flatten()):
            self.assertAlmostEqual(cov1, cov2)

    def test_is_linear(self):
        a, b, c, d = parameters('a, b, c, d')
        x, y = variables('x, y')

        model = Model({y: (a * x + c*x**2) + b})
        self.assertTrue(LinearLeastSquares.is_linear(model))

        model = Model({y: (a * x + c*x**2) + b + 2})
        self.assertTrue(LinearLeastSquares.is_linear(model))

        # This test should be made to work in a future version.
        # model = Model({y: a * x**2 + sympy.exp(x * b)})
        # t_model = (2 * y / g)**0.5
        # self.assertTrue(LinearLeastSquares.is_linear(model))

        model = Model({y: a * sympy.exp(x * b)})
        self.assertFalse(LinearLeastSquares.is_linear(model))

        model = Model({y: a * x**3 + b * c})
        self.assertFalse(LinearLeastSquares.is_linear(model))

        model = Model({y: a * x**3 + b * x + c})
        self.assertTrue(LinearLeastSquares.is_linear(model))

    def test_weights(self):
        """
        Compare NumericalLeastSquares with LinearLeastSquares to see if errors
        are implemented consistently.
        """
        from symfit import Variable, Parameter, Fit

        t_data = np.array([1.4, 2.1, 2.6, 3.0, 3.3])
        y_data = np.array([10, 20, 30, 40, 50])

        sigma = 0.2
        n = np.array([5, 3, 8, 15, 30])
        sigma_t = sigma / np.sqrt(n)

        # We now define our model
        t, y = variables('t, y')
        b = Parameter()
        sqrt_g_inv = Parameter() # sqrt_g_inv = sqrt(1/g). Currently needed to linearize.
        # t_model = (2 * y / g)**0.5
        t_model = {t: 2 * y**0.5 * sqrt_g_inv + b}

        # Different sigma for every point
        fit = Fit(t_model, y=y_data, t=t_data, sigma_t=sigma_t, absolute_sigma=False, minimizer=MINPACK)
        num_result_rel = fit.execute()

        fit = Fit(t_model, y=y_data, t=t_data, sigma_t=sigma_t, absolute_sigma=True, minimizer=MINPACK)
        num_result = fit.execute()

        # cov matrix should now be different
        for cov1, cov2 in zip(num_result_rel.covariance_matrix.flatten(), num_result.covariance_matrix.flatten()):
            # Make the absolute cov relative to see if it worked.
            ss_res = np.sum(num_result_rel.infodict['fvec']**2)
            degrees_of_freedom = len(fit.data[fit.model.dependent_vars[0].name]) - len(fit.model.params)
            s_sq = ss_res / degrees_of_freedom
            self.assertAlmostEqual(cov1, cov2 * s_sq)

        # print(fit.model.numerical_chi_jacobian[0](sqrt_g_inv=1, **fit.data))

        fit = LinearLeastSquares(t_model, y=y_data, t=t_data, sigma_t=sigma_t)
        fit_result = fit.execute()

        self.assertAlmostEqual(num_result.value(sqrt_g_inv), fit_result.value(sqrt_g_inv))
        self.assertAlmostEqual(num_result.value(b) / fit_result.value(b), 1.0, 5)
        # for cov1, cov2 in zip(num_result.params.covariance_matrix.flatten(), fit_result.params.covariance_matrix.flatten()):
        #     self.assertAlmostEqual(cov1, cov2)
        #     print(cov1, cov2)

        for cov1, cov2 in zip(num_result.covariance_matrix.flatten(), fit_result.covariance_matrix.flatten()):
            self.assertAlmostEqual(cov1 / cov2, 1.0, 5)
            # print(cov1, cov2)

    def test_backwards_compatibility(self):
        """
        The LinearLeastSquares should give results compatible with the NumericalLeastSquare's
        and curve_fit. To do this I test here the simple analytical model also used to calibrate
        the definition of absolute_sigma.
        """
        N = 1000
        sigma = 31.4 * np.ones(N)
        xn = np.arange(N, dtype=np.float)
        yn = np.zeros_like(xn)
        np.random.seed(10)
        yn = yn + np.random.normal(size=len(yn), scale=sigma)

        a = Parameter()
        y = Variable()
        model = {y: a}

        fit = LinearLeastSquares(model, y=yn, sigma_y=sigma, absolute_sigma=False)
        fit_result = fit.execute()

        fit = Fit(model, y=yn, sigma_y=sigma, absolute_sigma=False, minimizer=MINPACK)
        num_result = fit.execute()

        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn, sigma=sigma, absolute_sigma=False)


        self.assertAlmostEqual(fit_result.value(a), num_result.value(a), 5)
        self.assertAlmostEqual(fit_result.stdev(a), num_result.stdev(a), 5)

        self.assertAlmostEqual(fit_result.value(a), popt[0], 5)
        self.assertAlmostEqual(fit_result.stdev(a), pcov[0, 0]**0.5, 5)

        fit = LinearLeastSquares(model, y=yn, sigma_y=sigma, absolute_sigma=True)
        fit_result = fit.execute()

        fit = Fit(model, y=yn, sigma_y=sigma, absolute_sigma=True, minimizer=MINPACK)
        num_result = fit.execute()

        popt, pcov = curve_fit(lambda x, a: a * np.ones_like(x), xn, yn, sigma=sigma, absolute_sigma=True)

        self.assertAlmostEqual(fit_result.value(a), num_result.value(a), 5)
        self.assertAlmostEqual(fit_result.stdev(a), num_result.stdev(a), 5)

        self.assertAlmostEqual(fit_result.value(a), popt[0], 5)
        self.assertAlmostEqual(fit_result.stdev(a), pcov[0, 0]**0.5, 5)
    #
    def test_nonlinearfit(self):
        """
        Compare NumericalLeastSquares with LinearLeastSquares to see if errors
        are implemented consistently.
        """
        from symfit import Variable, Parameter, Fit

        t_data = np.array([1.4, 2.1, 2.6, 3.0, 3.3])
        y_data = np.array([10, 20, 30, 40, 50])

        sigma = 0.2
        n = np.array([5, 3, 8, 15, 30])
        sigma_t = sigma / np.sqrt(n)

        # We now define our model
        t, y = variables('t, y')
        g = Parameter(9.0)
        t_model = {t: (2 * y / g)**0.5}

        # Different sigma for every point
        fit = NonLinearLeastSquares(t_model, y=y_data, t=t_data, sigma_t=sigma_t)
        import time
        tick = time.time()
        fit_result = fit.execute()
#        print(time.time() - tick)

        fit = Fit(t_model, y=y_data, t=t_data, sigma_t=sigma_t, minimizer=MINPACK)
        tick = time.time()
        num_result = fit.execute()
#        print(time.time() - tick)

        self.assertAlmostEqual(num_result.value(g), fit_result.value(g))

        for cov1, cov2 in zip(num_result.covariance_matrix.flatten(), fit_result.covariance_matrix.flatten()):
            self.assertAlmostEqual(cov1, cov2)

    def test_2D_fitting(self):
        np.random.seed(1)
        xdata = np.random.randint(-10, 11, size=(2, 100))
        zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

        a = Parameter()
        b = Parameter(10)
        x = Variable()
        y = Variable()
        z = Variable()
        new = {z: a*x**2 + b*y**2}

        fit = NonLinearLeastSquares(new, x=xdata[0], y=xdata[1], z=zdata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), 2.5)
        self.assertAlmostEqual(np.abs(fit_result.value(b)), 7.0)

    # def test_2_gaussian_2d_fitting(self):
    #     np.random.seed(4242)
    #     mean = (0.3, 0.3) # x, y mean 0.6, 0.4
    #     cov = [[0.1**2,0],[0,0.1**2]]
    #     data = np.random.multivariate_normal(mean, cov, 100)
    #     mean = (0.7,0.7) # x, y mean 0.6, 0.4
    #     cov = [[0.1**2,0],[0,0.1**2]]
    #     data_2 = np.random.multivariate_normal(mean, cov, 100)
    #     data = np.vstack((data, data_2))
    #
    #     # Insert them as y,x here as np fucks up cartesian conventions.
    #     ydata, xedges, yedges = np.histogram2d(data[:,1], data[:,0], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
    #     xcentres = (xedges[:-1] + xedges[1:]) / 2
    #     ycentres = (yedges[:-1] + yedges[1:]) / 2
    #
    #     # Make a valid grid to match ydata
    #     xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
    #     # xdata = np.dstack((xx, yy)).T
    #
    #     x, y, z = variables('x, y, z')
    #
    #     x0_1 = Parameter(0.7, min=0.6, max=0.8)
    #     sig_x_1 = Parameter(0.1, min=0.0, max=0.2)
    #     y0_1 = Parameter(0.7, min=0.6, max=0.8)
    #     sig_y_1 = Parameter(0.1, min=0.0, max=0.2)
    #     A_1 = Parameter()
    #     g_1 = A_1 * Gaussian(x, x0_1, sig_x_1) * Gaussian(y, y0_1, sig_y_1)
    #
    #     x0_2 = Parameter(0.3, min=0.2, max=0.4)
    #     sig_x_2 = Parameter(0.1, min=0.0, max=0.2)
    #     y0_2 = Parameter(0.3, min=0.2, max=0.4)
    #     sig_y_2 = Parameter(0.1, min=0.0, max=0.2)
    #     A_2 = Parameter()
    #     g_2 = A_2 * Gaussian(x, x0_2, sig_x_2) * Gaussian(y, y0_2, sig_y_2)
    #
    #     model = {z: g_1 + g_2}
    #     fit = NonLinearLeastSquares(model, x=xx, y=yy, z=ydata)
    #     fit_result = fit.execute()
    #     print(fit_result)
    #
    #     # Equal up to some precision. Not much obviously.
    #     self.assertAlmostEqual(fit_result.params.x0_1, 0.7, 2)
    #     self.assertAlmostEqual(fit_result.params.y0_1, 0.7, 2)
    #     self.assertAlmostEqual(fit_result.params.x0_2, 0.3, 2)
    #     self.assertAlmostEqual(fit_result.params.y0_2, 0.3, 2)

    # def test_lagrange_equility(self):
    #     """
    #     Lagrange Multipliers.
    #     Example 1 from
    #     http://en.wikipedia.org/wiki/Lagrange_multiplier#Examples
    #     """
    #     x, y = parameters('x, y')
    #
    #     f = x + y
    #     constraints = [
    #         sympy.Eq(x**2 + y**2, 1)
    #     ]
    #
    #     fit = LagrangeMultipliers(f, constraints)
    #     self.assertTrue(
    #         [
    #             (sympy.sqrt(2)/2, sympy.sqrt(2)/2, sympy.sqrt(2)),
    #             (-sympy.sqrt(2)/2, -sympy.sqrt(2)/2, -sympy.sqrt(2))
    #         ] == fit.extrema
    #     )

    # def test_lagrange_inequility(self):
    #     """
    #     Examples taken from
    #     http://mat.gsia.cmu.edu/classes/QUANT/NOTES/chap4/node6.html
    #     """
    #     x = Parameter()
    #
    #     f = x**3 - 3 * x
    #     constraints = [
    #         sympy.Le(x, 2)
    #         # x <= 2,
    #     ]
    #
    #     # maximize
    #     fit = LagrangeMultipliers(-f, constraints)
    #     self.assertTrue([(2, -2), (-1, -2), (1, 2)] == fit.extrema)
    #
    #     # minimize
    #     fit = LagrangeMultipliers(f, constraints)
    #     self.assertTrue([(-1, 2), (1, -2)] == fit.extrema)
    #
    #     # Example 2
    #     x = Parameter()
    #     y = Parameter()
    #
    #     f = (x - 2)**2 + 2*(y - 1)**2
    #     constraints = [
    #         sympy.Le(x + 4*y, 3),
    #         sympy.Ge(x, y),
    #     ]
    #     fit = LagrangeMultipliers(f, constraints)
    #     print(fit.solutions)
    #     print(fit.extrema)
    #     self.assertTrue([(5/3, 1/3, 1)] == fit.extrema)

    # def test_lagrange_least_squares(self):
    #     """
    #     http://www.asp.ucar.edu/colloquium/1992/notes/part1/node36.html
    #     """
    #     a, b, c= parameters('a, b, c')
    #     a_i, b_i, c_i = variables('a_i, b_i, c_i')
    #     # a_i, b_i, c_i, s_a, s_b, s_c = variables('a_i, b_i, c_i, s_a, s_b, s_c')
    #     i, N = symbols('i, N', integer=True)
    #
    #     # f = sympy.summation(((a_i - a)/s_a)**2 + ((b_i - b)/s_b)**2 + ((c_i - c)/s_c)**2, (i, 0, N- 1))
    #     # f = sympy.summation(((a_i - a))**2 + ((b_i - b))**2 + ((c_i - c))**2, (i, 0, N- 1))
    #     f = ((a_i - a))**2 + ((b_i - b))**2 + ((c_i - c))**2
    #     constraints = [
    #         sympy.Eq(a + b + c, 180)
    #     ]
    #
    #     fit = LagrangeMultipliers(f, constraints)
    #
    #     xdata = np.array([
    #         [10., 10., 10., 10., 10., 10., 10.],
    #         [100., 100., 100., 100., 100., 100., 100.],
    #         [70., 70., 70., 70., 70., 70., 70.],
    #     ])
    #     # self.assertTrue([(2, 2), (-1, 2)] == fit.maxima)
    #     # self.assertTrue([(1, -2)] == fit.minima)
    #     # print(fit.lagrangian)
    #     # print('sol', fit.solutions)
    #     # print('l_0', fit.solutions[0][fit.l_params[0]])
    #     # print('a', fit.solutions[0][a])
    #     # extrema = fit.extrema
    #     # print(extrema[0].a)
    #
    #     fit = ConstrainedFit(f, constraints=constraints, x=xdata)
    #     fit.execute()

    # def test_simple_sigma(self):
    #     t_data = np.array([1.4, 2.1, 2.6, 3.0, 3.3])
    #     y_data = np.array([10, 20, 30, 40, 50])
    #
    #     sigma = 0.2
    #     n = np.array([5, 3, 8, 15, 30])
    #     # sigma_t = sigma / np.sqrt(n)
    #
    #     # We now define our model
    #     t = Variable()
    #     g = Parameter()
    #     y_model = 0.5 * g * t**2
    #
    #     constraints = [
    #         sympy.Le(g, 20)
    #     ]
    #
    #     fit = ConstrainedFit(y_model, x=t_data, y=y_data)# constraints=constraints)#, sigma=sigma_t)
    #     fit_result = fit.execute()
    #     print(fit_result)
    #     print([sympy.diff(fit.analytic_fit.lagrangian, p) for p in fit.analytic_fit.all_params])

if __name__ == '__main__':
    unittest.main()
