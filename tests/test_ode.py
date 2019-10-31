from __future__ import division, print_function
import sys
import unittest

import pytest
import numpy as np

from symfit import parameters, variables, ODEModel, exp, Fit, D, Model, GradientModel, Parameter
from symfit.core.minimizers import MINPACK
from symfit.distributions import Gaussian


class TestODE(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    @classmethod
    def setUpClass(cls):
        np.random.seed(6)

    def test_known_solution(self):
        p, c1 = parameters('p, c1')
        y, t = variables('y, t')
        p.value = 3.0

        model_dict = {
            D(y, t): - p * y,
        }

        # Lets say we know the exact solution to this problem
        sol = Model({y: exp(- p * t)})

        # Generate some data
        tdata = np.linspace(0, 3, 10001)
        ydata = sol(t=tdata, p=3.22)[0]
        ydata += np.random.normal(0, 0.005, ydata.shape)

        ode_model = ODEModel(model_dict, initial={t: 0.0, y: ydata[0]})
        fit = Fit(ode_model, t=tdata, y=ydata)
        ode_result = fit.execute()

        c1.value = ydata[0]
        fit = Fit(sol, t=tdata, y=ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(ode_result.value(p) / fit_result.value(p), 1, 2)
        self.assertAlmostEqual(ode_result.r_squared / fit_result.r_squared, 1, 4)
        self.assertAlmostEqual(ode_result.stdev(p) / fit_result.stdev(p), 1, 3)

    def test_van_der_pol(self):
        """
        http://hplgit.github.io/odespy/doc/pub/tutorial/html/main_odespy.html
        """
        u_0, u_1, t = variables('u_0, u_1, t')

        model_dict = {
            D(u_0, t): u_1,
            D(u_1, t): 3 * (1 - u_0**2) * u_1 - u_1
        }

        ode_model = ODEModel(model_dict, initial={t: 0.0, u_0: 2.0, u_1: 1.0})

        # # Generate some data
        # tdata = np.linspace(0, 1, 101)
        # plt.plot(tdata, ode_model(tdata)[0], color='red')
        # plt.plot(tdata, ode_model(tdata)[1], color='blue')
        # plt.show()

    def test_polgar(self):
        """
        Analysis of data published here:
        This whole ODE support was build to do this analysis in the first place
        """
        a, b, c, d, t = variables('a, b, c, d, t')
        k, p, l, m = parameters('k, p, l, m')

        a0 = 10
        b = a0 - d + a
        model_dict = {
            D(d, t): l * c * b - m * d,
            D(c, t): k * a * b - p * c - l * c * b + m * d,
            D(a, t): - k * a * b + p * c,
        }

        ode_model = ODEModel(model_dict, initial={t: 0.0, a: a0, c: 0.0, d: 0.0})

        # Generate some data
        tdata = np.linspace(0, 3, 1000)
        # Eval
        AA, AAB, BAAB = ode_model(t=tdata, k=0.1, l=0.2, m=.3, p=0.3)

        # plt.plot(tdata, AA, color='red', label='[AA]')
        # plt.plot(tdata, AAB, color='blue', label='[AAB]')
        # plt.plot(tdata, BAAB, color='green', label='[BAAB]')
        # plt.plot(tdata, b(d=BAAB, a=AA), color='pink', label='[B]')
        # plt.plot(tdata, AA + AAB + BAAB, color='black', label='total')
        # plt.legend()
        # plt.show()

    def test_initial_parameters(self):
        """
        Identical to test_polgar, but with a0 as free Parameter.
        """
        a, b, c, d, t = variables('a, b, c, d, t')
        k, p, l, m = parameters('k, p, l, m')

        a0 = Parameter('a0', min=0, value=10, fixed=True)
        c0 = Parameter('c0', min=0, value=0.1)
        b = a0 - d + a
        model_dict = {
            D(d, t): l * c * b - m * d,
            D(c, t): k * a * b - p * c - l * c * b + m * d,
            D(a, t): - k * a * b + p * c,
        }

        ode_model = ODEModel(model_dict, initial={t: 0.0, a: a0, c: c0, d: 0.0})

        # Generate some data
        tdata = np.linspace(0, 3, 1000)
        # Eval
        AA, AAB, BAAB = ode_model(t=tdata, k=0.1, l=0.2, m=.3, p=0.3, a0=10, c0=0)
        fit = Fit(ode_model, t=tdata, a=AA, c=AAB, d=BAAB)
        results = fit.execute()
        print(results)
        self.assertEqual(results.value(a0), 10)
        self.assertAlmostEqual(results.value(c0), 0)

        self.assertEqual([a0, c0, k, l, m, p], ode_model.params)
        self.assertEqual([a0, c0], ode_model.initial_params)
        self.assertEqual([a0, k, l, m, p], ode_model.model_params)

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

        # Analytical solution
        model = GradientModel({a: 1 / (k * t + 1 / a0)})
        fit = Fit(model, t=tdata, a=adata)
        fit_result = fit.execute()

        fit = Fit(ode_model, t=tdata, a=adata, b=None, minimizer=MINPACK)
        ode_result = fit.execute()
        self.assertAlmostEqual(ode_result.value(k) / fit_result.value(k), 1.0, 4)
        self.assertAlmostEqual(ode_result.stdev(k) / fit_result.stdev(k), 1.0, 4)
        self.assertAlmostEqual(ode_result.r_squared / fit_result.r_squared, 1, 4)

        fit = Fit(ode_model, t=tdata, a=adata, b=None)
        ode_result = fit.execute()
        self.assertAlmostEqual(ode_result.value(k) / fit_result.value(k), 1.0, 4)
        self.assertAlmostEqual(ode_result.stdev(k) / fit_result.stdev(k), 1.0, 4)
        self.assertAlmostEqual(ode_result.r_squared / fit_result.r_squared, 1, 4)

    def test_single_eval(self):
        """
        Eval an ODEModel at a single value rather than a vector.
        """
        x, y, t = variables('x, y, t')
        k, = parameters('k') # C is the integration constant.

        # The harmonic oscillator as a system, >1st order is not supported yet.
        harmonic_dict = {
            D(x, t): - k * y,
            D(y, t): k * x,
        }

        # Make a second model to prevent caching of integration results.
        # This also means harmonic_dict should NOT be a Model object.
        harmonic_model_array = ODEModel(harmonic_dict, initial={t: 0.0, x: 1.0, y: 0.0})
        harmonic_model_points = ODEModel(harmonic_dict, initial={t: 0.0, x: 1.0, y: 0.0})
        tdata = np.linspace(-100, 100, 101)
        X, Y = harmonic_model_array(t=tdata, k=0.1)
        # Shuffle the data to prevent using the result at time t to calculate
        # t+dt
        random_order = np.random.permutation(len(tdata))
        for idx in random_order:
            t = tdata[idx]
            X_val = X[idx]
            Y_val = Y[idx]
            X_point, Y_point = harmonic_model_points(t=t, k=0.1)
            self.assertAlmostEqual(X_point[0], X_val)
            self.assertAlmostEqual(Y_point[0], Y_val)


    def test_full_eval_range(self):
        """
        Test if ODEModels can be evaluated at t < t_initial.

        A bit of a no news is good news test.
        """
        tdata = np.array([0, 10, 26, 44, 70, 120])
        adata = 10e-4 * np.array([54, 44, 34, 27, 20, 14])
        a, b, t = variables('a, b, t')
        k, a0 = parameters('k, a0')
        k.value = 0.01
        t0 = tdata[2]
        a0 = adata[2]
        b0 = 0.02729855 # Obtained from evaluating from t=0.

        model_dict = {
            D(a, t): - k * a**2,
            D(b, t): k * a**2,
        }

        ode_model = ODEModel(model_dict, initial={t: t0, a: a0, b: b0})

        fit = Fit(ode_model, t=tdata, a=adata, b=None)
        ode_result = fit.execute()
        self.assertGreater(ode_result.r_squared, 0.95, 4)

        # Now start from a timepoint that is not in the t-array such that it
        # triggers another pathway to be taken in integrating it.
        # Again, no news is good news.
        ode_model = ODEModel(model_dict, initial={t: t0 + 1e-5, a: a0, b: b0})

        fit = Fit(ode_model, t=tdata, a=adata, b=None)
        ode_result = fit.execute()
        self.assertGreater(ode_result.r_squared, 0.95, 4)

    def test_odemodel_sanity(self):
        """
        If a user provides an ODE like model directly to fit without
        explicitly turning it into one, give a warning.
        """
        tdata = np.array([0, 10, 26, 44, 70, 120])
        adata = 10e-4 * np.array([54, 44, 34, 27, 20, 14])
        a, t = variables('a, t')
        k, a0 = parameters('k, a0')

        model_dict = {
            D(a, t): - k * a * t,
        }
        with self.assertRaises(RuntimeWarning):
            fit = Fit(model_dict, t=tdata, a=adata)

if __name__ == '__main__':
    unittest.main()
