from __future__ import division, print_function
import unittest

import numpy as np
from symfit import parameters, variables, ODEModel, exp, Fit, D
from symfit.core.minimizers import MINPACK
from symfit.distributions import Gaussian


class TestODE(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    def test_known_solution(self):
        p, c1, c2 = parameters('p, c1, c2')
        y, t = variables('y, t')
        p.value = 3.0

        model_dict = {
            D(y, t): - p * y,
        }

        # Lets say we know the exact solution to this problem
        sol = c1 * exp(- p * t)

        # Generate some data
        tdata = np.linspace(0, 3, 101)
        ydata = sol(t=tdata, c1=1.0, p=3.22)


        ode_model = ODEModel(model_dict, initial={t: 0.0, y: 1.0})
        fit = Fit(ode_model, t=tdata, y=ydata)
        fit_result = fit.execute()
        y_sol, = ode_model(tdata, **fit_result.params)

        self.assertAlmostEqual(3.22, fit_result.value(p), 2)

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

        # Generate some data
        tvec = np.linspace(0, 500, 1000)

        fit = Fit(ode_model, t=tdata, a=adata, b=None, minimizer=MINPACK)
        fit_result = fit.execute()
        # print(fit_result)
        self.assertAlmostEqual(fit_result.value(k), 4.302875e-01, 4)
        self.assertAlmostEqual(fit_result.stdev(k), 6.447068e-03, 4)

        fit = Fit(ode_model, t=tdata, a=adata, b=None)
        fit_result = fit.execute()
        # print(fit_result)
        print(fit_result.stdev(k))
        self.assertAlmostEqual(fit_result.value(k), 4.302875e-01, 4)
        self.assertTrue(fit_result.stdev(k) is None or np.isnan(fit_result.stdev(k)))

        # A, B = ode_model(t=tvec, **fit_result.params)
        # plt.plot()
        # plt.plot(tvec, A, label='[A]')
        # plt.plot(tvec, B, label='[B]')
        # plt.scatter(tdata, adata)
        # plt.legend()
        # plt.show()

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
        tdata = np.linspace(0, 100, 101)
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

        # plt.plot(tdata, Y)
        # plt.scatter(tdata[-1], Y_point)
        # plt.show()

    def test_mixed_model(self):
        """
        In principle some components of the model might be provided as ODEs
        while others are not. This is a slightly fabricated scenario to test if
        this is true.

        We take a harmonic oscilater as a system of first order ODEs and
        partially solve it. This should be the same as the original ODE.

        DISCLAIMER
        I'm not even conviced this should be allowed, and since it doesn't work
        out of the box I'm not going te break my head over it. If a usecase
        presents itself I'll look into it again.
        """
        pass


        # x, t = variables('x, t')
        # k, C = parameters('k, C') # C is the integration constant.
        #
        # # First order system, partially integrated
        # y = k * x * t,
        # mixed_dict = {
        #     D(x, t): - k * y,
        # }
        # mixed_model = ODEModel(mixed_dict, initial={t: 0.0, x: 1.0})
        #
        # tdata = np.linspace(0, 31.6, 1000)
        # # Eval
        # X, = mixed_model(t=tdata, k=0.1)
        #
        # plt.plot(t, X)
        # plt.show()
        #
        # x, y, t = variables('x, y, t')
        # k, C = parameters('k, C') # C is the integration constant.
        #
        # # The harmonic oscillator as a system, >1st order is not supported yet.
        # harmonic_dict = {
        #     D(x, t): - k * y,
        #     D(y, t): k * x,
        # }
        # harmonic_model = ODEModel(harmonic_dict, initial={t: 0.0, x: 1.0, y: 0.0})
        #
        # tdata = np.linspace(0, 31.6, 1000)
        # # Eval
        # X, Y = harmonic_model(t=tdata, k=0.1)
        #
        # plt.scatter(X, Y)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
