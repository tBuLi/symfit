from __future__ import division, print_function
import unittest
import warnings
import sympy
import types

import numpy as np
from symfit.api import *
from symfit.core.fit import *
from symfit.distributions import Gaussian

import matplotlib.pyplot as plt
import seaborn


class TestODE(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    def test_known_solution(self):
        p, c1, c2 = parameters('p, c1, c2')
        y, t = variables('y, t')
        p.value = 3.0

        model_dict = {
            Derivative(y, t): - p * y,
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

        self.assertAlmostEqual(3.22, fit_result.value(p))

    def test_van_der_pol(self):
        """
        http://hplgit.github.io/odespy/doc/pub/tutorial/html/main_odespy.html
        """
        u_0, u_1, t = variables('u_0, u_1, t')

        model_dict = {
            Derivative(u_0, t): u_1,
            Derivative(u_1, t): 3 * (1 - u_0**2) * u_1 - u_1
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
            Derivative(d, t): l * c * b - m * d,
            Derivative(c, t): k * a * b - p * c - l * c * b + m * d,
            Derivative(a, t): - k * a * b + p * c,
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

        fit = Fit(ode_model, t=tdata, a=adata, b=None)
        fit_result = fit.execute()
        # print(fit_result)
        self.assertAlmostEqual(fit_result.value(k), 4.302875e-01)
        self.assertAlmostEqual(fit_result.stdev(k), 6.447068e-03)

        # A, B = ode_model(t=tvec, **fit_result.params)
        # plt.plot()
        # plt.plot(tvec, A, label='[A]')
        # plt.plot(tvec, B, label='[B]')
        # plt.scatter(tdata, adata)
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
