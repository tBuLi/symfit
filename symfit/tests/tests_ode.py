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


class TestODE(unittest.TestCase):
    """
    Tests for the FitResults object.
    """
    # def test_known_solution(self):
    #     p, c1, c2 = parameters('p, c1, c2')
    #     y, t = variables('y, t')
    #     p.value = 3.0
    #
    #     model_dict = {
    #         Derivative(y, t): - p * y,
    #     }
    #
    #     print(model_dict[Derivative(y, t)], Derivative(y, t).free_symbols)
    #     # Lets say we know the exact solution to this problem
    #     sol = c1 * exp(- p * t)
    #
    #     # Generate some data
    #     tdata = np.linspace(-3, 3, 101)
    #     ydata = sol(t=tdata, c1=1, p=3.0)
    #
    #     ode_model = ODEModel(model_dict, initial={t: 0.0, y: 1.0}, dt=1/1000., domain=(-5.0, 5.0))
    #     print(ode_model.independent_vars)
    #     # print(ode_model.numerical_components)
    #     # print(len(tdata), len(list(ode_model.numerical_components)))
    #     plt.plot(tdata, ode_model.numerical_components(tdata), color='red')
    #     plt.plot(tdata, ydata)
    #     plt.show()

    # def test_van_der_pol(self):
    #     """
    #     http://hplgit.github.io/odespy/doc/pub/tutorial/html/main_odespy.html
    #     """
    #     u_0, u_1, t = variables('u_0, u_1, t')
    #
    #     model_dict = {
    #         Derivative(u_0, t): u_1,
    #         Derivative(u_1, t): 3 * (1 - u_0**2) * u_1 - u_1
    #     }
    #
    #     ode_model = ODEModel(model_dict, initial={t: 0.0, u_0: 2.0, u_1: 1.0}, dt=.001, domain=(0, 1.0))
    #
    #     # Generate some data
    #     tdata = np.linspace(0, 1, 101)
    #     plt.plot(tdata, ode_model.numerical_components(tdata), color='red')
    #     plt.show()

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

        ode_model = ODEModel(model_dict, initial={t: 0.0, a: a0, c: 0.0, d: 0.0}, dt=.001)

        # Generate some data
        tdata = np.linspace(0, 3, 1000)
        # Eval
        AA, AAB, BAAB = ode_model(t=tdata, k=0.1, l=0.2, m=.3, p=0.3)

        plt.plot(tdata, AA, color='red', label='[AA]')
        plt.plot(tdata, AAB, color='blue', label='[AAB]')
        plt.plot(tdata, BAAB, color='green', label='[BAAB]')
        plt.plot(tdata, b(d=BAAB, a=AA), color='pink', label='[B]')
        plt.plot(tdata, AA + AAB + BAAB, color='black', label='total')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    unittest.main()
