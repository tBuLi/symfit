"""
Testing for fit's containing indexed variables
"""

from __future__ import division, print_function
import unittest

import numpy as np

from symfit import (
    variables, parameters, symbols, Fit, Parameter, Variable, indices,
    Equality, Model, Idx, D, Sum
)
from symfit.core.minimizers import (
    BFGS, SLSQP, LBFGSB, NelderMead, COBYLA, DifferentialEvolution
)
from symfit.core.objectives import MinimizeModel
from symfit.distributions import Gaussian
from sympy.tensor.index_methods import get_indices, get_contraction_structure

class TestIndexedFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_linear_model(self):
        """
        Simple indexed linear model test. This test is just there to guarantee
        that straight forward fitting still works, even if we don't do anything
        with the indices explicitly.
        :return:
        """
        a, b = parameters('a, b')
        x, y = variables('x, y', indexed=True)
        i = symbols('i', cls=Idx)
        model = {y[i]: a * x[i] + b}

        data = [[0, 1], [1, 0], [3, 2], [5, 4]]
        xdata, ydata = (np.array(i, dtype='float64') for i in zip(*data))

        fit = Fit(model, x=xdata, y=ydata)

        self.assertEqual(type(y), type(list(fit.model.sigmas.keys())[0]))
        self.assertEqual(fit.model[y[i]].free_symbols, {a, b, x, i})
        self.assertEqual(
            fit.model.symbol2indexed,
            {y: y[i], x: x[i], fit.model.sigmas[y]: fit.model.sigmas[y][i], a: a, b: b}
        )
        fit_result = fit.execute()

        # values from Mathematica
        self.assertAlmostEqual(fit_result.value(a), 0.694915, 6)
        self.assertAlmostEqual(fit_result.value(b), 0.186441, 6)

    @unittest.skip('Indexed parameter support pending')
    def test_matrix_equation(self):
        """
        Test if symfit is able to solve a simpel Ax=y linear system.
        :return:
        """
        A_mat = np.array([[1, 1, 1], [0, 2, 5], [2, 5, -1]])
        y_vec = np.array([6, -4, 27])

        x, = parameters('x', indexed=True)
        A, y = variables('A, y', indexed=True)
        i, j = indices('i, j', range=len(y_vec))

        model = {y[i]: A[i, j] * x[j]}
        fit = Fit(model, y=y_vec, A=A_mat)
        fit_result = fit.execute()
        np.testing.assert_array_equal(
            np.array([5., 3., -2.]), fit_result.value(x)
        )

    def test_lagrange_multiplier_raw(self):
        """
        Test the addition of a lagrange multiplier. This uses internal objects
        directly, and is not reflective of the API our users will normally deal
        with.
        :return:
        """
        data = [[0, 1], [1, 0], [3, 2], [5, 4]]
        xdata, ydata = (np.array(i, dtype='float64') for i in zip(*data))

        a, b = parameters('a, b')
        x, y = variables('x, y', indexed=True)
        X2, = variables('X2')
        i, = indices('i', range=len(ydata))
        l = Parameter('l')

        model = Model({y[i]: a * x[i] + b})
        chi2_raw = (model[y[i]] - y[i])**2
        chi2 = {X2: Sum(chi2_raw, i)}
        self.assertEqual({i}, get_indices(chi2_raw)[0])
        self.assertEqual(
            chi2[X2](x=xdata, y=ydata, a=2, b=4),
            np.sum((2 * xdata + 4 - ydata) ** 2)
        )
        self.assertEqual(
            model.numerical_chi_squared(x=xdata, y=ydata,
                                        sigma_y=np.ones_like(ydata), a=2, b=4),
            np.sum((2 * xdata + 4 - ydata) ** 2)
        )

        # Minimize the chi2 model directly.
        fit_chi2 = Fit(chi2, x=xdata, y=ydata, X2=None, objective=MinimizeModel)
        fit = Fit(model, x=xdata, y=ydata)
        fit_chi2_result = fit_chi2.execute()
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.value(a), fit_chi2_result.value(a))
        self.assertAlmostEqual(fit_result.value(b), fit_chi2_result.value(b))
        self.assertNotAlmostEqual(fit_result.value(a) + fit_result.value(b), 1.0)

        L = chi2[X2] + l * (a + b - 1)
        # The jacobian of L should equal zero, that's the system to solve.
        dLdp = Model(Model(L).jacobian[0])

        minimizers = [BFGS, SLSQP, LBFGSB, NelderMead, DifferentialEvolution]
        fit_slsqp = Fit(model, x=xdata, y=ydata,
                        constraints=[Equality(a + b, 1)])
        fit_slsqp_result = fit_slsqp.execute()
        self.assertAlmostEqual(fit_slsqp_result.value(a) + fit_slsqp_result.value(b), 1.0)
        for minimizer in minimizers:
            # The new components should be set to 0.
            fit = Fit(dLdp, x=xdata, y=ydata, minimizer=minimizer,
                      **{str(v): np.array([0.0]) for v in dLdp.dependent_vars})
            if minimizer is DifferentialEvolution:
                a.min, a.max = 0.5, 0.7
                b.min, b.max = 0.3, 0.5
                l.min, l.max = -0.8, -0.6
                a.value = 0.6
                b.value = 0.4
                l.value = 0.7
                fit_result = fit.execute()
            else:
                fit_result = fit.execute()

            self.assertAlmostEqual(fit_result.value(a) + fit_result.value(b), 1.0, 4)
            self.assertAlmostEqual(fit_result.value(a), fit_slsqp_result.value(a), 4)
            self.assertAlmostEqual(fit_result.value(b), fit_slsqp_result.value(b), 4)


    def test_lagrange_multiplier_api(self):
        """
        Test the addition of a lagrange multiplier, using the convenient API.
        :return:
        """
        data = [[0, 1], [1, 0], [3, 2], [5, 4]]
        xdata, ydata = (np.array(i, dtype='float64') for i in zip(*data))

        a, b = parameters('a, b')
        x, y = variables('x, y', indexed=True)
        i, = indices('i', range=len(ydata))
        l = Parameter('l')

        minimizers = [BFGS, SLSQP, LBFGSB, NelderMead, COBYLA, DifferentialEvolution]
        model = {y[i]: a * x[i] + b}
        fit_slsqp = Fit(model, x=xdata, y=ydata, constraints=[Equality(a + b, 1)])
        fit_slsqp_result = fit_slsqp.execute()
        # Try if we now truly have constraints with any minimizer.
        for minimizer in minimizers:
            fit = Fit(model, x=xdata, y=ydata,
                      constraints={l: Equality(a + b, 1)},
                      minimizer=minimizer)
            fit_result = fit.execute()

            self.assertAlmostEqual(fit_result.value(a), fit_slsqp_result.value(a))
            self.assertAlmostEqual(fit_result.value(b), fit_slsqp_result.value(b))
            self.assertNotAlmostEqual(fit_result.value(a) + fit_result.value(b), 1.0)

if __name__ == '__main__':
    unittest.main()