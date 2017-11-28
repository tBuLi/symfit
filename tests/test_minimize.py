from __future__ import division, print_function
import unittest
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit, Model
)
from symfit.core.objectives import MinimizeModel
from symfit.core.minimizers import BFGS


class TestMinimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    # TODO: Should be 2 tests?
    def test_minimize(self):
        """
        Tests maximizing a function with and without constraints, taken from the
        scipy `minimize` tutorial. Compare the symfit result with the scipy
        result.
        https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
        """
        x = Parameter(-1.0)
        y = Parameter(1.0)
        z = Variable()
        model = Model({z: 2*x*y + 2*x - x**2 - 2*y**2})

        constraints = [
            Ge(y - 1, 0),  # y - 1 >= 0,
            Eq(x**3 - y, 0),  # x**3 - y == 0,
        ]

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

        # Unconstrained fit
        res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
               method='BFGS', options={'disp': False})
        fit = Fit(model=- model)
        self.assertIsInstance(fit.objective, MinimizeModel)
        self.assertIsInstance(fit.minimizer, BFGS)

        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(x) / res.x[0], 1.0, 6)
        self.assertAlmostEqual(fit_result.value(y) / res.x[1], 1.0, 6)

        # Same test, but with constraints in place.
        res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
               constraints=cons, method='SLSQP', options={'disp': False})

        from symfit.core.minimizers import SLSQP
        fit = Fit(- model, constraints=constraints)
        self.assertEqual(fit.constraints[0].constraint_type, Ge)
        self.assertEqual(fit.constraints[1].constraint_type, Eq)
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.value(x), res.x[0], 6)
        self.assertAlmostEqual(fit_result.value(y), res.x[1], 6)

    def test_constraint_types(self):
        x = Parameter(-1.0)
        y = Parameter(1.0)
        z = Variable()
        model = Model({z: 2*x*y + 2*x - x**2 - 2*y**2})

        # These types are not allowed constraints.
        for relation in [Lt, Gt, Ne]:
            with self.assertRaises(ModelError):
                Fit(model, constraints=[relation(x, y)])

        # Should execute without problems.
        for relation in [Eq, Ge, Le]:
            Fit(model, constraints=[relation(x, y)])

        fit = Fit(model, constraints=[Le(x, y)])
        # Le should be transformed to Ge
        self.assertIs(fit.constraints[0].constraint_type, Ge)

        # Redo the standard test as a Le
        constraints = [
            Le(- y + 1, 0),  # y - 1 >= 0,
            Eq(x**3 - y, 0),  # x**3 - y == 0,
        ]
        std_constraints = [
            Ge(y - 1, 0),  # y - 1 >= 0,
            Eq(x**3 - y, 0),  # x**3 - y == 0,
        ]

        fit = Fit(- model, constraints=constraints)
        std_fit = Fit(- model, constraints=std_constraints)
        self.assertEqual(fit.constraints[0].constraint_type, Ge)
        self.assertEqual(fit.constraints[1].constraint_type, Eq)
        fit_result = fit.execute()
        std_result = std_fit.execute()
        self.assertAlmostEqual(fit_result.value(x), std_result.value(x))
        self.assertAlmostEqual(fit_result.value(y), std_result.value(y))



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
