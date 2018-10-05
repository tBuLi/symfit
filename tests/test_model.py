from __future__ import division, print_function
import unittest
from collections import OrderedDict

import numpy as np

from symfit import (
    Fit, parameters, variables, Model, Constraint, ODEModel, D, Eq,
    CallableModel, CallableNumericalModel
)


class TestModel(unittest.TestCase):
    """
    Tests for Model objects.
    """
    def test_model_as_dict(self):
        x, y_1, y_2 = variables('x, y_1, y_2')
        a, b = parameters('a, b')

        model_dict = OrderedDict([(y_1, a * x**2), (y_2, 2 * x * b)])
        model = Model(model_dict)

        self.assertEqual(id(model[y_1]), id(model_dict[y_1]))
        self.assertEqual(id(model[y_2]), id(model_dict[y_2]))
        self.assertEqual(len(model), len(model_dict))
        self.assertEqual(model.items(), model_dict.items())
        self.assertEqual(model.keys(), model_dict.keys())
        self.assertEqual(list(model.values()), list(model_dict.values()))
        self.assertTrue(y_1 in model)
        self.assertFalse(model[y_1] in model)

    def test_order(self):
        """
        The model has to behave like an OrderedDict. This is of the utmost importance!
        """
        x, y_1, y_2 = variables('x, y_1, y_2')
        a, b = parameters('a, b')

        model_dict = {y_2: a * x**2, y_1: 2 * x * b}
        model = Model(model_dict)

        self.assertEqual(model.dependent_vars, list(model.keys()))


    # @unittest.skip('This might not be wise. What do we expect happens when we negate a model?')
    def test_neg(self):
        """
        Test negation of all model types
        """
        x, y_1, y_2 = variables('x, y_1, y_2')
        a, b = parameters('a, b')

        model_dict = {y_2: a * x ** 2, y_1: 2 * x * b}
        model = Model(model_dict)

        model_neq  = - model
        for key in model:
            self.assertEqual(model[key], - model_neq[key])

        # Constraints
        constraint = Constraint(Eq(a * x, 2), model)

        constraint_neq = - constraint
        # for key in constraint:
        self.assertEqual(constraint[constraint.dependent_vars[0]], - constraint_neq[constraint_neq.dependent_vars[0]])

        # On a constraint we expect the model to stay unchanged, not negated
        self.assertEqual(id(constraint.model), id(model))

        # ODEModel
        odemodel = ODEModel({D(y_1, x): a * x}, initial={a: 1.0})

        odemodel_neq = - odemodel
        for key in odemodel:
            self.assertEqual(odemodel[key], - odemodel_neq[key])

        # On a constraint we expect the model to stay unchanged, not negated
        self.assertEqual(id(constraint.model), id(model))

        # raise NotImplementedError('')

    def test_NumericalModel(self):
        x, y = variables('x, y')
        a, b = parameters('a, b')

        model = CallableModel({y: a * x + b})
        numerical_model = CallableNumericalModel({y: lambda x, a, b: a * x + b}, [x], [a, b])
        self.assertEqual(model.__signature__, numerical_model.__signature__)

        xdata = np.linspace(0, 10)
        ydata = model(x=xdata, a=5.5, b=15.0).y + np.random.normal(0, 1)
        np.testing.assert_almost_equal(
            model(x=xdata, a=5.5, b=15.0),
            numerical_model(x=xdata, a=5.5, b=15.0),
        )

        faulty_model = CallableNumericalModel({y: lambda x, a, b: a * x + b}, [], [a, b])
        self.assertNotEqual(model.__signature__, faulty_model.__signature__)
        with self.assertRaises(TypeError):
            # This is an incorrect signature, even though the lambda function is
            # correct. Should fail.
            faulty_model(xdata, 5.5, 15.0)

        fit = Fit(model, x=xdata, y=ydata)
        analytical_result = fit.execute()
        fit = Fit(numerical_model, x=xdata, y=ydata)
        numerical_result = fit.execute()
        for val1, val2 in zip(analytical_result.params, numerical_result.params):
            self.assertAlmostEqual(val1, val2)

if __name__ == '__main__':
    unittest.main()

