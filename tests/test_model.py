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

    def test_CallableNumericalModel(self):
        x, y, z = variables('x, y, z')
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

        faulty_model = CallableNumericalModel({y: lambda x, a, b: a * x + b},
                                              [], [a, b])
        self.assertNotEqual(model.__signature__, faulty_model.__signature__)
        with self.assertRaises(TypeError):
            # This is an incorrect signature, even though the lambda function is
            # correct. Should fail.
            faulty_model(xdata, 5.5, 15.0)

        # Faulty model whose components do not all accept all of the args
        faulty_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b, z: lambda x, a: x**a}, [x], [a, b]
        )
        self.assertEqual(model.__signature__, faulty_model.__signature__)
        with self.assertRaises(TypeError):
            # Lambda got an unexpected keyword 'b'
            faulty_model(xdata, 5.5, 15.0)

        # Faulty model with a wrongly named argument
        faulty_model = CallableNumericalModel(
            {y: lambda x, a, c=5: a * x + c}, [x], [a, b]
        )
        self.assertEqual(model.__signature__, faulty_model.__signature__)
        with self.assertRaises(TypeError):
            # Lambda got an unexpected keyword 'b'
            faulty_model(xdata, 5.5, 15.0)


        # Correct version of the previous model
        numerical_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b, z: lambda x, a, b: x**a}, [x], [a, b]
        )
        # Correct version of the previous model
        mixed_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b, z: x ** a}, [x],
            [a, b]
        )
        np.testing.assert_almost_equal(
            numerical_model(x=xdata, a=5.5, b=15.0),
            mixed_model(x=xdata, a=5.5, b=15.0)
        )

        # Check if the fits are the same
        fit = Fit(model, x=xdata, y=ydata)
        analytical_result = fit.execute()
        fit = Fit(numerical_model, x=xdata, y=ydata)
        numerical_result = fit.execute()
        for param in [a, b]:
            self.assertAlmostEqual(
                analytical_result.value(param),
                numerical_result.value(param)
            )
            self.assertAlmostEqual(
                analytical_result.stdev(param),
                numerical_result.stdev(param)
            )
        self.assertAlmostEqual(analytical_result.r_squared, numerical_result.r_squared)

        # Test if the constrained syntax is supported
        fit = Fit(numerical_model, x=xdata, y=ydata, constraints=[Eq(a, b)])
        constrained_result = fit.execute()
        self.assertAlmostEqual(constrained_result.value(a), constrained_result.value(b))

    def test_CallableNumericalModel2D(self):
        """
        Apply a CallableNumericalModel to 2D data, to see if it is
        agnostic to data shape.
        """
        shape = (30, 40)

        def function(a, b):
            out = np.ones(shape) * a
            out[15:, :] += b
            return out

        a, b = parameters('a, b')
        y, = variables('y')

        model = CallableNumericalModel({y: function}, [], [a, b])
        data = 15 * np.ones(shape)
        data[15:, :] += 20

        fit = Fit(model, y=data)
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.value(a), 15)
        self.assertAlmostEqual(fit_result.value(b), 20)

        def flattened_function(a, b):
            out = np.ones(shape) * a
            out[15:, :] += b
            return out.flatten()

        model = CallableNumericalModel({y: flattened_function}, [], [a, b])
        data = 15 * np.ones(shape)
        data[15:, :] += 20
        data = data.flatten()

        fit = Fit(model, y=data)
        flat_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), flat_result.value(a))
        self.assertAlmostEqual(fit_result.value(b), flat_result.value(b))
        self.assertAlmostEqual(fit_result.stdev(a), flat_result.stdev(a))
        self.assertAlmostEqual(fit_result.stdev(b), flat_result.stdev(b))
        self.assertAlmostEqual(fit_result.r_squared, flat_result.r_squared)

if __name__ == '__main__':
    unittest.main()

