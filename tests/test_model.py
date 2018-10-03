from __future__ import division, print_function
import unittest
from collections import OrderedDict

from symfit import (
    Variable, Parameter, Fit, FitResults, LinearLeastSquares, parameters, indices,
    variables, NonLinearLeastSquares, Model, TaylorModel, Constraint, ODEModel, D, Eq
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

    def test_indexed_model(self):
        """
        Test a Model with many indexed and non-indexed parameters and variables.
        """
        x, y, z = variables('x, y, z', indexed=True)
        a, b = parameters('a, b', indexed=True)
        c, d = parameters('c, d')
        i, j = indices('i, j')
        model = Model({
            z[i]: a[i, j] * x[j] + b[i, j] * y[j] + c * x[i]**2 + d * y[i]**2
        })

        self.assertEqual(model.params, [a, b, c, d])
        self.assertEqual(model.indexed_params, [a, b])
        self.assertEqual(model.unindexed_params, [c, d])
        self.assertEqual(model.indices, [i, j])
        self.assertEqual(model.dependent_vars, [z])
        self.assertEqual(model.independent_vars, [x, y])
        # Check the translation to indexed/unindexed
        self.assertEqual(
            [model.symbol2indexed[var] for var in model.independent_vars],
            [x[j], y[j]]
        )
        self.assertEqual(
            [model.symbol2indexed[var] for var in model.dependent_vars],
            [z[i]]
        )
        self.assertEqual(
            [model.symbol2indexed[var] for var in model.params],
            [a[i, j], b[i, j], c, d]
        )

if __name__ == '__main__':
    unittest.main()
