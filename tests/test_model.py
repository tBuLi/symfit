from __future__ import division, print_function
import unittest
from collections import OrderedDict
import pickle
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

import numpy as np

from symfit import (
    Fit, parameters, variables, Model, ODEModel, D, Eq,
    CallableModel, CallableNumericalModel, Inverse, MatrixSymbol, Symbol, sqrt,
    Function, diff
)
from symfit.core.fit import (
    jacobian_from_model, hessian_from_model, ModelError
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


    def test_neg(self):
        """
        Test negation of all model types
        """
        x, y_1, y_2 = variables('x, y_1, y_2')
        a, b = parameters('a, b')

        model_dict = {y_2: a * x ** 2, y_1: 2 * x * b}
        model = Model(model_dict)

        model_neg = - model
        for key in model:
            self.assertEqual(model[key], - model_neg[key])

        # Constraints
        constraint = Model.as_constraint(Eq(a * x, 2), model)

        constraint_neg = - constraint
        # for key in constraint:
        self.assertEqual(constraint[constraint.dependent_vars[0]], - constraint_neg[constraint_neg.dependent_vars[0]])

        # ODEModel
        odemodel = ODEModel({D(y_1, x): a * x}, initial={a: 1.0})

        odemodel_neg = - odemodel
        for key in odemodel:
            self.assertEqual(odemodel[key], - odemodel_neg[key])

        # For models with interdependency, negation should only change the
        # dependent components.
        model_dict = {x: y_1**2, y_1: a * y_2 + b}
        model = Model(model_dict)

        model_neg = - model
        for key in model:
            if key in model.dependent_vars:
                self.assertEqual(model[key], - model_neg[key])
            elif key in model.interdependent_vars:
                self.assertEqual(model[key], model_neg[key])
            else:
                raise Exception('There should be no such variable')


    def test_CallableNumericalModel(self):
        x, y, z = variables('x, y, z')
        a, b = parameters('a, b')

        model = CallableModel({y: a * x + b})
        numerical_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b}, [x], [a, b]
        )
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
            {y: lambda x, a, b: a * x + b, z: lambda x, a: x ** a},
            connectivity_mapping={y: {a, b, x}, z: {x, a}}
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
        zdata = mixed_model(x=xdata, a=5.5, b=15.0).z + np.random.normal(0, 1)

        # Check if the fits are the same
        fit = Fit(mixed_model, x=xdata, y=ydata, z=zdata)
        mixed_result = fit.execute()
        fit = Fit(numerical_model, x=xdata, y=ydata, z=zdata)
        numerical_result = fit.execute()
        for param in [a, b]:
            self.assertAlmostEqual(
                mixed_result.value(param),
                numerical_result.value(param)
            )
            self.assertAlmostEqual(
                mixed_result.stdev(param),
                numerical_result.stdev(param)
            )
        self.assertAlmostEqual(mixed_result.r_squared, numerical_result.r_squared)

        # Test if the constrained syntax is supported
        fit = Fit(numerical_model, x=xdata, y=ydata, z=zdata, constraints=[Eq(a, b)])
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

    def test_pickle(self):
        """
        Make sure models can be pickled are preserved when pickling
        """
        a, b = parameters('a, b')
        x, y = variables('x, y')
        exact_model = Model({y: a * x ** b})
        constraint = Model.as_constraint(Eq(a, b), exact_model)
        num_model = CallableNumericalModel(
            {y: a * x ** b}, independent_vars=[x], params=[a, b]
        )
        connected_num_model = CallableNumericalModel(
            {y: a * x ** b}, connectivity_mapping={y: {x, a, b}}
        )
        # Test if lsoda args and kwargs are pickled too
        ode_model = ODEModel({D(y, x): a * x + b}, {x: 0.0}, 3, 4, some_kwarg=True)

        models = [exact_model, constraint, num_model, ode_model,
                  connected_num_model]
        for model in models:
            new_model = pickle.loads(pickle.dumps(model))
            # Compare signatures
            self.assertEqual(model.__signature__, new_model.__signature__)
            # Trigger the cached vars because we compare `__dict__` s
            model.vars
            new_model.vars
            # Explicitly make sure the connectivity mapping is identical.
            self.assertEqual(model.connectivity_mapping,
                             new_model.connectivity_mapping)
            if not isinstance(model, ODEModel):
                model.function_dict
                model.vars_as_functions
                new_model.function_dict
                new_model.vars_as_functions
            self.assertEqual(model.__dict__, new_model.__dict__)

    def test_MatrixSymbolModel(self):
        """
        Test a model which is defined by ModelSymbols, see #194
        """
        N = Symbol('N', integer=True)
        M = MatrixSymbol('M', N, N)
        W = MatrixSymbol('W', N, N)
        I = MatrixSymbol('I', N, N)
        y = MatrixSymbol('y', N, 1)
        c = MatrixSymbol('c', N, 1)
        a, b = parameters('a, b')
        z, x = variables('z, x')

        model_dict = {
            W: Inverse(I + M / a ** 2),
            c: - W * y,
            z: sqrt(c.T * c)
        }
        # TODO: This should be a Model in the future, but sympy is not yet
        # capable of computing Matrix derivatives at the time of writing.
        model = CallableModel(model_dict)

        self.assertEqual(model.params, [a])
        self.assertEqual(model.independent_vars, [I, M, y])
        self.assertEqual(model.dependent_vars, [z])
        self.assertEqual(model.interdependent_vars, [W, c])
        self.assertEqual(model.connectivity_mapping,
                         {W: {I, M, a}, c: {W, y}, z: {c}})
        # Generate data
        iden = np.eye(2)
        M_mat = np.array([[2, 1], [3, 4]])
        y_vec = np.array([3, 5])

        eval_model = model(I=iden, M=M_mat, y=y_vec, a=0.1)
        W_manual = np.linalg.inv(iden + M_mat / 0.1 ** 2)
        c_manual = - W_manual.dot(y_vec)
        z_manual = np.atleast_1d(np.sqrt(c_manual.T.dot(c_manual)))
        np.testing.assert_allclose(eval_model.W, W_manual)
        np.testing.assert_allclose(eval_model.c, c_manual)
        np.testing.assert_allclose(eval_model.z, z_manual)

        # Now try to retrieve the value of `a` from a fit
        a.value = 0.2
        fit = Fit(model, z=z_manual, I=iden, M=M_mat, y=y_vec)
        fit_result = fit.execute()
        eval_model = model(I=iden, M=M_mat, y=y_vec, **fit_result.params)
        self.assertAlmostEqual(0.1, np.abs(fit_result.value(a)))
        np.testing.assert_allclose(eval_model.W, W_manual, rtol=1e-5)
        np.testing.assert_allclose(eval_model.c, c_manual, rtol=1e-5)
        np.testing.assert_allclose(eval_model.z, z_manual, rtol=1e-5)

        # TODO: add constraints to Matrix model. But since Matrix expressions
        # can not yet be derived, this needs #154 to be solved first.

    def test_interdependency_invalid(self):
        """
        Create an invalid model with interdependency.
        """
        a, b, c = parameters('a, b, c')
        x, y, z = variables('x, y, z')

        with self.assertRaises(ModelError):
            # Invalid, parameters can not be keys
            model_dict = {
                c: a ** 3 * x + b ** 2,
                z: c ** 2 + a * b
            }
            model = Model(model_dict)
        with self.assertRaises(ModelError):
            # Invalid, parameters can not be keys
            model_dict = {c: a ** 3 * x + b ** 2}
            model = Model(model_dict)


    def test_interdependency(self):
        a, b = parameters('a, b')
        x, y, z = variables('x, y, z')
        model_dict = {
            y: a**3 * x + b**2,
            z: y**2 + a * b
        }
        callable_model = CallableModel(model_dict)
        self.assertEqual(callable_model.independent_vars, [x])
        self.assertEqual(callable_model.interdependent_vars, [y])
        self.assertEqual(callable_model.dependent_vars, [z])
        self.assertEqual(callable_model.params, [a, b])
        self.assertEqual(callable_model.connectivity_mapping,
                         {y: {a, b, x}, z: {a, b, y}})
        np.testing.assert_almost_equal(callable_model(x=3, a=1, b=2),
                                       np.atleast_2d([7, 51]).T)
        for var, func in callable_model.vars_as_functions.items():
            self.assertEqual(
                set(str(x) for x in callable_model.connectivity_mapping[var]),
                set(str(x.__class__) if isinstance(x, Function) else str(x)
                    for x in func.args)
            )

        jac_model = jacobian_from_model(callable_model)
        self.assertEqual(jac_model.params, [a, b])
        self.assertEqual(jac_model.dependent_vars, [D(z, a), D(z, b), z])
        self.assertEqual(jac_model.interdependent_vars, [D(y, a), D(y, b), y])
        self.assertEqual(jac_model.independent_vars, [x])
        for p1, p2 in zip_longest(jac_model.__signature__.parameters, [x, a, b]):
            self.assertEqual(str(p1), str(p2))
        # The connectivity of jac_model should be that from it's own components
        # plus that of the model. The latter is needed to properly compute the
        # Hessian.
        self.assertEqual(
            jac_model.connectivity_mapping,
             {D(y, a): {a, x},
              D(y, b): {b},
              D(z, a): {b, y, D(y, a)},
              D(z, b): {a, y, D(y, b)},
              y: {a, b, x}, z: {a, b, y}
              }
        )
        self.assertEqual(
            jac_model.model_dict,
            {D(y, a): 3 * a**2 * x,
             D(y, b): 2 * b,
             D(z, a): b + 2 * y * D(y, a),
             D(z, b): a + 2 * y * D(y, b),
             y: callable_model[y], z: callable_model[z]
             }
        )
        for var, func in jac_model.vars_as_functions.items():
            self.assertEqual(
                set(x.name for x in jac_model.connectivity_mapping[var]),
                set(str(x.__class__) if isinstance(x, Function) else str(x)
                    for x in func.args)
            )
        hess_model = hessian_from_model(callable_model)
        # Result according to Mathematica
        hess_as_dict = {
            D(y, (a, 2)): 6 * a * x,
            D(y, a, b): 0,
            D(y, b, a): 0,
            D(y, (b, 2)): 2,
            D(z, (a, 2)): 2 * D(y, a)**2 + 2 * y * D(y, (a, 2)),
            D(z, a, b): 1 + 2 * D(y, b) * D(y, a) + 2 * y * D(y, a, b),
            D(z, b, a): 1 + 2 * D(y, b) * D(y, a) + 2 * y * D(y, a, b),
            D(z, (b, 2)): 2 * D(y, b)**2 + 2 * y * D(y, (b, 2)),
            D(y, a): 3 * a ** 2 * x,
            D(y, b): 2 * b,
            D(z, a): b + 2 * y * D(y, a),
            D(z, b): a + 2 * y * D(y, b),
            y: callable_model[y], z: callable_model[z]
        }
        self.assertEqual(len(hess_model), len(hess_as_dict))
        for key, expr in hess_model.items():
            self.assertEqual(expr, hess_as_dict[key])

        self.assertEqual(hess_model.params, [a, b])
        self.assertEqual(
            hess_model.dependent_vars,
            [D(z, (a, 2)), D(z, a, b), D(z, (b, 2)), D(z, b, a),
             D(z, a), D(z, b), z]
        )
        self.assertEqual(hess_model.interdependent_vars,
                         [D(y, (a, 2)), D(y, a), D(y, b), y])
        self.assertEqual(hess_model.independent_vars, [x])


        model = Model(model_dict)
        np.testing.assert_almost_equal(model(x=3, a=1, b=2),
                                       np.atleast_2d([7, 51]).T)
        np.testing.assert_almost_equal(model.eval_jacobian(x=3, a=1, b=2),
                                       np.array([[[9], [4]], [[128], [57]]]))
        np.testing.assert_almost_equal(
            model.eval_hessian(x=3, a=1, b=2),
            np.array([[[[18], [0]], [[0], [2]]],
                           [[[414], [73]], [[73], [60]]]]))

        self.assertEqual(model.__signature__, model.jacobian_model.__signature__)
        self.assertEqual(model.__signature__, model.hessian_model.__signature__)

if __name__ == '__main__':
    unittest.main()

