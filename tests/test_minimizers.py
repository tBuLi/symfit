from __future__ import division, print_function
import unittest
import sys
import warnings

import numpy as np
import pickle
import multiprocessing as mp

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit,
    Model, FitResults, variables, CallableNumericalModel
)
from symfit.core.minimizers import *
from symfit.core.objectives import LeastSquares, MinimizeModel, VectorLeastSquares

# Defined at the global level because local functions can't be pickled.
def f(x, a, b):
    return a * x + b

def chi_squared(x, y, a, b, sum=True):
    if sum:
        return np.sum((y - f(x, a, b)) ** 2)
    else:
        return (y - f(x, a, b)) ** 2

def worker(fit_obj):
    return fit_obj.execute()

class SqrtLeastSquares(LeastSquares):
    """
    Minimizes the square root of LeastSquares. This seems to help SLSQP in
    particular, and is considered good practise since the square can grow to
    rapidly, leading to numerical errors.
    """
    # TODO: Make this a standard objective, and perhaps even THE standard
    # objective. This lightweight version is given without proper testing
    # because only the call is relevant, and this makes our multiprocessing test
    # work.
    def __call__(self, *args, **kwargs):
        chi2 = super(SqrtLeastSquares, self).__call__(*args, **kwargs)
        return np.sqrt(chi2)

    def eval_jacobian(self, *args, **kwargs):
        sqrt_chi2 = self(*args, **kwargs)
        chi2_jac = super(SqrtLeastSquares, self).eval_jacobian(*args, **kwargs)
        return 0.5 * (1 / sqrt_chi2) * chi2_jac

    def eval_hessian(self, *args, **kwargs):
        sqrt_chi2 = self(*args, **kwargs)
        sqrt_chi2_jac = self.eval_jacobian(*args, **kwargs)
        chi2 = super(SqrtLeastSquares, self).__call__(*args, **kwargs)
        chi2_jac = super(SqrtLeastSquares, self).eval_jacobian(*args, **kwargs)
        chi2_hess = super(SqrtLeastSquares, self).eval_hessian(*args, **kwargs)
        return - 0.5 * (1 / chi2) * np.outer(sqrt_chi2_jac, chi2_jac) \
               + 0.5 * (1 / sqrt_chi2) * chi2_hess

def subclasses(base, leaves_only=True):
    """
    Recursively create a set of subclasses of ``object``.

    :param object: Class
    :param leaves_only: If ``True``, return only the leaves of the subclass tree
    :return: (All leaves of) the subclass tree.
    """
    base_subs = set(base.__subclasses__())
    if not base_subs or not leaves_only:
        all_subs = {base}
    else:
        all_subs = set()
    for sub in list(base_subs):
        sub_subs = subclasses(sub, leaves_only=leaves_only)
        all_subs.update(sub_subs)
    return all_subs

class TestMinimize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_custom_objective(self):
        """
        Compare the result of a custom objective with the symbolic result.
        :return:
        """
        # Create test data
        xdata = np.linspace(0, 100, 25)  # From 0 to 100 in 100 steps
        a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
        b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
        ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

        # Normal symbolic fit
        a = Parameter('a', value=0, min=0.0, max=1000)
        b = Parameter('b', value=0, min=0.0, max=1000)
        x = Variable('x')
        y = Variable('y')
        model = a * x + b

        fit = Fit(model, xdata, ydata, minimizer=BFGS)
        fit_result = fit.execute()

        def f(x, a, b):
            return a * x + b

        def chi_squared(a, b):
            return np.sum((ydata - f(xdata, a, b))**2)

        with warnings.catch_warnings(record=True) as w:
            # Should no longer raise warnings, because internally we practice
            # what we preach.
            warnings.simplefilter("always")
            fit_custom = BFGS(chi_squared, [a, b])
            self.assertTrue(len(w) == 0)

        fit_custom_result = fit_custom.execute()

        self.assertIsInstance(fit_custom_result, FitResults)
        self.assertAlmostEqual(fit_custom_result.value(a) / fit_result.value(a), 1.0, 5)
        self.assertAlmostEqual(fit_custom_result.value(b) / fit_result.value(b), 1.0, 4)

        # New preferred usage, multi component friendly.
        with self.assertRaises(TypeError):
            callable_model = CallableNumericalModel(
                chi_squared,
                connectivity_mapping={y: {a, b}}
            )
        callable_model = CallableNumericalModel(
            {y: chi_squared},
            connectivity_mapping={y: {a, b}}
        )
        self.assertEqual(callable_model.params, [a, b])
        self.assertEqual(callable_model.independent_vars, [])
        self.assertEqual(callable_model.dependent_vars, [y])
        self.assertEqual(callable_model.interdependent_vars, [])
        self.assertEqual(callable_model.connectivity_mapping, {y: {a, b}})
        fit_custom = BFGS(callable_model, [a, b])
        fit_custom_result = fit_custom.execute()

        self.assertIsInstance(fit_custom_result, FitResults)
        self.assertAlmostEqual(fit_custom_result.value(a) / fit_result.value(a), 1.0, 5)
        self.assertAlmostEqual(fit_custom_result.value(b) / fit_result.value(b), 1.0, 4)

    def test_custom_parameter_names(self):
        """
        For cusom objective functions you still have to provide a list of Parameter
        objects to use with the same name as the keyword arguments to your function.
        """
        a = Parameter()
        c = Parameter()

        def chi_squared(a, b):
            """
            Dummy function with different keyword argument names
            """
            pass

        fit_custom = BFGS(chi_squared, [a, c])
        with self.assertRaises(TypeError):
            fit_custom.execute()

    def test_powell(self):
        """
        Powell with a single parameter gave an error because a 0-d array was
        returned by scipy. So no error here is winning.
        """
        x, y = variables('x, y')
        a, b = parameters('a, b')
        b.fixed = True

        model = Model({y: a * x + b})
        xdata = np.linspace(0, 10)
        ydata = model(x=xdata, a=5.5, b=15.0).y + np.random.normal(0, 1)
        fit = Fit({y: a * x + b}, x=xdata, y=ydata, minimizer=Powell)
        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.value(b), 1.0)

    def test_jac_hess(self):
        """
        Make sure both the Jacobian and Hessian are passed to the minimizer.
        """
        x, y = variables('x, y')
        a, b = parameters('a, b')
        b.fixed = True

        model = Model({y: a * x + b})
        xdata = np.linspace(0, 10)
        ydata = model(x=xdata, a=5.5, b=15.0).y + np.random.normal(0, 1)
        fit = Fit({y: a * x + b}, x=xdata, y=ydata, minimizer=TrustConstr)
        self.assertIsInstance(fit.minimizer.objective, LeastSquares)
        self.assertIsInstance(fit.minimizer.jacobian.__self__, LeastSquares)
        self.assertIsInstance(fit.minimizer.hessian.__self__, LeastSquares)

        fit_result = fit.execute()
        self.assertAlmostEqual(fit_result.value(b), 1.0)

    def test_pickle(self):
        """
        Test the picklability of the different minimizers.
        """
        # Create test data
        xdata = np.linspace(0, 100, 100)  # From 0 to 100 in 100 steps
        a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
        b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
        ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

        # Normal symbolic fit
        a = Parameter('a', value=0, min=0.0, max=1000)
        b = Parameter('b', value=0, min=0.0, max=1000)
        x, y = variables('x, y')

        # Make a set of all ScipyMinimizers, and add a chained minimizer.
        scipy_minimizers = list(subclasses(ScipyMinimize))
        chained_minimizer = (DifferentialEvolution, BFGS)
        scipy_minimizers.append(chained_minimizer)
        constrained_minimizers = subclasses(ScipyConstrainedMinimize)
        # Test for all of them if they can be pickled.
        for minimizer in scipy_minimizers:
            if minimizer in constrained_minimizers:
                constraints = [Ge(b, a)]
            else:
                constraints = []
            model = CallableNumericalModel(
                {y: f},
                independent_vars=[x], params=[a, b]
            )
            fit = Fit(model, x=xdata, y=ydata, minimizer=minimizer,
                      constraints=constraints)
            if minimizer is not MINPACK:
                self.assertIsInstance(fit.objective, LeastSquares)
                self.assertIsInstance(fit.minimizer.objective, LeastSquares)
            else:
                self.assertIsInstance(fit.objective, VectorLeastSquares)
                self.assertIsInstance(fit.minimizer.objective, VectorLeastSquares)

            fit = fit.minimizer  # Just check if the minimizer pickles
            dump = pickle.dumps(fit)
            pickled_fit = pickle.loads(dump)
            problematic_attr = [
                'objective', '_pickle_kwargs', 'wrapped_objective',
                'constraints', 'wrapped_constraints',
                'local_minimizer', 'minimizers'
            ]

            for key, value in fit.__dict__.items():
                new_value = pickled_fit.__dict__[key]
                try:
                    self.assertEqual(value, new_value)
                except AssertionError as err:
                    if key in problematic_attr:
                        # These attr are new instances, and therefore do not
                        # pass an equality test. All we can do is see if they
                        # are at least the same type.
                        if isinstance(value, (list, tuple)):
                            for val1, val2 in zip(value, new_value):
                                self.assertTrue(isinstance(val1, val2.__class__))
                                if key == 'constraints':
                                    self.assertEqual(val1.model.constraint_type,
                                                     val2.model.constraint_type)
                                    self.assertEqual(
                                        list(val1.model.model_dict.values())[0],
                                        list(val2.model.model_dict.values())[0]
                                    )
                                    self.assertEqual(val1.model.independent_vars,
                                                     val2.model.independent_vars)
                                    self.assertEqual(val1.model.params,
                                                     val2.model.params)
                                    self.assertEqual(val1.model.__signature__,
                                                     val2.model.__signature__)
                                elif key == 'wrapped_constraints':
                                    if isinstance(val1, dict):
                                        self.assertEqual(val1['type'],
                                                         val2['type'])
                                        self.assertEqual(set(val1.keys()),
                                                         set(val2.keys()))
                                    elif isinstance(val1, NonlinearConstraint):
                                        # For trust-ncg we manually check if
                                        # their dicts are equal, because no
                                        # __eq__ is implemented on
                                        # NonLinearConstraint
                                        self.assertEqual(len(val1.__dict__),
                                                         len(val2.__dict__))
                                        for key in val1.__dict__:
                                            try:
                                                self.assertEqual(
                                                    val1.__dict__[key],
                                                    val2.__dict__[key]
                                                )
                                            except AssertionError:
                                                self.assertIsInstance(
                                                    val1.__dict__[key],
                                                    val2.__dict__[key].__class__
                                                )
                                    else:
                                        raise NotImplementedError(
                                            'No such constraint type is known.'
                                        )
                        elif key == '_pickle_kwargs':
                            FitResults._array_safe_dict_eq(value, new_value)
                        else:
                            self.assertTrue(isinstance(new_value, value.__class__))
                    else:
                        raise err
            self.assertEqual(set(fit.__dict__.keys()),
                             set(pickled_fit.__dict__.keys()))

            # Test if we converge to the same result.
            np.random.seed(2)
            res_before = fit.execute()
            np.random.seed(2)
            res_after = pickled_fit.execute()
            self.assertTrue(FitResults._array_safe_dict_eq(res_before.__dict__,
                                                           res_after.__dict__))

    def test_multiprocessing(self):
        """
        To make sure pickling truly works, try multiprocessing. No news is good
        news.
        """
        np.random.seed(2)
        x = np.arange(100, dtype=float)
        a_values = np.array([1, 2, 3])
        np.random.shuffle(a_values)

        def gen_fit_objs(x, a, minimizer):
            """Generates linear fits with different a parameter values."""
            for a_i in a:
                a_par = Parameter('a', 4.0, min=0.0, max=20)
                b_par = Parameter('b', 1.2, min=0.0, max=2)
                x_var = Variable('x')
                y_var = Variable('y')

                model = CallableNumericalModel({y_var: f}, [x_var], [a_par, b_par])

                fit = Fit(
                    model, x, a_i * x + 1, minimizer=minimizer,
                    objective=SqrtLeastSquares if minimizer is not MINPACK else VectorLeastSquares
                )
                yield fit

        minimizers = subclasses(ScipyMinimize)
        chained_minimizer = (DifferentialEvolution, BFGS)
        minimizers.add(chained_minimizer)

        pool = mp.Pool()
        for minimizer in minimizers:
            results = pool.map(worker, gen_fit_objs(x, a_values, minimizer))
            a_results = [res.params['a'] for res in results]
            minimizer_results = [res.minimizer for res in results]
            # Check the results
            np.testing.assert_almost_equal(a_values, a_results, decimal=2)
            for result in results:
                # Check that we are actually using the right minimizer
                if isinstance(result.minimizer, ChainedMinimizer):
                    for used, target in zip(result.minimizer.minimizers, minimizer):
                        self.assertIsInstance(used, target)
                else:
                    self.assertIsInstance(result.minimizer, minimizer)
                self.assertIsInstance(result.iterations, int)

    def test_minimizer_constraint_compatibility(self):
        """
        Test if #156 has been solved, and test all the other constraint styles.
        """
        x, y, z = variables('x, y, z')
        a, b, c = parameters('a, b, c')
        b.fixed = True

        model = Model({z: a * x**2 - b * y**2 + c})
        # Generate data, z has to be scalar for MinimizeModel to be happy
        xdata = 3 #np.linspace(0, 10)
        ydata = 5 # np.linspace(0, 10)
        zdata = model(a=2, b=3, c=5, x=xdata, y=ydata).z
        data_dict = {x: xdata, y: ydata, z: zdata}

        # Equivalent ways of defining the same constraint
        constraint_model = Model.as_constraint(a - c, model, constraint_type=Eq)
        constraint_model.params = model.params
        constraints = [
            Eq(a, c),
            MinimizeModel(constraint_model, data=data_dict),
            constraint_model
        ]

        objective = MinimizeModel(model, data=data_dict)
        for constraint in constraints:
            fit = SLSQP(objective, parameters=[a, b, c],
                        constraints=[constraint])
            wrapped_constr = fit.wrapped_constraints[0]['fun'].model
            self.assertIsInstance(wrapped_constr, Model)
            self.assertEqual(wrapped_constr.params, model.params)
            self.assertEqual(wrapped_constr.jacobian_model.params, model.params)
            self.assertEqual(wrapped_constr.hessian_model.params, model.params)
            # Set the data for the dependent var of the constraint to None
            # Normally this is handled by Fit because here we interact with the
            # Minimizer directly, it is up to us.
            constraint_var = fit.wrapped_constraints[0]['fun'].model.dependent_vars[0]
            objective.data[constraint_var] = None
            fit.execute()

        # No scipy style dicts allowed.
        with self.assertRaises(TypeError):
            fit = SLSQP(MinimizeModel(model, data=data_dict),
                        parameters=[a, b, c],
                        constraints=[
                            {'type': 'eq', 'fun': lambda a, b, c: a - c}
                        ]
            )

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
