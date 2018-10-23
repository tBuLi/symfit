from __future__ import division, print_function
import unittest
import sys
import warnings

import numpy as np
import pickle
import multiprocessing as mp

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit,
    Model, FitResults, variables, CallableNumericalModel, Constraint
)
from symfit.core.minimizers import *
from symfit.core.support import partial

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
        x = Variable()
        model = a * x + b

        fit = Fit(model, xdata, ydata, minimizer=BFGS)
        fit_result = fit.execute()

        def f(x, a, b):
            return a * x + b

        def chi_squared(a, b):
            return np.sum((ydata - f(xdata, a, b))**2)

        fit_custom = BFGS(chi_squared, [a, b])
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

    def test_pickle(self):
        """
        Test the picklability of the different minimizers.
        """
        # Create test data
        xdata = np.linspace(0, 100, 2)  # From 0 to 100 in 100 steps
        a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
        b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
        ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

        # Normal symbolic fit
        a = Parameter('a', value=0, min=0.0, max=1000)
        b = Parameter('b', value=0, min=0.0, max=1000)

        # Make a set of all ScipyMinimizers, and add a chained minimizer.
        scipy_minimizers = subclasses(ScipyMinimize)
        chained_minimizer = partial(ChainedMinimizer,
                                    minimizers=[DifferentialEvolution, BFGS])
        scipy_minimizers.add(chained_minimizer)
        constrained_minimizers = subclasses(ScipyConstrainedMinimize)
        # Test for all of them if they can be pickled.
        for minimizer in scipy_minimizers:
            if minimizer is MINPACK:
                fit = minimizer(
                    partial(chi_squared, x=xdata, y=ydata, sum=False),
                    [a, b]
                )
            elif minimizer in constrained_minimizers:
                # For constraint minimizers we also add a constraint, just to be
                # sure constraints are treated well.
                dummy_model = CallableNumericalModel({}, independent_vars=[], params=[a, b])
                fit = minimizer(
                    partial(chi_squared, x=xdata, y=ydata),
                    [a, b],
                    constraints=[Constraint(Ge(b, a), model=dummy_model)]
                )
            elif isinstance(minimizer, partial) and issubclass(minimizer.func, ChainedMinimizer):
                init_minimizers = []
                for sub_minimizer in minimizer.keywords['minimizers']:
                    init_minimizers.append(sub_minimizer(
                        partial(chi_squared, x=xdata, y=ydata),
                        [a, b]
                    ))
                minimizer.keywords['minimizers'] = init_minimizers
                fit = minimizer(partial(chi_squared, x=xdata, y=ydata), [a, b])
            else:
                fit = minimizer(partial(chi_squared, x=xdata, y=ydata), [a, b])

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
            self.assertEqual(res_before, res_after)

    def test_multiprocessing(self):
        """
        To make sure pickling truly works, try multiprocessing. No news is good
        news.
        """
        np.random.seed(2)
        x = np.arange(100, dtype=float)
        y = x + 0.25 * x * np.random.rand(100)
        a_values = np.arange(3) + 1
        np.random.shuffle(a_values)

        def gen_fit_objs(x, y, a, minimizer):
            for a_i in a:
                a_par = Parameter('a', 5, min=0.0, max=20)
                b_par = Parameter('b', 1, min=0.0, max=2)
                x_var = Variable('x')
                y_var = Variable('y')

                model = CallableNumericalModel({y_var: f}, [x_var], [a_par, b_par])

                fit = Fit(model, x, a_i * y + 1, minimizer=minimizer)
                yield fit

        minimizers = subclasses(ScipyMinimize)
        chained_minimizer = (DifferentialEvolution, BFGS)
        minimizers.add(chained_minimizer)

        all_results = {}
        pool = mp.Pool()
        for minimizer in minimizers:
            results = pool.map(worker, gen_fit_objs(x, y, a_values, minimizer))
            all_results[minimizer] = [res.params['a'] for res in results]


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
