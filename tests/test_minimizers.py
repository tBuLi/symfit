from __future__ import division, print_function
import unittest
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit,
    Model, FitResults, variables
)
from symfit.core.objectives import MinimizeModel
from symfit.core.minimizers import BFGS, Powell


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
