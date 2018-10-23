from __future__ import division, print_function
import unittest
import warnings
import pickle

import numpy as np

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit,
    Model, FitResults, variables, CallableNumericalModel, Constraint
)
from symfit.core.objectives import (
    VectorLeastSquares, LeastSquares, LogLikelihood, MinimizeModel
)
from symfit.core.fit_results import FitResults

class TestObjectives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_pickle(self):
        """
        Test the picklability of the built-in objectives.
        """
        # Create test data
        xdata = np.linspace(0, 100, 25)  # From 0 to 100 in 100 steps
        a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
        b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
        ydata = a_vec * xdata + b_vec  # Point scattered around the line 5 * x + 105

        # Normal symbolic fit
        a = Parameter('a', value=0, min=0.0, max=1000)
        b = Parameter('b', value=0, min=0.0, max=1000)
        x, y = variables('x, y')
        model = Model({y: a * x + b})

        for objective in [VectorLeastSquares, LeastSquares, LogLikelihood, MinimizeModel]:
            obj = objective(model, data={'x': xdata, 'y': ydata})
            new_obj = pickle.loads(pickle.dumps(obj))
            self.assertTrue(FitResults._array_safe_dict_eq(obj.__dict__,
                                                           new_obj.__dict__))


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
