from __future__ import division, print_function
import unittest
import warnings

import numpy as np

from symfit import (
    MatrixSymbol, Fit, CallableModel, Parameter
)
from symfit.core.linear_solvers import LstSq, LstSqBounds
from symfit.core.models import ModelError
from symfit.core.support import key2str


class TestLinearSolvers(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        A_mat = np.array([[3, 1], [1, 2]])
        y_mat = np.array([[9], [8]])

        L = M = 2
        N = 1

        x = MatrixSymbol(Parameter('x'), M, N)
        A = MatrixSymbol('A', L, M)
        y = MatrixSymbol('y', L, N)

        self.simple_model = CallableModel({y: A * x})
        self.data = {A: A_mat, y: y_mat}

        x = MatrixSymbol(Parameter('x_bounded', min=np.array([[2.1], [2.5]])), M, N)
        A = MatrixSymbol('A', L, M)
        y = MatrixSymbol('y', L, N)

        self.bounded_model = CallableModel({y: A * x})

    def test_unbounded(self):
        for Solver in [LstSq, LstSqBounds]:
            solver = Solver(self.simple_model, data=self.data)
            ans = solver.execute()
            np.testing.assert_almost_equal(ans.params['x'], np.array([[2.], [3.]]))

    def test_fit(self):
        """
        Fit should be able to decide between the minimizers on the fly.
        :return:
        """
        results = {}
        for model, Solver in [(self.simple_model, LstSq),
                              (self.bounded_model, LstSqBounds)]:
            fit = Fit(model, **key2str(self.data))
            self.assertIsInstance(fit.linear_solver, Solver)
            fit_result = fit.execute()
            results[Solver] = fit_result

        np.testing.assert_almost_equal(results[LstSq].params['x'],
                                       np.array([[2.], [3.]]))
        np.testing.assert_almost_equal(results[LstSqBounds].params['x_bounded'],
                                       np.array([[2.1], [2.85]]))

    def test_numpy_lsqtsqbounds(self):
        solver = LstSqBounds(self.bounded_model, data=self.data)
        ans = solver.execute()
        lb, ub = self.bounded_model.bounds[0]
        self.assertTrue(np.all(ans.params['x_bounded'] >= lb))
        self.assertTrue(np.all(ans.params['x_bounded'] < ub))


    def test_nodata(self):
        solver = LstSq(self.simple_model, data={})
        with self.assertRaises(TypeError):
            solver.execute()

    def test_nonlinear_problems(self):
        L = M = 2
        x = MatrixSymbol(Parameter('x'), M, M)
        A = MatrixSymbol('A', L, M)
        y = MatrixSymbol('y', L, M)

        model = CallableModel({y: A * x**2})
        solver = LstSq(model, data={})
        with self.assertRaises(ModelError):
            solver.execute()


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
