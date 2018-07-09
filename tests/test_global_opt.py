# -*- coding: utf-8 -*-
from __future__ import division, print_function
import unittest

import numpy as np

from symfit import (
    variables, parameters, Fit, Parameter, Variable,
    Equality, Model
)
from symfit.core.minimizers import BFGS, DifferentialEvolution
from symfit.distributions import Gaussian

class TestGlobalOpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        mean = (0.4, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.01**2, 0], [0, 0.01**2]]
        data = np.random.multivariate_normal(mean, cov, 2500000)
        print(np.mean(data, axis=0))

        # Insert them as y,x here as np fucks up cartesian conventions.
        cls.ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=200,
                                               range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        cls.xx, cls.yy = np.meshgrid(xcentres, ycentres, sparse=False)
        # xdata = np.dstack((xx, yy)).T

    def setUp(self):
        x = Variable('x')
        y = Variable('y')
        xmin, xmax = -5, 5
        self.x0_1 = Parameter('x01', value=0, min=xmin, max=xmax)
        self.sig_x_1 = Parameter('sigx1', value=0, min=0.0, max=1)
        self.y0_1 = Parameter('y01', value=0, min=xmin, max=xmax)
        self.sig_y_1 = Parameter('sigy1', value=0, min=0.0, max=1)
        self.A_1 = Parameter('A1', min=0, max=1000)
        g_1 = self.A_1 * Gaussian(x, self.x0_1, self.sig_x_1) *\
                Gaussian(y, self.y0_1, self.sig_y_1)

        self.model = Model(g_1)
    
    def test_diff_evo(self):
        """
        Tests fitting to a scalar gaussian with 2 independent variables with
        wide bounds.
        """
        
        fit = Fit(self.model, self.xx, self.yy, self.ydata, minimizer=BFGS)
        fit_result = fit.execute()

        self.assertIsInstance(fit.minimizer, BFGS)

        # Make sure a local optimizer doesn't find the answer.
        self.assertNotAlmostEqual(fit_result.value(self.x0_1), 0.4, 1)
        self.assertNotAlmostEqual(fit_result.value(self.y0_1), 0.4, 1)

        # On to the main event
        fit = Fit(self.model, self.xx, self.yy, self.ydata,
                  minimizer=DifferentialEvolution)
        fit_result = fit.execute(polish=True, seed=0, tol=1e-4, maxiter=50)
        print(fit_result)
        # Global minimizers are really bad at finding local minima though, so
        # roughly equal is good enough.
        self.assertAlmostEqual(fit_result.value(self.x0_1), 0.4, 1)
        self.assertAlmostEqual(fit_result.value(self.y0_1), 0.4, 1)

    def test_chained_min(self):
        """Test fitting with a chained minimizer"""
        curvals = [p.value for p in self.model.params]
        fit = Fit(self.model, self.xx, self.yy, self.ydata,
                  minimizer=[DifferentialEvolution, BFGS])
        fit_result = fit.execute(minimizer_kwargs=[dict(seed=0, tol=1e-4, maxiter=10), {}])
        print(fit_result)
        self.assertAlmostEqual(fit_result.value(self.x0_1), 0.4, 4)
        self.assertAlmostEqual(fit_result.value(self.y0_1), 0.4, 4)
        self.assertEqual(curvals, [p.value for p in self.model.params])


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