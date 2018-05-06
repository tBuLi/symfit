from __future__ import division, print_function
import unittest

import numpy as np

from symfit import (
    variables, parameters, Fit, Parameter, Variable,
    Equality, Model
)
from symfit.core.minimizers import BFGS, MINPACK, SLSQP, LBFGSB
from symfit.distributions import Gaussian

class TestAutoFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_vector_fitting(self):
        """
        Test the behavior in the presence of bounds or constraints: `Fit` should
        select `ConstrainedNumericalLeastSquares` when bounds or constraints are
        provided, or for vector models in general. For scalar models, use
        `NumericalLeastSquares`.
        """
        a, b = parameters('a, b')
        a_i, = variables('a_i')

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        # Make a new scalar model.
        scalar_model = {a_i: a + b}
        simple_fit = Fit(
            model=scalar_model,
            a_i=xdata[0],
            minimizer=MINPACK
        )
        self.assertIsInstance(simple_fit.minimizer, MINPACK)

        constrained_fit = Fit(
            model=scalar_model,
            a_i=xdata[0],
            constraints=[Equality(a + b, 110)]
        )
        self.assertIsInstance(constrained_fit.minimizer, SLSQP)

        a.min = 0
        a.max = 25
        a.value = 10
        b.min = 80
        b.max = 120
        b.value = 100
        bound_fit = Fit(
            model=scalar_model,
            a_i=xdata[0],
        )
        self.assertIsInstance(bound_fit.minimizer, LBFGSB)

        # Repeat all of the above for the Vector model
        a, b, c = parameters('a, b, c')
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        simple_fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        self.assertIsInstance(simple_fit.minimizer, BFGS)

        constrained_fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
            constraints=[Equality(a + b + c, 180)]
        )
        self.assertIsInstance(constrained_fit.minimizer, SLSQP)

        a.min = 0
        a.max = 25
        a.value = 10
        b.min = 80
        b.max = 120
        b.value = 100
        bound_fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        self.assertIsInstance(bound_fit.minimizer, LBFGSB)

        fit_result = bound_fit.execute()
        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 6)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 6)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 6)

    def test_vector_fitting_bounds(self):
        """
        Tests fitting to a 3 component vector valued function, with bounds.
        """
        a, b, c = parameters('a, b, c')
        a.min = 0
        a.max = 25
        b.min = 0
        b.max = 500
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 4)

    def test_vector_fitting_guess(self):
        """
        Tests fitting to a 3 component vector valued function, with guesses.
        """
        a, b, c = parameters('a, b, c')
        a.value = 10
        b.value = 100
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        xdata = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = Fit(
            model=model,
            a_i=xdata[0],
            b_i=xdata[1],
            c_i=xdata[2],
        )
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(a), np.mean(xdata[0]), 4)
        self.assertAlmostEqual(fit_result.value(b), np.mean(xdata[1]), 4)
        self.assertAlmostEqual(fit_result.value(c), np.mean(xdata[2]), 4)

    def test_global_fitting(self):
        """
        In case of shared parameters between the components of the model, `Fit`
        should automatically use `ConstrainedLeastSquares`.
        :return:
        """
        x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')
        y0, a_1, a_2, b_1, b_2 = parameters('y0, a_1, a_2, b_1, b_2')

        # The following vector valued function links all the equations together
        # as stated in the intro.
        model = Model({
            y_1: a_1 * x_1**2 + b_1 * x_1 + y0,
            y_2: a_2 * x_2**2 + b_2 * x_2 + y0,
        })
        self.assertTrue(model.shared_parameters)

        # Generate data from this model
        xdata1 = np.linspace(0, 10)
        xdata2 = xdata1[::2] # Only every other point.

        ydata1, ydata2 = model(x_1=xdata1, x_2=xdata2, a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
        # Add some noise to make it appear like real data
        np.random.seed(1)
        ydata1 += np.random.normal(0, 2, size=ydata1.shape)
        ydata2 += np.random.normal(0, 2, size=ydata2.shape)

        xdata = [xdata1, xdata2]
        ydata = [ydata1, ydata2]

        # Guesses
        a_1.value = 100
        a_2.value = 50
        b_1.value = 1
        b_2.value = 1
        y0.value = 10

        fit = Fit(
            model, x_1=xdata[0], x_2=xdata[1], y_1=ydata[0], y_2=ydata[1]
        )
        self.assertIsInstance(fit.minimizer, BFGS)

        # The next model does not share parameters, but is still a vector
        model = Model({
            y_1: a_1 * x_1**2 + b_1 * x_1,
            y_2: a_2 * x_2**2 + b_2 * x_2,
        })
        fit = Fit(
            model, x_1=xdata[0], x_2=xdata[1], y_1=ydata[0], y_2=ydata[1]
        )
        self.assertFalse(model.shared_parameters)
        self.assertIsInstance(fit.minimizer, BFGS)

        # Scalar model, still use bfgs.
        model = Model({
            y_1: a_1 * x_1**2 + b_1 * x_1,
        })
        fit = Fit(model, x_1=xdata[0], y_1=ydata[0])
        self.assertFalse(model.shared_parameters)
        self.assertIsInstance(fit.minimizer, BFGS)

    def test_gaussian_2d_fitting(self):
        """
        Tests fitting to a scalar gaussian function with 2 independent
        variables.
        """
        mean = (0.6, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.2**2, 0], [0, 0.1**2]]

        data = np.random.multivariate_normal(mean, cov, 1000000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100,
                                               range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

        x0 = Parameter(value=mean[0], min=0.0, max=1.0)
        sig_x = Parameter(value=0.2, min=0.0, max=0.3)
        y0 = Parameter(value=mean[1], min=0.0, max=1.0)
        sig_y = Parameter(value=0.1, min=0.0, max=0.3)
        A = Parameter(value=np.mean(ydata), min=0.0)
        x = Variable('x')
        y = Variable('y')
        g = Variable('g')

        model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)})
        fit = Fit(model, x=xx, y=yy, g=ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(x0), np.mean(data[:, 0]), 3)
        self.assertAlmostEqual(fit_result.value(y0), np.mean(data[:, 1]), 3)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_x)), np.std(data[:, 0]), 2)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_y)), np.std(data[:, 1]), 2)
        self.assertGreaterEqual(fit_result.r_squared, 0.96)

    def test_gaussian_2d_fitting_background(self):
        """
        Tests fitting to a scalar gaussian function with 2 independent
        variables to data with a background. Added after #149.
        """
        mean = (0.6, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.2**2, 0], [0, 0.1**2]]
        background = 3.0

        data = np.random.multivariate_normal(mean, cov, 500000)
        # print(data.shape)
        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=100,
                                               range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2
        ydata += background # Background

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False, indexing='ij')

        x0 = Parameter(value=1.1 * mean[0], min=0.0, max=1.0)
        sig_x = Parameter(value=1.1 * 0.2, min=0.0, max=0.3)
        y0 = Parameter(value=1.1 * mean[1], min=0.0, max=1.0)
        sig_y = Parameter(value=1.1 * 0.1, min=0.0, max=0.3)
        A = Parameter(value=1.1 * np.mean(ydata), min=0.0)
        b = Parameter(value=1.2 * background, min=0.0)
        x = Variable('x')
        y = Variable('y')
        g = Variable('g')

        model = Model({g: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y) + b})

        # ydata, = model(x=xx, y=yy, x0=mean[0], y0=mean[1], sig_x=np.sqrt(cov[0][0]), sig_y=np.sqrt(cov[1][1]), A=1, b=3.0)
        fit = Fit(model, x=xx, y=yy, g=ydata)
        fit_result = fit.execute()

        self.assertAlmostEqual(fit_result.value(x0) / np.mean(data[:, 0]), 1.0,  2)
        self.assertAlmostEqual(fit_result.value(y0) / np.mean(data[:, 1]), 1.0, 2)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_x)) / np.std(data[:, 0]), 1.0, 2)
        self.assertAlmostEqual(np.abs(fit_result.value(sig_y)) / np.std(data[:, 1]), 1.0, 2)
        self.assertAlmostEqual(background / fit_result.value(b), 1.0, 1)
        self.assertGreaterEqual(fit_result.r_squared / 0.96, 1.0)

if __name__ == '__main__':
    unittest.main()