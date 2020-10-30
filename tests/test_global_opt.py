# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
from __future__ import division, print_function
import pytest
import sys

import numpy as np

from symfit import (
    Fit, Parameter, Variable, Model, GradientModel
)
from symfit.core.minimizers import BFGS, DifferentialEvolution
from symfit.distributions import Gaussian

if sys.version_info >= (3, 0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


class TestGlobalOptGaussian:
    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        mean = (0.4, 0.4)  # x, y mean 0.6, 0.4
        cov = [[0.01**2, 0], [0, 0.01**2]]

        # TODO: evaluate gaussian at 200x200 points (?!) and add appropriate noise
        data = np.random.multivariate_normal(mean, cov, 2500000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        cls.ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=200,
                                                   range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        cls.xx, cls.yy = np.meshgrid(xcentres, ycentres, sparse=False)
        # xdata = np.dstack((xx, yy)).T

    def setup_method(self):
        x = Variable('x')
        y = Variable('y')
        xmin, xmax = -5, 5
        self.x0_1 = Parameter('x01', value=0, min=xmin, max=xmax)
        self.sig_x_1 = Parameter('sigx1', value=0, min=0.0, max=1)
        self.y0_1 = Parameter('y01', value=0, min=xmin, max=xmax)
        self.sig_y_1 = Parameter('sigy1', value=0, min=0.0, max=1)
        self.A_1 = Parameter('A1', min=0, max=1000)
        g_1 = self.A_1 * Gaussian(x, self.x0_1, self.sig_x_1) * Gaussian(y, self.y0_1, self.sig_y_1)

        self.model = GradientModel(g_1)

    def test_diff_evo(self):
        """
        Tests fitting to a scalar gaussian with 2 independent variables with
        wide bounds.
        """

        fit = Fit(self.model, self.xx, self.yy, self.ydata, minimizer=BFGS)
        fit_result = fit.execute()

        assert isinstance(fit.minimizer, BFGS)

        # Make sure a local optimizer doesn't find the answer.
        assert not fit_result.value(self.x0_1) == pytest.approx(0.4, 1e-1)
        assert not fit_result.value(self.y0_1) == pytest.approx(0.4, 1e-1)

        # On to the main event
        fit = Fit(self.model, self.xx, self.yy, self.ydata,
                  minimizer=DifferentialEvolution)
        fit_result = fit.execute(polish=True, seed=0, tol=1e-4, maxiter=100)
        # Global minimizers are really bad at finding local minima though, so
        # roughly equal is good enough.
        assert fit_result.value(self.x0_1) == pytest.approx(0.4, 1e-1)
        assert fit_result.value(self.y0_1) == pytest.approx(0.4, 1e-1)

    def test_chained_min(self):
        """Test fitting with a chained minimizer"""
        curvals = [p.value for p in self.model.params]
        fit = Fit(self.model, self.xx, self.yy, self.ydata,
                  minimizer=[DifferentialEvolution, BFGS])
        fit_result = fit.execute(
            DifferentialEvolution={'seed': 0, 'tol': 1e-4, 'maxiter': 10}
        )
        assert fit_result.value(self.x0_1) == pytest.approx(0.4, 1e-4)
        assert fit_result.value(self.y0_1) == pytest.approx(0.4, 1e-4)
        assert curvals == [p.value for p in self.model.params]

    def test_chained_min_signature(self):
        """
        Test the automatic generation of the signature for ChainedMinimizer
        """
        minimizers = [
            BFGS, DifferentialEvolution, BFGS, DifferentialEvolution, BFGS
        ]

        fit = Fit(self.model, self.xx, self.yy, self.ydata,
                  minimizer=minimizers)

        names = [
            'BFGS', 'DifferentialEvolution', 'BFGS_2',
            'DifferentialEvolution_2', 'BFGS_3'
        ]
        for name, param_name in zip(names, fit.minimizer.__signature__.parameters):
            assert name == param_name
        # Check for equal lengths because zip is slippery that way
        assert len(names) == len(fit.minimizer.__signature__.parameters)

        for param in fit.minimizer.__signature__.parameters.values():
            assert param.kind == inspect_sig.Parameter.KEYWORD_ONLY
        # Make sure keywords end up at the right minimizer.
        with pytest.raises(TypeError):
            # This is not a valid kwarg to DiffEvo, but it is to BFGS. Check if
            # we really go by name of the Minimizer, not by order.
            fit.execute(DifferentialEvolution={'return_all': False})


def test_mexican_hat():
    """
    Test that global minimisation finds the global minima, and doesn't
    affect the value of parameters.
    """
    x = Parameter('x')
    x.min, x.max = -100, 100
    x.value = -2.5
    y = Variable('y')

    model = Model({y: x**4 - 10 * x**2 - x})  # Skewed Mexican hat
    fit = Fit(model, minimizer=[DifferentialEvolution, BFGS])
    fit_result1 = fit.execute(DifferentialEvolution={'seed': 0})

    fit = Fit(model)
    fit_result2 = fit.execute()

    assert fit_result1.value(x) > 0
    assert fit_result2.value(x) < 0
