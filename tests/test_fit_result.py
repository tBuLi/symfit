# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import pytest
import pickle
from collections import OrderedDict

import numpy as np
from symfit import (
    Variable, Parameter, Fit, FitResults, Eq, Ge, CallableNumericalModel, Model
)
from symfit.distributions import BivariateGaussian
from symfit.core.minimizers import (
    BaseMinimizer, MINPACK, BFGS, NelderMead, ChainedMinimizer, BasinHopping
)
from symfit.core.objectives import (
    LogLikelihood, LeastSquares, VectorLeastSquares, MinimizeModel
)


def ge_constraint(a):  # Has to be in the global namespace for pickle.
    return a - 1


class TestTestResult():

    @classmethod
    def setup_class(cls):
        xdata = np.linspace(1, 10, 10)
        ydata = 3 * xdata ** 2

        cls.a = Parameter('a')
        cls.b = Parameter('b')

        x = Variable('x')
        y = Variable('y')
        model = Model({y: cls.a * x ** cls.b})

        fit = Fit(model, x=xdata, y=ydata)
        cls.fit_result = fit.execute()
        fit = Fit(model, x=xdata, y=ydata, minimizer=MINPACK)
        cls.minpack_result = fit.execute()
        fit = Fit(model, x=xdata, objective=LogLikelihood)
        cls.likelihood_result = fit.execute()
        fit = Fit(model, x=xdata, y=ydata, minimizer=[BFGS, NelderMead])
        cls.chained_result = fit.execute()

        z = Variable('z')
        constraints = [
            Eq(cls.a, cls.b),
            CallableNumericalModel.as_constraint(
                {z: ge_constraint}, connectivity_mapping={z: {cls.a}},
                constraint_type=Ge, model=model
            )
        ]
        fit = Fit(model, x=xdata, y=ydata, constraints=constraints)
        cls.constrained_result = fit.execute()
        fit = Fit(model, x=xdata, y=ydata, constraints=constraints,
                  minimizer=BasinHopping)
        cls.constrained_basinhopping_result = fit.execute()

    def test_params_type(self):
        assert isinstance(self.fit_result.params, OrderedDict)

    def test_minimizer_output_type(self):
        assert isinstance(self.fit_result.minimizer_output, dict)
        assert isinstance(self.minpack_result.minimizer_output, dict)
        assert isinstance(self.likelihood_result.minimizer_output, dict)

    def test_fitting(self):
        """
        Test if the fitting worked in the first place.
        """
        assert isinstance(self.fit_result, FitResults)
        assert self.fit_result.value(self.a) == pytest.approx(3.0)
        assert self.fit_result.value(self.b) == pytest.approx(2.0)

        assert isinstance(self.fit_result.stdev(self.a), float)
        assert isinstance(self.fit_result.stdev(self.b), float)

        assert isinstance(self.fit_result.r_squared, float)
        # by definition since there's no fuzzyness
        assert self.fit_result.r_squared == 1.0

    def test_fitting_2(self):
        np.random.seed(43)
        mean = (0.62, 0.71)  # x, y mean 0.7, 0.7
        cov = [
            [0.102**2, 0],
            [0, 0.07**2]
        ]
        data_1 = np.random.multivariate_normal(mean, cov, 10**5)
        mean = (0.33, 0.28)  # x, y mean 0.3, 0.3
        cov = [  # rho = 0.25
            [0.05 ** 2, 0.25 * 0.05 * 0.101],
            [0.25 * 0.05 * 0.101, 0.101 ** 2]
        ]
        data_2 = np.random.multivariate_normal(mean, cov, 10**5)
        data = np.vstack((data_1, data_2))

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:, 1], data[:, 0], bins=200,
                                               range=[[0.0, 1.0], [0.0, 1.0]],
                                               density=True)
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)

        x = Variable('x')
        y = Variable('y')

        x0_1 = Parameter('x0_1', value=0.6, min=0.5, max=0.7)
        sig_x_1 = Parameter('sig_x_1', value=0.1, min=0.0, max=0.2)
        y0_1 = Parameter('y0_1', value=0.7, min=0.6, max=0.8)
        sig_y_1 = Parameter('sig_y_1', value=0.05, min=0.0, max=0.2)
        rho_1 = Parameter('rho_1', value=0.0, min=-0.5, max=0.5)
        A_1 = Parameter('A_1', value=0.5, min=0.3, max=0.7)
        g_1 = A_1 * BivariateGaussian(x=x, y=y, mu_x=x0_1, mu_y=y0_1,
                                      sig_x=sig_x_1, sig_y=sig_y_1, rho=rho_1)

        x0_2 = Parameter('x0_2', value=0.3, min=0.2, max=0.4)
        sig_x_2 = Parameter('sig_x_2', value=0.05, min=0.0, max=0.2)
        y0_2 = Parameter('y0_2', value=0.3, min=0.2, max=0.4)
        sig_y_2 = Parameter('sig_y_2', value=0.1, min=0.0, max=0.2)
        rho_2 = Parameter('rho_2', value=0.26, min=0.0, max=0.8)
        A_2 = Parameter('A_2', value=0.5, min=0.3, max=0.7)
        g_2 = A_2 * BivariateGaussian(x=x, y=y, mu_x=x0_2, mu_y=y0_2,
                                      sig_x=sig_x_2, sig_y=sig_y_2, rho=rho_2)

        model = g_1 + g_2
        fit = Fit(model, xx, yy, ydata)
        fit_result = fit.execute()

        assert fit_result.r_squared > 0.95
        for param in fit.model.params:
            try:
                assert fit_result.stdev(param)**2 == pytest.approx(fit_result.variance(param))
            except AssertionError:
                assert fit_result.variance(param) <= 0.0
                assert np.isnan(fit_result.stdev(param))

        # Covariance matrix should be symmetric
        for param_1 in fit.model.params:
            for param_2 in fit.model.params:
                assert fit_result.covariance(param_1, param_2) == pytest.approx(fit_result.covariance(param_2, param_1), rel=1e-3)

    def test_minimizer_included(self):
        """"The minimizer used should be included in the results."""
        assert isinstance(self.constrained_result.minimizer, BaseMinimizer)
        assert isinstance(self.constrained_basinhopping_result.minimizer, BaseMinimizer)
        assert isinstance(self.likelihood_result.minimizer, BaseMinimizer)
        assert isinstance(self.fit_result.minimizer, BaseMinimizer)
        assert isinstance(self.chained_result.minimizer, ChainedMinimizer)
        for minimizer, cls in zip(self.chained_result.minimizer.minimizers, [BFGS, NelderMead]):
            assert isinstance(minimizer, cls)

    def test_objective_included(self):
        """"The objective used should be included in the results."""
        assert isinstance(self.fit_result.objective, LeastSquares)
        assert isinstance(self.minpack_result.objective, VectorLeastSquares)
        assert isinstance(self.likelihood_result.objective, LogLikelihood)
        assert isinstance(self.constrained_result.objective, LeastSquares)
        assert isinstance(self.constrained_basinhopping_result.objective, LeastSquares)

    def test_constraints_included(self):
        """
        Test if the constraints have been properly fed to the results object so
        we can easily print their compliance.
        """
        # For a constrained fit we expect a list of MinimizeModel objectives.
        for constrained_result in [self.constrained_result, self.constrained_basinhopping_result]:
            assert isinstance(constrained_result.constraints, list)
            for constraint in self.constrained_result.constraints:
                assert isinstance(constraint, MinimizeModel)

    def test_message_included(self):
        """Status message should be included."""
        assert isinstance(self.fit_result.status_message, str)
        assert isinstance(self.minpack_result.status_message, str)
        assert isinstance(self.likelihood_result.status_message, str)
        assert isinstance(self.constrained_result.status_message, str)
        assert isinstance(self.constrained_basinhopping_result.status_message, str)

    def test_pickle(self):
        for fit_result in [self.fit_result, self.chained_result,
                           self.constrained_basinhopping_result,
                           self.constrained_result, self.likelihood_result]:
            dumped = pickle.dumps(fit_result)
            new_result = pickle.loads(dumped)
            assert sorted(fit_result.__dict__.keys()) == sorted(new_result.__dict__.keys())
            for k, v1 in fit_result.__dict__.items():
                v2 = new_result.__dict__[k]
                if k == 'minimizer':
                    assert type(v1) == type(v2)
                elif k != 'minimizer_output':  # Ignore minimizer_output
                    if isinstance(v1, np.ndarray):
                        assert v1 == pytest.approx(v2, nan_ok=True)

    def test_gof_presence(self):
        """
        Test if the expected goodness of fit estimators are present.
        """
        assert hasattr(self.fit_result, 'objective_value')
        assert hasattr(self.fit_result, 'r_squared')
        assert hasattr(self.fit_result, 'chi_squared')
        assert not hasattr(self.fit_result, 'log_likelihood')
        assert not hasattr(self.fit_result, 'likelihood')

        assert hasattr(self.minpack_result, 'objective_value')
        assert hasattr(self.minpack_result, 'r_squared')
        assert hasattr(self.minpack_result, 'chi_squared')
        assert not hasattr(self.minpack_result, 'log_likelihood')
        assert not hasattr(self.minpack_result, 'likelihood')

        assert hasattr(self.likelihood_result, 'objective_value')
        assert not hasattr(self.likelihood_result, 'r_squared')
        assert not hasattr(self.likelihood_result, 'chi_squared')
        assert hasattr(self.likelihood_result, 'log_likelihood')
        assert hasattr(self.likelihood_result, 'likelihood')
