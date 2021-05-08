# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import pytest

import numpy as np
from scipy.optimize import minimize, basinhopping

from symfit import (
    Variable, Parameter, Eq, Ge, Le, Lt, Gt, Ne, parameters, ModelError, Fit,
    Model, cos, CallableNumericalModel
)
from symfit.core.objectives import MinimizeModel
from symfit.core.minimizers import BFGS, BasinHopping, LBFGSB, SLSQP, NelderMead
from symfit.core.support import partial


def setup_module():
    np.random.seed(0)


@pytest.mark.usefixtures("minimize_fixture")
class TestMinimize:
    """
    Tests maximizing a function with and without constraints, taken from the
    scipy `minimize` tutorial. Compare the symfit result with the scipy
    result.
    https://docs.scipy.org/doc/scipy-0.18.1/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
    """

    @pytest.fixture(autouse=True)
    def minimize_fixture(self):
        """
        Set up the parameters, model, constraints for the minimization fits.

        These tests used to purposefully use unamed variable, however this has
        been changed as this feature is now being depricated in a future
        version of Symfit.
        """
        x = Parameter('x', value=-1.0)
        y = Parameter('y', value=1.0)
        self.x = x
        self.y = y
        self.model = Model(2 * x * y + 2 * x - x ** 2 - 2 * y ** 2)

        self.constraints = [
            Ge(y - 1, 0),  # y - 1 >= 0,
            Eq(x**3 - y, 0),  # x**3 - y == 0,
        ]

        self.cons = (
            {'type': 'eq',
             'fun': lambda x: np.array([x[0]**3 - x[1]]),
             'jac': lambda x: np.array([3.0 * (x[0]**2.0), -1.0])},
            {'type': 'ineq',
             'fun': lambda x: np.array([x[1] - 1]),
             'jac': lambda x: np.array([0.0, 1.0])}
        )

    @staticmethod
    def func(x, sign=1.0):
        """ Objective function """
        return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)

    @staticmethod
    def func_deriv(x, sign=1.0):
        """ Derivative of objective function """
        dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
        dfdx1 = sign*(2*x[0] - 4*x[1])
        return np.array([dfdx0, dfdx1])

    def test_minimize_unconstrained(self):
        """
        Test an unconstrained fit.
        """
        res = minimize(self.func, [-1.0, 1.0], args=(-1.0,), jac=self.func_deriv,
                   method='BFGS', options={'disp': False})
        fit = Fit(model=-self.model)
        assert isinstance(fit.objective, MinimizeModel)
        assert isinstance(fit.minimizer, BFGS)

        fit_result = fit.execute()

        assert fit_result.value(self.x) == pytest.approx(res.x[0], 1e-6)
        assert fit_result.value(self.y) == pytest.approx(res.x[1], 1e-6)

    def test_minimize_constrained(self):
        """
        Test a constrained fit.
        """
        res = minimize(self.func, [-1.0, 1.0], args=(-1.0,), jac=self.func_deriv,
                   constraints=self.cons, method='SLSQP', options={'disp': False})

        fit = Fit(-self.model, constraints=self.constraints)
        assert fit.constraints[0].constraint_type == Ge
        assert fit.constraints[1].constraint_type == Eq
        fit_result = fit.execute()
        assert fit_result.value(self.x) == pytest.approx(res.x[0], 1e-6)
        assert fit_result.value(self.y) == pytest.approx(res.x[1], 1e-6)


def test_constraint_types():
    x = Parameter('x', value=-1.0)
    y = Parameter('y', value=1.0)
    z = Variable('z')
    model = Model({z: 2*x*y + 2*x - x**2 - 2*y**2})

    # These types are not allowed constraints.
    for relation in [Lt, Gt, Ne]:
        with pytest.raises(ModelError):
            Fit(model, constraints=[relation(x, y)])


    # Should execute without problems.
    for relation in [Eq, Ge, Le]:
        Fit(model, constraints=[relation(x, y)])

    fit = Fit(model, constraints=[Le(x, y)])
    # Le should be transformed to Ge
    assert fit.constraints[0].constraint_type is Ge

    # Redo the standard test as a Le
    constraints = [
        Le(- y + 1, 0),  # y - 1 >= 0,
        Eq(x**3 - y, 0),  # x**3 - y == 0,
    ]
    std_constraints = [
        Ge(y - 1, 0),  # y - 1 >= 0,
        Eq(x**3 - y, 0),  # x**3 - y == 0,
    ]

    fit = Fit(-model, constraints=constraints)
    std_fit = Fit(-model, constraints=std_constraints)
    assert fit.constraints[0].constraint_type == Ge
    assert fit.constraints[1].constraint_type == Eq
    assert fit.constraints[0].params == [x, y]
    assert fit.constraints[1].params == [x, y]
    assert fit.constraints[0].jacobian_model.params == [x, y]
    assert fit.constraints[1].jacobian_model.params == [x, y]
    assert fit.constraints[0].hessian_model.params == [x, y]
    assert fit.constraints[1].hessian_model.params == [x, y]
    assert fit.constraints[0].__signature__ == fit.constraints[1].__signature__
    fit_result = fit.execute()
    std_result = std_fit.execute()
    assert fit_result.value(x) == pytest.approx(std_result.value(x))
    assert fit_result.value(y) == pytest.approx(std_result.value(y))


def test_basinhopping_large():
    """
    Test the basinhopping method of scipy.minimize. This is based of scipy's docs
    as found here: https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.anneal.html
    """
    def f1(z, *params):
        x, y = z
        a, b, c, d, e, f, g, h, i, j, k, l, scale = params
        return (a * x ** 2 + b * x * y + c * y ** 2 + d * x + e * y + f)

    def f2(z, *params):
        x, y = z
        a, b, c, d, e, f, g, h, i, j, k, l, scale = params
        return (-g * np.exp(-((x - h) ** 2 + (y - i) ** 2) / scale))

    def f3(z, *params):
        x, y = z
        a, b, c, d, e, f, g, h, i, j, k, l, scale = params
        return (-j * np.exp(-((x - k) ** 2 + (y - l) ** 2) / scale))

    def func(z, *params):
        x, y = z
        a, b, c, d, e, f, g, h, i, j, k, l, scale = params
        return f1(z, *params) + f2(z, *params) + f3(z, *params)

    def f_symfit(x1, x2, params):
        z = [x1, x2]
        return func(z, *params)

    params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)

    x0 = np.array([2., 2.])
    np.random.seed(555)
    res = basinhopping(func, x0, minimizer_kwargs={'args': params})

    np.random.seed(555)
    x1, x2 = parameters('x1, x2', value=x0)
    fit = BasinHopping(partial(f_symfit, params=params), [x1, x2])
    fit_result = fit.execute()

    assert res.x[0] == fit_result.value(x1)
    assert res.x[1] == fit_result.value(x2)
    assert res.fun == fit_result.objective_value


def test_basinhopping():
    def func(x):
        return np.cos(14.5 * x - 0.3) + (x + 0.2) * x
    x0 = [1.]
    np.random.seed(555)
    res = basinhopping(func, x0, minimizer_kwargs={"method": "BFGS"}, niter=200)
    np.random.seed(555)
    x, = parameters('x')
    fit = BasinHopping(func, [x], local_minimizer=BFGS)
    fit_result = fit.execute(niter=200)
    # fit_result = fit.execute(minimizer_kwargs={"method": "BFGS"}, niter=200)

    assert res.x == fit_result.value(x)
    assert res.fun == fit_result.objective_value


def test_basinhopping_2d():
    def func2d(x):
        f = np.cos(14.5 * x[0] - 0.3) + (x[1] + 0.2) * x[1] + (x[0] + 0.2) * x[0]
        df = np.zeros(2)
        df[0] = -14.5 * np.sin(14.5 * x[0] - 0.3) + 2. * x[0] + 0.2
        df[1] = 2. * x[1] + 0.2
        return f, df

    def func2d_symfit(x1, x2):
        f = np.cos(14.5 * x1 - 0.3) + (x2 + 0.2) * x2 + (x1 + 0.2) * x1
        return f

    def jac2d_symfit(x1, x2):
        df = np.zeros(2)
        df[0] = -14.5 * np.sin(14.5 * x1 - 0.3) + 2. * x1 + 0.2
        df[1] = 2. * x2 + 0.2
        return df

    np.random.seed(555)
    minimizer_kwargs = {'method': 'BFGS', 'jac': True}
    x0 = [1.0, 1.0]
    res = basinhopping(func2d, x0, minimizer_kwargs=minimizer_kwargs, niter=200)

    np.random.seed(555)
    x1, x2 = parameters('x1, x2', value=x0)

    with pytest.raises(TypeError):
        fit = BasinHopping(
            func2d_symfit, [x1, x2],
            local_minimizer=NelderMead(func2d_symfit, [x1, x2],
                                       jacobian=jac2d_symfit)
        )

    fit = BasinHopping(
        func2d_symfit, [x1, x2],
        local_minimizer=BFGS(func2d_symfit, [x1, x2], jacobian=jac2d_symfit)
    )
    fit_result = fit.execute(niter=200)
    assert isinstance(fit.local_minimizer.jacobian, MinimizeModel)
    assert isinstance(fit.local_minimizer.jacobian.model, CallableNumericalModel)
    assert res.x[0] == fit_result.value(x1)
    assert res.x[1] == fit_result.value(x2)
    assert res.fun == fit_result.objective_value

    # Now compare with the symbolic equivalent
    np.random.seed(555)
    model = cos(14.5 * x1 - 0.3) + (x2 + 0.2) * x2 + (x1 + 0.2) * x1
    fit = Fit(model, minimizer=BasinHopping)
    fit_result = fit.execute()
    assert res.x[0] == fit_result.value(x1)
    assert res.x[1] == fit_result.value(x2)
    assert res.fun == fit_result.objective_value
    assert isinstance(fit.minimizer.local_minimizer, BFGS)

    # Impose constrains
    np.random.seed(555)
    model = cos(14.5 * x1 - 0.3) + (x2 + 0.2) * x2 + (x1 + 0.2) * x1
    fit = Fit(model, minimizer=BasinHopping, constraints=[Eq(x1, x2)])
    fit_result = fit.execute()
    assert fit_result.value(x1) == fit_result.value(x2)
    assert isinstance(fit.minimizer.local_minimizer, SLSQP)

    # Impose bounds
    np.random.seed(555)
    x1.min = 0.0
    model = cos(14.5 * x1 - 0.3) + (x2 + 0.2) * x2 + (x1 + 0.2) * x1
    fit = Fit(model, minimizer=BasinHopping)
    fit_result = fit.execute()
    assert fit_result.value(x1) >= x1.min
    assert isinstance(fit.minimizer.local_minimizer, LBFGSB)
