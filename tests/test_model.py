# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from __future__ import division, print_function
import pytest
from collections import OrderedDict
import pickle
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

import numpy as np

from symfit import (
    Fit, parameters, variables, Model, ODEModel, D, Eq,
    CallableModel, CallableNumericalModel, Inverse, MatrixSymbol, Symbol, sqrt,
    Function, diff
)
from symfit.core.models import (
    jacobian_from_model, hessian_from_model, ModelError, ModelOutput
)


"""
Tests for Model objects.
"""


def test_model_as_dict():
    x, y_1, y_2 = variables('x, y_1, y_2')
    a, b = parameters('a, b')

    model_dict = OrderedDict([(y_1, a * x**2), (y_2, 2 * x * b)])
    model = Model(model_dict)

    assert model[y_1] is model_dict[y_1]
    assert model[y_2] is model_dict[y_2]
    assert len(model) == len(model_dict)
    assert model.items() == model_dict.items()
    assert model.keys() == model_dict.keys()
    assert list(model.values()) == list(model_dict.values())
    assert y_1 in model
    assert not model[y_1] in model


def test_order():
    """
    The model has to behave like an OrderedDict. This is of the utmost importance!
    """
    x, y_1, y_2 = variables('x, y_1, y_2')
    a, b = parameters('a, b')

    model_dict = {y_2: a * x**2, y_1: 2 * x * b}
    model = Model(model_dict)

    assert model.dependent_vars == list(model.keys())


def test_neg():
    """
    Test negation of all model types
    """
    x, y_1, y_2 = variables('x, y_1, y_2')
    a, b = parameters('a, b')

    model_dict = {y_2: a * x ** 2, y_1: 2 * x * b}
    model = Model(model_dict)

    model_neg = - model
    for key in model:
        assert model[key] == - model_neg[key]

    # Constraints
    constraint = Model.as_constraint(Eq(a * x, 2), model)

    constraint_neg = - constraint
    # for key in constraint:
    assert constraint[constraint.dependent_vars[0]] == - constraint_neg[constraint_neg.dependent_vars[0]]

    # ODEModel
    odemodel = ODEModel({D(y_1, x): a * x}, initial={a: 1.0})

    odemodel_neg = - odemodel
    for key in odemodel:
        assert odemodel[key] == - odemodel_neg[key]

    # For models with interdependency, negation should only change the
    # dependent components.
    model_dict = {x: y_1**2, y_1: a * y_2 + b}
    model = Model(model_dict)

    model_neg = - model
    for key in model:
        if key in model.dependent_vars:
            assert model[key] == - model_neg[key]
        elif key in model.interdependent_vars:
            assert model[key] == model_neg[key]
        else:
            pytest.fail()


def test_CallableNumericalModel():
    x, y, z = variables('x, y, z')
    a, b = parameters('a, b')

    model = CallableModel({y: a * x + b})
    numerical_model = CallableNumericalModel(
        {y: lambda x, a, b: a * x + b}, 
        connectivity_mapping={y: {x, a, b}}
    )
    assert model.__signature__ == numerical_model.__signature__

    xdata = np.linspace(0, 10)
    ydata = model(x=xdata, a=5.5, b=15.0).y + np.random.normal(0, 1)

    symbolic_answer = np.array(model(x=xdata, a=5.5, b=15.0))
    numerical_answer = np.array(numerical_model(x=xdata, a=5.5, b=15.0))

    assert numerical_answer == pytest.approx(symbolic_answer)

    faulty_model = CallableNumericalModel({y: lambda x, a, b: a * x + b},
                                          connectivity_mapping={y: {a, b}})
    assert not model.__signature__ == faulty_model.__signature__
    with pytest.raises(TypeError):
        # This is an incorrect signature, even though the lambda function is
        # correct. Should fail.
        faulty_model(xdata, 5.5, 15.0)

    # Faulty model whose components do not all accept all of the args
    with pytest.warns(DeprecationWarning):
        faulty_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b, z: lambda x, a: x**a}, [x], [a, b]
        )
    assert model.__signature__ == faulty_model.__signature__

    with pytest.raises(TypeError):
        # Lambda got an unexpected keyword 'b'
        faulty_model(xdata, 5.5, 15.0)

    # Faulty model with a wrongly named argument
    faulty_model = CallableNumericalModel(
        {y: lambda x, a, c=5: a * x + c}, connectivity_mapping={y: {x, a, b}}
    )
    assert model.__signature__ == faulty_model.__signature__

    with pytest.raises(TypeError):
        # Lambda got an unexpected keyword 'b'
        faulty_model(xdata, 5.5, 15.0)

    # Correct version of the previous model
    numerical_model = CallableNumericalModel(
        {y: lambda x, a, b: a * x + b, z: lambda x, a: x ** a},
        connectivity_mapping={y: {a, b, x}, z: {x, a}}
    )
    # Correct version of the previous model
    mixed_model = CallableNumericalModel(
        {y: lambda x, a, b: a * x + b, z: x ** a}, 
        connectivity_mapping={y: {x, a, b}}
    )

    numberical_answer = np.array(numerical_model(x=xdata, a=5.5, b=15.0))
    mixed_answer = np.array(mixed_model(x=xdata, a=5.5, b=15.0))
    assert numberical_answer == pytest.approx(mixed_answer)

    zdata = mixed_model(x=xdata, a=5.5, b=15.0).z + np.random.normal(0, 1)

    # Check if the fits are the same
    fit = Fit(mixed_model, x=xdata, y=ydata, z=zdata)
    mixed_result = fit.execute()
    fit = Fit(numerical_model, x=xdata, y=ydata, z=zdata)
    numerical_result = fit.execute()
    for param in [a, b]:
        assert mixed_result.value(param) == pytest.approx(numerical_result.value(param))
        if mixed_result.stdev(param) is not None and numerical_result.stdev(param) is not None:
            assert mixed_result.stdev(param) == pytest.approx(numerical_result.stdev(param))
        else:
            assert  mixed_result.stdev(param) is None and numerical_result.stdev(param) is None
    assert mixed_result.r_squared == pytest.approx(numerical_result.r_squared)

    # Test if the constrained syntax is supported
    fit = Fit(numerical_model, x=xdata, y=ydata,
              z=zdata, constraints=[Eq(a, b)])
    constrained_result = fit.execute()
    assert constrained_result.value(a) == pytest.approx(constrained_result.value(b))


def test_CallableNumericalModel_infer_connectivity():
    """
    When a CallableNumericalModel is initiated with symbolical and
    non-symbolical components, only the connectivity mapping for
    non-symbolical part has to be provided.
    """
    x, y, z = variables('x, y, z')
    a, b = parameters('a, b')
    model_dict = {z: lambda y, a, b: a * y + b,
                  y: x ** a}
    mixed_model = CallableNumericalModel(
        model_dict, connectivity_mapping={z: {y, a, b}}
    )
    assert mixed_model.connectivity_mapping == {z: {y, a, b}, y: {x, a}}


def test_CallableNumericalModel2D():
    """
    Apply a CallableNumericalModel to 2D data, to see if it is
    agnostic to data shape.
    """
    shape = (30, 40)

    def function(a, b):
        out = np.ones(shape) * a
        out[15:, :] += b
        return out

    a, b = parameters('a, b')
    y, = variables('y')

    model = CallableNumericalModel({y: function}, connectivity_mapping={y: {a, b}})
    data = 15 * np.ones(shape)
    data[15:, :] += 20

    fit = Fit(model, y=data)
    fit_result = fit.execute()
    assert fit_result.value(a) == pytest.approx(15)
    assert fit_result.value(b) == pytest.approx(20)

    def flattened_function(a, b):
        out = np.ones(shape) * a
        out[15:, :] += b
        return out.flatten()

    model = CallableNumericalModel({y: flattened_function}, connectivity_mapping={y: {a, b}})
    data = 15 * np.ones(shape)
    data[15:, :] += 20
    data = data.flatten()

    fit = Fit(model, y=data)
    flat_result = fit.execute()

    assert fit_result.value(a) == pytest.approx(flat_result.value(a))
    assert fit_result.value(b) == pytest.approx(flat_result.value(b))

    assert fit_result.stdev(a) is None and flat_result.stdev(a) is None
    assert fit_result.stdev(b) is None and flat_result.stdev(b) is None

    assert fit_result.r_squared == pytest.approx(flat_result.r_squared)


def test_pickle():
    """
    Make sure models can be pickled are preserved when pickling
    """
    a, b = parameters('a, b')
    x, y = variables('x, y')
    exact_model = Model({y: a * x ** b})
    constraint = Model.as_constraint(Eq(a, b), exact_model)
    with pytest.warns(DeprecationWarning):
        num_model = CallableNumericalModel(
            {y: a * x ** b}, independent_vars=[x], params=[a, b]
        )
    connected_num_model = CallableNumericalModel(
        {y: a * x ** b}, connectivity_mapping={y: {x, a, b}}
    )
    # Test if lsoda args and kwargs are pickled too
    ode_model = ODEModel({D(y, x): a * x + b}, {x: 0.0}, 3, 4, some_kwarg=True)

    models = [exact_model, constraint, num_model, ode_model, connected_num_model]
    for model in models:
        new_model = pickle.loads(pickle.dumps(model))
        # Compare signatures
        assert model.__signature__ == new_model.__signature__
        # Trigger the cached vars because we compare `__dict__` s
        model.vars
        new_model.vars
        # Explicitly make sure the connectivity mapping is identical.
        assert model.connectivity_mapping == new_model.connectivity_mapping
        if not isinstance(model, ODEModel):
            model.function_dict
            model.vars_as_functions
            new_model.function_dict
            new_model.vars_as_functions
        assert model.__dict__ == new_model.__dict__


def test_MatrixSymbolModel():
    """
    Test a model which is defined by ModelSymbols, see #194
    """
    N = Symbol('N', integer=True)
    M = MatrixSymbol('M', N, N)
    W = MatrixSymbol('W', N, N)
    I = MatrixSymbol('I', N, N)
    y = MatrixSymbol('y', N, 1)
    c = MatrixSymbol('c', N, 1)
    a, b = parameters('a, b')
    z, x = variables('z, x')

    model_dict = {
        W: Inverse(I + M / a ** 2),
        c: - W * y,
        z: sqrt(c.T * c)
    }
    # TODO: This should be a Model in the future, but sympy is not yet
    # capable of computing Matrix derivatives at the time of writing.
    model = CallableModel(model_dict)

    assert model.params == [a]
    assert model.independent_vars == [I, M, y]
    assert model.dependent_vars == [z]
    assert model.interdependent_vars == [W, c]
    assert model.connectivity_mapping == {W: {I, M, a}, c: {W, y}, z: {c}}
    # Generate data
    iden = np.eye(2)
    M_mat = np.array([[2, 1], [3, 4]])
    y_vec = np.array([3, 5])

    eval_model = model(I=iden, M=M_mat, y=y_vec, a=0.1)
    W_manual = np.linalg.inv(iden + M_mat / 0.1 ** 2)
    c_manual = - W_manual.dot(y_vec)
    z_manual = np.atleast_1d(np.sqrt(c_manual.T.dot(c_manual)))
    assert eval_model.W == pytest.approx(W_manual)
    assert eval_model.c == pytest.approx(c_manual)
    assert eval_model.z == pytest.approx(z_manual)

    # Now try to retrieve the value of `a` from a fit
    a.value = 0.2
    fit = Fit(model, z=z_manual, I=iden, M=M_mat, y=y_vec)
    fit_result = fit.execute()
    eval_model = model(I=iden, M=M_mat, y=y_vec, **fit_result.params)
    assert 0.1 == pytest.approx(np.abs(fit_result.value(a)))
    assert eval_model.W == pytest.approx(W_manual)
    assert eval_model.c == pytest.approx(c_manual)
    assert eval_model.z == pytest.approx(z_manual)

    # TODO: add constraints to Matrix model. But since Matrix expressions
    # can not yet be derived, this needs #154 to be solved first.


def test_interdependency_invalid():
    """
    Create an invalid model with interdependency.
    """
    a, b, c = parameters('a, b, c')
    x, y, z = variables('x, y, z')

    with pytest.raises(ModelError):
        # Invalid, parameters can not be keys
        model_dict = {
            c: a ** 3 * x + b ** 2,
            z: c ** 2 + a * b
        }
        model = Model(model_dict)

    with pytest.raises(ModelError):
        # Invalid, parameters can not be keys
        model_dict = {c: a ** 3 * x + b ** 2}
        model = Model(model_dict)


def test_interdependency():
    a, b = parameters('a, b')
    x, y, z = variables('x, y, z')
    model_dict = {
        y: a**3 * x + b**2,
        z: y**2 + a * b
    }
    callable_model = CallableModel(model_dict)
    assert callable_model.independent_vars == [x]
    assert callable_model.interdependent_vars == [y]
    assert callable_model.dependent_vars == [z]
    assert callable_model.params == [a, b]
    assert callable_model.connectivity_mapping == {y: {a, b, x}, z: {a, b, y}}
    assert callable_model(x=3, a=1, b=2) == pytest.approx(np.atleast_2d([7, 51]).T)
    for var, func in callable_model.vars_as_functions.items():
        # TODO comment on what this does
        str_con_map = set(x.name for x in callable_model.connectivity_mapping[var])
        str_args = set(str(x.__class__) if isinstance(x, Function) else x.name
                       for x in func.args)
        assert str_con_map == str_args

    jac_model = jacobian_from_model(callable_model)
    assert jac_model.params == [a, b]
    assert jac_model.dependent_vars == [D(z, a), D(z, b), z]
    assert jac_model.interdependent_vars == [D(y, a), D(y, b), y]
    assert jac_model.independent_vars == [x]
    for p1, p2 in zip_longest(jac_model.__signature__.parameters, [x, a, b]):
        assert str(p1) == str(p2)
    # The connectivity of jac_model should be that from it's own components
    # plus that of the model. The latter is needed to properly compute the
    # Hessian.
    jac_con_map = {D(y, a): {a, x},
                   D(y, b): {b},
                   D(z, a): {b, y, D(y, a)},
                   D(z, b): {a, y, D(y, b)},
                   y: {a, b, x}, z: {a, b, y}}
    assert jac_model.connectivity_mapping == jac_con_map
    jac_model_dict = {D(y, a): 3 * a**2 * x,
                      D(y, b): 2 * b,
                      D(z, a): b + 2 * y * D(y, a),
                      D(z, b): a + 2 * y * D(y, b),
                      y: callable_model[y], z: callable_model[z]}
    assert jac_model.model_dict == jac_model_dict
    for var, func in jac_model.vars_as_functions.items():
        str_con_map = set(x.name for x in jac_model.connectivity_mapping[var])
        str_args = set(str(x.__class__) if isinstance(x, Function) else x.name
                       for x in func.args)
        assert str_con_map == str_args

    hess_model = hessian_from_model(callable_model)
    # Result according to Mathematica
    hess_as_dict = {
        D(y, (a, 2)): 6 * a * x,
        D(y, a, b): 0,
        D(y, b, a): 0,
        D(y, (b, 2)): 2,
        D(z, (a, 2)): 2 * D(y, a)**2 + 2 * y * D(y, (a, 2)),
        D(z, a, b): 1 + 2 * D(y, b) * D(y, a) + 2 * y * D(y, a, b),
        D(z, b, a): 1 + 2 * D(y, b) * D(y, a) + 2 * y * D(y, a, b),
        D(z, (b, 2)): 2 * D(y, b)**2 + 2 * y * D(y, (b, 2)),
        D(y, a): 3 * a ** 2 * x,
        D(y, b): 2 * b,
        D(z, a): b + 2 * y * D(y, a),
        D(z, b): a + 2 * y * D(y, b),
        y: callable_model[y], z: callable_model[z]
    }
    assert dict(hess_model) == hess_as_dict

    assert hess_model.params == [a, b]
    assert hess_model.dependent_vars == [D(z, (a, 2)), D(z, a, b), D(z, (b, 2)), D(z, b, a), D(z, a), D(z, b), z]
    assert hess_model.interdependent_vars == [D(y, (a, 2)), D(y, a), D(y, b), y]
    assert hess_model.independent_vars == [x]

    model = Model(model_dict)
    assert model(x=3, a=1, b=2) == pytest.approx(np.atleast_2d([7, 51]).T)
    assert model.eval_jacobian(x=3, a=1, b=2) == pytest.approx(np.array([[[9], [4]], [[128], [57]]]))
    assert model.eval_hessian(x=3, a=1, b=2) == pytest.approx(np.array([[[[18], [0]], [[0], [2]]],[[[414], [73]], [[73], [60]]]]))

    assert model.__signature__ == model.jacobian_model.__signature__
    assert model.__signature__ == model.hessian_model.__signature__


def test_ModelOutput():
    """
    Test the ModelOutput object. To prevent #267 from recurring,
    we attempt to make a model with more than 255 variables.
    """
    params = parameters(','.join('a{}'.format(i) for i in range(300)))
    data = np.ones(300)
    output = ModelOutput(params, data)
    assert len(output) == 300
    assert isinstance(output._asdict(), OrderedDict)
    assert output._asdict() is not output.output_dict
    assert output._asdict() == output.output_dict

def test_model_output_pickle():
    """
    Test that ModelOutput objects can be pickled and unpickled.
    """
    params = parameters(','.join('a{}'.format(i) for i in range(10)))
    data = np.ones(10)
    output = ModelOutput(params, data)

    jar = pickle.dumps(output)
    result = pickle.loads(jar)

    assert output._asdict() == result._asdict()


