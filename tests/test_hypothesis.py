from .hypothesis_helpers import model_with_data, expression_strategy, polynomial_strategy
from hypothesis import strategies as st
from hypothesis import given, note, assume, settings, HealthCheck

import numpy as np
from pytest import approx

from symfit import Fit, variables, parameters
from symfit.core.minimizers import TrustConstr, SLSQP, LBFGSB


DEPENDENT_VARS = st.sampled_from(variables('Y1, Y2'))
INDEPENDENT_VARS = st.sampled_from(variables('x, y, z'))
PARAMETERS = st.sampled_from(parameters('a, b, c'))

def _cmp_params(reference, found, abs=None, rel=None):
    reference = reference.copy()
    for k, v in reference.items():
        reference[k] = approx(v, abs=abs, rel=rel)
    assert found == reference


def _cmp_model_vals(reference, found, abs=None, rel=None):
    actual = {}
    for k, v in found._asdict().items():
        if str(k) in reference:
            actual[str(k)] = approx(v, abs=abs, rel=rel)
    assert reference == actual


def expressions(vars, pars):
    return st.one_of(polynomial_strategy(vars, pars), expression_strategy(vars, pars))


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(model_with_data(DEPENDENT_VARS,
                       INDEPENDENT_VARS,
                       PARAMETERS,
                       expression_strategy))
def test_simple(model_with_data):
    """
    Generate a model with data and parameters, and fit the exact data (without
    any noise), with initial parameter values at the correct answer. Check that
    the minimizer doesn't drift away from this correct answer.
    Note. It is not possible to give approximate parameter values as starting
    point, since there may be multiple minima (and hypothesis *will* find those
    cases)
    """
    model, param_vals, indep_vals, dep_vals = model_with_data
    note(str(model))
    note(str(model(**indep_vals, **param_vals)))

    fit = Fit(model, **indep_vals, **dep_vals)
    result = fit.execute()
    note(str(result))
    note(str(fit.minimizer))
    _cmp_params(param_vals, result.params)


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(st.data())
def test_linear_models(data):
    """
    Generate a linear model (specifically, a polynomial), and pick parameter
    values between their min and max, and perform the fit. Note that there
    is still no noise on the data.
    """
    minimizer = data.draw(st.sampled_from([None, LBFGSB, SLSQP, TrustConstr]), label='minimizer')
    # minimizer = None
    model, param_vals, indep_vals, dep_vals = data.draw(
        model_with_data(DEPENDENT_VARS,
                        INDEPENDENT_VARS,
                        PARAMETERS,
                        polynomial_strategy,
                        allow_interdependent=False),  # If true, no longer linear
        label="Model"
    )
    note(str(minimizer))
    note(str(model))
    for param in model.params:
        note("{}: {} {} {}".format(param, param.min, param.value, param.max))
    note(indep_vals)
    note(dep_vals)

    init_vals = param_vals.copy()
    for param in model.params:
        param.value = data.draw(st.floats(
            -1e3, 1e3,
            allow_nan=False,
            allow_infinity=False), label="Parameter value {}".format(param))
        init_vals[str(param)] = param.value
        note((param.name, param.min, param.value, param.max))

    fit = Fit(model, **indep_vals, **dep_vals, minimizer=minimizer)
    obj_jac = fit.objective.eval_jacobian(**init_vals)
    # If the jacobian is too large you get numerical issues (in particular
    # SLSQP struggles?).
    assume(np.all(np.abs(obj_jac) < 1e5))
    note(str(obj_jac))
    note(str(fit.objective.eval_hessian(**init_vals)))

    result = fit.execute()
    # "Minimization stopped due to precision loss" and variants
    # assume(result.infodict['success'])
    note(str(result.covariance_matrix))
    note(str(result))
    note(str(model(**indep_vals, **result.params)))
    note(str(fit.objective(**result.params)))
    note(str(fit.objective.eval_jacobian(**result.params)))
    note(str(fit.objective.eval_hessian(**result.params)))
    note(str(fit.minimizer))

    tmp = {v: dep_vals[str(v)] for v in dep_vals if not v.startswith('sigma_')}
    _cmp_model_vals(tmp, model(**result.params, **indep_vals), rel=1e-3, abs=1e-2)
