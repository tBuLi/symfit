from .hypothesis_helpers import model_with_data, expression_strategy, polynomial_strategy
from hypothesis import strategies as st
from hypothesis import given, note, assume, settings, HealthCheck, event

import numpy as np
from pytest import approx

from symfit import Fit, variables, parameters
from symfit.core.minimizers import TrustConstr, SLSQP, LBFGSB


DEPENDENT_VARS = st.sampled_from(variables('Y1, Y2'))
INDEPENDENT_VARS = st.sampled_from(variables('x, y, z'))
PARAMETERS = st.sampled_from(parameters('a, b, c'))

def _cmp_result(reference, found, abs=None, rel=None):
    reference = reference.copy()
    for k, v in reference.items():
        reference[k] = approx(v, abs=abs, rel=rel)
    assert found == reference

    # assert reference.keys() == found.keys()
    # ref_vals = []
    # found_vals = []
    # for k in reference:
    #     ref_vals.append(reference[k])
    #     found_vals.append(found[k])
    # return found_vals == approx(ref_vals, abs=abs, rel=rel)


def expressions(vars, pars):
    return st.one_of(polynomial_strategy(vars, pars), expression_strategy(vars, pars))


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(model_with_data(DEPENDENT_VARS,
                       INDEPENDENT_VARS,
                       PARAMETERS,
                       expression_strategy))
def test_simple(model_with_data):
    model, param_vals, indep_vals, dep_vals = model_with_data
    note(str(model))
    assume(model.params)  # Stacking error in model.eval_jacobian

    fit = Fit(model, **indep_vals, **dep_vals)
    result = fit.execute()
    note(str(result))
    note(str(model(**indep_vals, **result.params)))
    note(str(fit.minimizer))
    _cmp_result(param_vals, result.params)


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
@given(st.data())
def test_linear_models(data):
    # minimizer = data.draw(st.sampled_from([None, LBFGSB, TrustConstr, SLSQP]), label='minimizer')
    minimizer = None
    model, param_vals, indep_vals, dep_vals = data.draw(
        model_with_data(DEPENDENT_VARS,
                        INDEPENDENT_VARS,
                        PARAMETERS,
                        polynomial_strategy,
                        allow_interdependent=False)
    )
    note(str(model))
    note(param_vals)
    note(indep_vals)
    note(dep_vals)

    assume(model.params)  # Stacking error in model.eval_jacobian
    init_vals = param_vals.copy()
    for param in model.params:
        # param.value = data.draw(st.floats(param.min, param.max, allow_nan=False, allow_infinity=False))
        param.value = data.draw(st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
        init_vals[str(param)] = param.value
        note((param.name, param.min, param.value, param.max))
        # assert param.min is None or param.min <= param_vals[str(param)]
        # assert param.max is None or param_vals[str(param)] <= param.max

    fit = Fit(model, **indep_vals, **dep_vals, minimizer=minimizer)
    obj_jac = fit.objective.eval_jacobian(**init_vals)
    note(str(obj_jac))
    note(str(fit.objective.eval_hessian(**init_vals)))
    # Exclude some edge cases that won't do anything due to numerical precision
    # This also catches cases like y=a*x with y=0 and x=0
    assume(np.any(np.abs(obj_jac) > 1e-3))

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
    # If the R2 isn't one, check whether the parameters are ~equal.
    if result.r_squared != approx(1, abs=5e-3):
        event('R squared != 1')
        _cmp_result(param_vals, result.params, rel=1e-3, abs=1e-2)
    else:
        event('R squared == 1')