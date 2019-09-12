from .hypothesis_helpers import model_with_data
from hypothesis import strategies as st
from hypothesis import given, note, assume, settings, HealthCheck

import numpy as np
from pytest import approx

from symfit import Fit, variables, parameters


DEPENDENT_VARS = st.lists(st.sampled_from(variables('Y1, Y2')), unique=True, min_size=1)
SYMBOLS = st.sampled_from(parameters('a, b, c') + variables('x, y, z'))

def _cmp_result(reference, found):
    assert reference.keys() == found.keys()
    ref_vals = []
    found_vals = []
    for k in reference:
        ref_vals.append(reference[k])
        found_vals.append(found[k])
    assert found_vals == approx(ref_vals)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(model_with_data(DEPENDENT_VARS, SYMBOLS))
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
