import numpy as np

from symfit import Model, variables
from symfit.core.support import key2str
import sympy

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from hypothesis import assume


MAX_VAL = 2**10
MIN_VAL = 2**-10
FLOATS = st.floats(allow_nan=False, allow_infinity=False,
                   min_value=MIN_VAL, max_value=MAX_VAL,
                   allow_subnormal=False, width=32)
# Some hoops to jump through to limit the floats to MAX_VAL > abs(f) > MIN_VAL,
# but still allow f=0
NUMBERS = st.one_of(st.just(0), FLOATS, FLOATS.map(lambda f: -f))


def _is_number(obj):
    obj = sympy.sympify(obj)
    return obj.is_number


@st.composite
def expressions(draw, variables, parameters, constants, unary_ops, operations,
                n_leaves, allow_number=False):
    """
    A strategy to generate arbitrary :mod:`Sympy` expressions, based on
    strategies for generating `symbols` and `operations`.

    Parameters
    ----------
    variables: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as independent variables.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        A strategy that generates Parameters.
    constants: hypothesis.LazySearchStrategy[number]
        A strategy that generates numbers. Should probably contain at least 1
        and -1.
    unary_ops: hypothesis.LazySearchStrategy[sympy.Operation]
        A strategy that generates sympy operations that take a single argument.
    operations: hypothesis.LazySearchStrategy[tuple[int, sympy.Operation, bool]]
        A strategy that generates operations with the associated number of
        arguments and whether they commute.
    n_leaves: int
        The number of symbols to initially generate. Larger numbers result in
        more complex expressions.
    allow_number: bool
        Whether the produced expression can be just a number.

    Returns
    -------
    sympy.Expression
    """
    # This will work by generating an expression tree, except that we
    # immediately collapse it: Generate the leaves, and combine those by
    # applying operations to them. For extra flavour, apply some unary
    # operations along the way.

    # Make sure we draw at least one not numerical symbol, unless we allow
    # an expression consisting of just a number
    if not allow_number:
        first = st.one_of(variables, parameters)
        rest = st.lists(st.one_of(constants, variables, parameters),
                        min_size=n_leaves - 1, max_size=n_leaves - 1)
        symbol_st = st.tuples(first, rest).map(lambda t: [t[0], *t[1]])
    else:
        symbol_st = st.lists(st.one_of(constants, variables, parameters),
                             min_size=n_leaves, max_size=n_leaves)
    leaves = draw(symbol_st)

    unary_ops = st.lists(unary_ops, min_size=0, max_size=3)
    first_ops = draw(st.lists(unary_ops,
                              min_size=len(leaves),
                              max_size=len(leaves)))

    for idx, ops in enumerate(first_ops):
        for op in ops:
            leaves[idx] = op(leaves[idx])

    while len(leaves) != 1:
        nargs, op, comm = draw(operations)
        # Draw indices for the leaves that get combined by the operation.
        idxs_st = st.lists(st.integers(min_value=0, max_value=len(leaves) - 1),
                           min_size=nargs,
                           max_size=nargs,
                           unique=True)
        # ... and if the operation commutes, make sure the indices are sorted.
        idxs = draw(idxs_st.map(lambda x: sorted(x) if comm else x))
        # idxs = draw(idxs_st)
        args = [leaves[idx] for idx in idxs]
        # Remove the drawn symbols from `leaves`
        leaves = [leaf for idx, leaf in enumerate(leaves) if idx not in idxs]
        new_node = op(*args)
        ops = draw(unary_ops)
        for op in ops:
            new_node = op(new_node)
        leaves.append(new_node)
    expr = leaves[0]
    assume(allow_number or not expr.is_number)
    return expr


def expression_strategy(variables, parameters, n_leaves=5):
    """
    Helper function that calls :func:`expressions` with sane
    defaults.

    Parameters
    ----------
    variables: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as independent variables.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        A strategy that generates Parameters.
    n_leaves: int
        The number of symbols to initially generate. Larger numbers result in
        more complex expressions.

    Returns
    -------
    hypothesis.LazySearchStrategy[sympy.Expr]
    """
    constants = st.sampled_from([1, -1, 2, 3,
                                 sympy.Rational(1, 2), sympy.pi, sympy.E])
    ops = st.sampled_from([(2, sympy.Add, True),
                           (2, sympy.Mul, True),
                           # Can cause imaginary numbers (e.g. (-4)**1.2)
                           # (2, sympy.Pow, False),
                           # (2, sympy.log, False),
                           ])
    unary_ops = st.sampled_from([lambda e: -e,
                                 lambda e: 1/e,
                                 sympy.sin,
                                 sympy.exp])
    expr = expressions(variables=variables, parameters=parameters,
                       constants=constants, unary_ops=unary_ops,
                       operations=ops, n_leaves=n_leaves)
    return expr


@st.composite
def polynomial_expressions(draw, variables, parameters, constants,
                           max_degree=10, max_terms=5):
    """
    Generates polynomial expressions that are linear in `parameters` and
    `constants`. The produced polynomials will have at most `max_terms` terms,
    and the degree will be between 1 and `max_degree`.

    Parameters
    ----------
    variables: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as independent variables.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        A strategy that generates Parameters.
    constants: hypothesis.LazySearchStrategy[number]
        A strategy that generates constants.
    max_degree: int
        Maximum degree of the produced polynomial
    max_terms: int
        Maximum number of terms of the produced polynomial

    Returns
    -------
    hypothesis.LazySearchStrategy[sympy.Expr]
    """
    indep_vars = draw(st.lists(variables, unique=True, min_size=1))
    orders = st.dictionaries(
        # Dict keys are the exponents for the variables, values the coefficient
        # https://docs.sympy.org/latest/modules/polys/domainsintro.html#sparse-polynomial-representation
        keys=st.tuples(*[st.integers(min_value=0, max_value=max_degree)]*len(indep_vars)).filter(lambda t: 1 <= sum(t) <= max_degree),
        values=st.one_of(constants, parameters),
        min_size=1,
        max_size=max_terms
    )
    orders = draw(orders)
    expr = sympy.Poly(orders, *indep_vars).as_expr()
    return expr


def polynomial_strategy(variables, parameters, max_degree=5, max_terms=5):
    """
    Helper function that calls :func:`polynomial_expressions` with sane
    defaults.

    Parameters
    ----------
    variables: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as independent variables.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        A strategy that generates Parameters.
    max_degree: int
        Maximum degree of the produced polynomial
    max_terms: int
        Maximum number of terms of the produced polynomial

    Returns
    -------
    hypothesis.LazySearchStrategy[sympy.Expr]
    """
    return polynomial_expressions(variables, parameters, st.just(1),
                                  max_degree=max_degree, max_terms=max_terms)


# TODO: ODEModels
@st.composite
def models(draw, dependent_vars, independent_vars, parameters, expressions_st,
           allow_interdependent=True):
    """
    A strategy for generating :class:`symfit.Model` s. 

    Parameters
    ----------
    dependent_vars: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as dependent variables.
    independent_vars: hypothesis.LazySearchStrategy[symfit.Variable]
        A strategy that generates Variables. These variables will be
        used as independent variables.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        A strategy that generates Parameters.
    expressions_st: callable[(hypothesis.LazySearchStrategy[symfit.Variable],
                              hypothesis.LazySearchStrategy[symfit.Parameter]),
                             hypothesis.LazySearchStrategy[sympy.Expr]]
        A function that when given variables and parameters, produces a strategy
        that produces expressions.

    Returns
    -------
    symfit.Model
    """
    model_dict = {}
    dep_vars = draw(st.lists(dependent_vars, unique=True, min_size=1, max_size=3))
    for idx, dependent_var in enumerate(dep_vars):
        # Dependent vars can depend on other dependent vars: {y1: k, y2: y1+1}.
        # But this can cause cyclical dependency ({y1: y2, y2: y1}), so limit
        # to idx < jdx. Note that this also prevents {y1: y1}.
        other_vars = dep_vars[:idx]
        if other_vars and allow_interdependent:
            # Can't sample_from and empty list
            valid_variables = st.one_of(independent_vars, st.sampled_from(other_vars))
        else:
            valid_variables = independent_vars
        expression_st = expressions_st(valid_variables, parameters)
        expr = draw(expression_st)
        # Ban expressions that have complex infinity, which for some reason
        # can't be printed to numpy. Maybe add more?
        assume(not expr.has(sympy.zoo))
        model_dict[dependent_var] = expr
    model = Model(model_dict)
    return model


def _same_shape_vars(model):
    """
    Helper function that returns a set of frozensets of all variables in model
    that should have broadcast compatible shapes.
    """
    indep_vars = model.independent_vars
    con_map = model.connectivity_mapping.copy()
    shape_groups = {var: {var} for var in indep_vars}

    # Deal with interdependent vars
    for k, group in con_map.items():
        for var in group:
            if var in con_map:
                con_map[k] = group | con_map[var]
                con_map[k] -= {var}

    for group in con_map.values():
        comb = {var for var in group if var in indep_vars}
        shape_group = set()
        for var in comb:
            shape_group.update(shape_groups[var])
        for var in shape_group:
            shape_groups[var] = shape_group
    shape_groups = set(frozenset(g) for g in shape_groups.values())
    return shape_groups


@st.composite
def independent_data_shapes(draw, model, min_num_values=0, min_value=1,
                            max_value=None, min_dim=1, max_dim=None):
    """
    Strategy that generates shapes for the independent variables of `model` such
    that variables that need to be broadcast compatible will be the same shape.

    Parameters
    ----------
    model: symfit.Model
        Shapes will be generated for this model's independent variables.
    min_num_values: int
        The minimum number of values the final arrays should contain.
    min_value: int
        The minimum size a dimension can be.
    max_value: int
        The maximum size a dimension can be.
    min_dim: int
        The minimum number of dimensions the shape should have.
    max_dim: int
        The maximum number of dimensions the shape should have.

    Returns
    -------
    dict[str]: tuple[int, ...]
    """
    shape_groups = map(tuple, _same_shape_vars(model))
    # Insist on a minimal number of data points to prevent underspecified models
    # This is probably overly strict, since it enforces the same minimal shape
    # on *all* components/variables. Instead, it would probably be better to
    # do this per element of model.connectivity_mapping
    shapes = npst.array_shapes(
        min_dims=min_dim,
        max_dims=max_dim,
        min_side=min_value,
        max_side=max_value
    ).filter(lambda shape: np.prod(shape) >= min_num_values)
    grp_shape = draw(st.fixed_dictionaries({var_grp: shapes for var_grp in shape_groups}))
    indep_shape = {}
    for vars, shape in grp_shape.items():
        # Not all shapes have to be the same size, but they should be
        # broadcastable. Since it's a pain to find a shape A that is compatible
        # with both B, C and D we'll just say that the last shape can be
        # different.
        for var in vars[:-1]:
            indep_shape[var] = shape
        # Although technically possible, it can cause a bunch of numerical
        # issues, and it's meaning is debatable at best.
        # indep_shape[vars[-1]] = npst.broadcastable_shapes(shape,
        #     min_dims=min_dim,
        #     max_dims=max_dim,
        #     min_side=min_value,
        #     max_side=max_value
        # )
        indep_shape[vars[-1]] = shape
    return indep_shape


def _count_unique_values(indep_vals):
    return sum(
        np.sum(np.unique(ar.ravel(), return_counts=True)[1]) for ar in indep_vals.values()
    )


@st.composite
def indep_values(draw, model):
    """
    Strategy for generating independent data for a given `model`.

    Parameters
    ----------
    model: symfit.Model

    Returns
    -------
    dict[str, np.array]
        Dict of numpy arrays with data for every independent variable in
        `model`.
    """
    params = model.params

    # We need just a bit more data than parameters so that there's a best answer
    min_num_values = len(params) + 3
    indep_shape = draw(independent_data_shapes(model, min_num_values))

    indep_vals = {}
    for var, shape in indep_shape.items():
        data = npst.arrays(float, shape, elements=NUMBERS, unique=False).map(np.sort)
        indep_vals[str(var)] = data

    # Data points don't have to be unique, but there should be enough
    # independent ones to overspecify the fit.
    data = draw(st.fixed_dictionaries(indep_vals).filter(lambda d: _count_unique_values(d) >= min_num_values))
    return data


@st.composite
def param_values(draw, model):
    """
    Strategy for generating values for the parameters of `model`. Will also set
    the values and bounds of the parameters.

    Parameters
    ----------
    model: symfit.Model

    Returns
    -------
    dict[str, float]
    """
    param_vals = st.fixed_dictionaries({str(param): NUMBERS for param in model.params})
    data = draw(param_vals)
    for param in model.params:
        param.value = data[str(param)]
        lower, upper = draw(
            st.lists(
                st.one_of(
                    st.none(),
                    st.floats(1, 1+2*abs(param.value), allow_infinity=False, allow_nan=False)
                ),
                min_size=2,
                max_size=2
            )
        )
        param.min = param.value - lower if lower is not None else None
        param.max = param.value + upper if upper is not None else None

    return data


@st.composite
def dep_values(draw, model, indep_data, param_vals):
    """
    Strategy for generating dependent data for the given model, independent
    data, and parameter values.

    Parameters
    ----------
    model: symfit.Model
        The model to be used. Will be called with the given data and parameter
        values.
    indep_data: dict[str, np.array]
        Data for all the independent variables in `model`.
    param_vals: dict[str, float]
        Values for all the parameters in `model`.

    Returns
    -------
    dict[str, np.array]
        Data for the dependent variables of `model`, as well as associated
        sigmas.
    """
    dep_data = model(**indep_data, **param_vals)._asdict()

    # You can't provide data for interdependent variables. #270
    dep_data = {
        var: data
        for var, data in dep_data.items()
        if var in model.dependent_vars
    }
    shapes = {var: data.shape for var, data in dep_data.items() if data is not None}
    sigmas = {
        'sigma_{}'.format(str(var)): st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=5),
            npst.arrays(float, shape, elements=st.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=5))
        ) for var, shape in shapes.items()
    }
    sigmas = draw(st.fixed_dictionaries(dict(**sigmas)))
    return dict(**sigmas, **key2str(dep_data))


@st.composite
def model_with_data(draw, dependent_vars, independent_vars, parameters, expr_strategy, max_jacobian=1e3, **kwargs):
    """
    Strategy that generates a model with associated data using the given
    dependent variables.

    Parameters
    ----------
    dependent_vars: hypothesis.LazySearchStrategy[symfit.Variable]
        Strategy to use to generate dependent variables for the model.
    independent_vars: hypothesis.LazySearchStrategy[symfit.Variable]
        Strategy to use to generate independent variables for the model.
    parameters: hypothesis.LazySearchStrategy[symfit.Parameter]
        Strategy to use to generate parameters for the model.
    expr_strategy: hypothesis.LazySearchStrategy[sympy.Expr]
        Strategy to use to generate expressions for the model.
    max_jacobian: float or None
        If not None, the maximum values allowed for the Jacobian to avoid
        numerical instabilities

    Returns
    -------
    symfit.Model
        The generated model.
    dict[str, float]
        Values for the parameters.
    dict[str, np.array]
        Data for the independent variables.
    dict[str, np.array]
        Data for the dependent variables, as well as sigmas.
    """
    def _is_valid_data(data):
        indep_data, param_data = data
        try:
            dep_data = model(**indep_data, **param_data)
            jac = model.eval_jacobian(**indep_data, **param_data)
        except ZeroDivisionError:
            # Can happen with, for example: a**(-1.0) with a=0
            return False

        if max_jacobian is not None:
            low_jac = all(np.all(np.abs(component) <= max_jacobian) for component in jac)
        else:
            low_jac = True

        # Let's make sure all dependant data is finite and real.
        finite_data = all(data is None or (np.all(np.isfinite(data) & np.isreal(data))) for data in dep_data)

        return low_jac and finite_data

    model = draw(models(dependent_vars, independent_vars, parameters, expr_strategy, **kwargs))
    indep_data, param_data = draw(st.tuples(indep_values(model), param_values(model)),
                                  label='independent data, parameters')
    assume(_is_valid_data([indep_data, param_data]))

    dep_data = draw(dep_values(model, indep_data, param_data), label='dependent data')
    return model, param_data, indep_data, dep_data


if __name__ == '__main__':
    DEPENDENT_VARS = st.sampled_from(variables('Y1, Y2'))
    INDEPENDENT_VARS = st.sampled_from(variables('x, y, z'))
    PARAMETERS = st.sampled_from(variables('a, b, c'))
    for _ in range(5):
        print(polynomial_expressions(INDEPENDENT_VARS, PARAMETERS, st.just(1)).example())

    # for _ in range(5):
    #     print(list(map(str, model_with_data(DEPENDENT_VARS, INDEPENDENT_VARS, PARAMETERS).example())))
