from functools import reduce
import operator

import numpy as np

from symfit import Model, parameters, variables, Variable
from symfit.core.support import key2str
import sympy

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from hypothesis import assume, note


NUMBERS = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3)


def _is_sorted(lst, cmp=lambda x, y: x <= y):
    return all(cmp(x, y) for x, y in zip(lst[:-1], lst[1:]))


def _is_number(obj):
    obj = sympy.sympify(obj)
    return obj.is_number


@st.composite
def exprs(draw, symbols, unary_ops, operations, n_leaves, allow_number=False):
    """
    A strategy to generate arbitrary :mod:`Sympy` expressions, based on
    strategies for generating `symbols` and `operations`.

    Parameters
    ----------
    symbols: hypothesis.LazySearchStrategy[sympy.Symbol]
        A strategy that generates sympy symbols. Should probably include
        constants such as 1 and -1.
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
    symbol_st = st.lists(symbols, min_size=n_leaves, max_size=n_leaves)
    # Assert we draw at least one not numerical symbol, unless we allow numbers
    leaves = draw(symbol_st.filter(
        lambda l: allow_number or any(not _is_number(val) for val in l))
    )

    # TODO: allow chaining of unary ops
    unary_ops = st.one_of(st.none(), unary_ops)
    first_ops = draw(st.lists(unary_ops, min_size=len(leaves), max_size=len(leaves)))
    leaves = [op(leaf) if op else leaf for op, leaf in zip(first_ops, leaves)]
 
    while len(leaves) != 1:
        nargs, op, comm = draw(operations)
        # Draw indices for the leaves that get combined by the operation.
        idxs_st = st.lists(st.integers(min_value=0, max_value=len(leaves) - 1),
                           min_size=nargs,
                           max_size=nargs,
                           unique=True)
        # ... and if the operation commutes, make sure the indices are sorted.
        idxs = draw(idxs_st.filter(lambda l: not comm or _is_sorted(l)))
        args = [leaves[idx] for idx in idxs]
        # Remove the drawn symbols from `leaves`
        leaves = [leaf for idx, leaf in enumerate(leaves) if idx not in idxs]
        new_node = op(*args)
        op = draw(unary_ops)
        if op:
            new_node = op(new_node)
        leaves.append(new_node)
    expr = leaves[0]
    assume(allow_number or not expr.is_number)
    return expr


# TODO: ODEModels
@st.composite
def models(draw, dependent_vars, symbols, leaves=5):
    """
    A strategy for generating :class:`symfit.Model` s. 

    Parameters
    ----------
    dependent_vars: hypothesis.LazySearchStrategy[list[symfit.Variable]]
        A strategy that generates lists of Variables. These variables will be
        used as dependent variables.
    symbols: hypothesis.LazySearchStrategy[symfit.Parameter or symfit.Variable]
        A strategy that generates either a parameter or independent variable.
    steps: int
        The number of operations per expression. Larger numbers result is more
        complex models.

    Returns
    -------
    symfit.Model
    """
    constants = st.sampled_from([1, -1, 2, 3, sympy.Rational(1, 2), sympy.pi, sympy.E])
    symbols = st.one_of(constants, symbols)
    ops = st.sampled_from([(2, sympy.Add, True),
                           (2, sympy.Mul, True),
                           # Can cause imaginary numbers (e.g. (-4)**1.2)
                           # (2, sympy.Pow, False),
                           # (2, sympy.log, False),
                          ])
    unary_ops = st.sampled_from([sympy.sin,
                                 sympy.exp,
                                 lambda e: -e,
                                 lambda e: 1/e])
    model_dict = {}
    dep_vars = draw(dependent_vars)
    for idx, dependent_var in enumerate(dep_vars):
        # Dependent vars can depend on other dependent vars: {y1: k, y2: y1+1}.
        # But this can cause cyclical dependency ({y1: y2, y2: y1}), so limit to
        # idx < jdx. Note that this also prevents {y1: y1}.
        other_vars = [dep_var for jdx, dep_var in enumerate(dep_vars) if idx < jdx]
        if other_vars:
            # Can't sample_from and empty list
            symbs = st.one_of(symbols, st.sampled_from(other_vars))
        else:
            symbs = symbols
        expressions = exprs(symbs, unary_ops, ops, leaves, False)
        expr = draw(expressions)
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
        for var in list(group):
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
def independent_data_shapes(draw, model, min_value=1, max_value=10, min_dim=1, max_dim=3):
    """
    Strategy that generates shapes for the independent variables of `model` such
    that variables that need to be broadcast compatible will be the same shape.

    Parameters
    ----------
    model: symfit.Model
        Shapes will be generated for this model's independent variables.
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
    indep_vars = model.independent_vars
    shape_groups = map(tuple, _same_shape_vars(model))
    shapes = npst.array_shapes(min_dim, max_dim, min_value, max_value)
    grp_shape = draw(st.fixed_dictionaries({var_grp: shapes for var_grp in shape_groups}))
    indep_shape = {}
    for vars, shape in grp_shape.items():
        # Not all shapes have to be the same size, but they should be
        # broadcastable. Since it's a pain to find a shape A that is compatible
        # with both B, C and D we'll just say that the last shape can be
        # different.
        for var in vars[:-1]:
            indep_shape[var] = shape
        indep_shape[vars[-1]] = npst.broadcastable_shapes(shape, min_dim, max_dim, min_value, max_value)
    return indep_shape


def numpy_arrays(shape, elements=NUMBERS, **kwargs):
    """
    Basic strategy for generating numpy arrays of the specified `shape`, with
    the specified `elements`.

    Parameters
    ----------
    shape: tuple[int, ...]
        The shape the array should get.
    elements: hypothesis.LazySearchStrategy
        The strategy which should be used to generate elements of the array.
    kwargs: dict[str]
        Additional keyword arguments to be passed to
        :func:`hypothesis.extra.numpy.arrays`

    Returns
    -------
    hypothesis.LazySearchStrategy[np.array]
    """
    return npst.arrays(float, shape, elements=NUMBERS, **kwargs)
    

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
    indep_vars = model.independent_vars
    dep_vars = model.dependent_vars
    params = model.params

    indep_shape = draw(independent_data_shapes(model))

    indep_vals = {}
    for var, shape in indep_shape.items():
        data = numpy_arrays(shape, unique=True).map(np.sort)
        indep_vals[str(var)] = data

    data = draw(st.fixed_dictionaries(indep_vals))

    return data


@st.composite
def param_values(draw, model):
    """
    Strategy for generating values for the parameters of `model`. Will also set
    the values of the parameters.

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
    # You can't provide data for interdependent variables.
    dep_data = {
        var: data
        for var, data in dep_data.items()
        if var in model.dependent_vars
    }
    shapes = {var: data.shape for var, data in dep_data.items() if data is not None}
    sigmas = {
        'sigma_{}'.format(str(var)): st.one_of(
            st.none(),
            st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=5),
            numpy_arrays(shape)
        ) for var, shape in shapes.items()
    }
    sigmas = draw(st.fixed_dictionaries(dict(**sigmas)))
    return dict(**sigmas, **key2str(dep_data))


@st.composite
def model_with_data(draw, dependent_vars, symbols, leaves=5):
    """
    Strategy that generates a model with associated data using the given
    dependent variables.

    Parameters
    ----------
    dependent_vars: hypothesis.LazySearchStrategy[list[symfit.Variable]]
        Strategy to use to generate dependent variables for the model.
    symbols: hypothesis.LazySearchStrategy[symfit.Variable or symfit.Parameter]
        Strategy to use to generate independent variables and parameters for the
        model
    leaves: int
        The number of operations to perform when generating the expressions for
        every dependent component of the model. More steps means complexer
        expressions.

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
    model = draw(models(dependent_vars, symbols, leaves=leaves))
    indep_data, param_data = draw(st.tuples(indep_values(model), 
                                            param_values(model)),
                                  label='independent data, parameters')
    try:
        dep_data = draw(dep_values(model, indep_data, param_data), label='dependent data')
    except ArithmeticError:
        assume(False)  # Some model + data that causes numerical issues
    for data in dep_data.values():
        # In addition, let's make sure all data is finite and real.
        assume(data is None or (np.all(np.isfinite(data)) and np.all(np.isreal(data))))
    return model, param_data, indep_data, dep_data


if __name__ == '__main__':
    DEPENDENT_VARS = st.lists(st.sampled_from(variables('Y1, Y2')), unique=True, min_size=1)
    SYMBOLS = st.sampled_from(parameters('a, b, c') + variables('x, y, z'))

    for _ in range(5):
        print(list(map(str, model_with_data(DEPENDENT_VARS, SYMBOLS).example())))
