from collections import namedtuple, Mapping, OrderedDict, Sequence
import copy
import sys
import warnings
from abc import abstractmethod

import sympy
from sympy.core.relational import Relational
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
from toposort import toposort

from symfit.core.argument import Parameter, Variable
from .support import (
    seperate_symbols, keywordonly, sympy_to_py, key2str, partial,
    cached_property, D
)
from .minimizers import (
    BFGS, SLSQP, LBFGSB, BaseMinimizer, GradientMinimizer, ConstrainedMinimizer,
    ScipyMinimize, MINPACK, ChainedMinimizer, BasinHopping
)
from .objectives import (
    LeastSquares, BaseObjective, MinimizeModel, VectorLeastSquares, LogLikelihood
)
from .fit_results import FitResults

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


def variabletuple(typename, variables, *args, **kwargs):
    """
    Create a :func:`namedtuple` using :class:`~sympy.core.argument.Variable`'s
    whoses names will be used as `field_names`.

    The main reason for using this object is the `_asdict()` method: whereas a
    ``namedtuple`` initiates such an :func:`collections.OrderedDict` with the
    ``field_names`` as keys, this object returns a
    :func:`collections.OrderedDict` which immidiatelly has the ``Variable``
    objects as keys.

    Example::

        >>> x = Variable('x')
        >>> Result = variabletuple('Result', [x])
        >>> res = Result(5.0)
        >>> res._asdict()
        OrderedDict((x, 5.0))

    :param typename: Name of the `variabletuple`.
    :param variables: List of :class:`~sympy.core.argument.Variable`, to be used
        as `field_names`
    :param args: See :func:`collections.namedtuple`
    :param kwargs: See :func:`collections.namedtuple`
    :return: Type ``typename``
    """
    def _asdict(self):
        return OrderedDict(zip(variables, self))

    field_names = [var.name for var in variables]
    named = namedtuple(typename, field_names, *args, **kwargs)
    named._asdict = _asdict
    return named


class ModelError(Exception):
    """
    Raised when a problem occurs with a model.
    """
    pass


class BaseModel(Mapping):
    """
    ABC for ``Model``'s. Makes sure models are iterable.
    Models can be initiated from Mappings or Iterables of Expressions,
    or from an expression directly.
    Expressions are not enforced for ducktyping purposes.
    """
    def __init__(self, model):
        """
        Initiate a Model from a dict::

            a = Model({y: x**2})

        Preferred way of initiating ``Model``, since now you know what the dependent variable is called.

        :param model: dict of ``Expr``, where dependent variables are the keys.
        """
        if not isinstance(model, Mapping):
            try:
                iter(model)
            except TypeError:
                # Model is still a scalar model
                model = [model]
            # TODO: this will break upon deprecating the auto-generation of
            # names for Variables. At this time, a DummyVariable object
            # should be introduced to fulfill the same role.
            # Also, catching the warnings should then be removed, as this is
            # just to prevent the DeprecationWarning from appearing.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = {Variable(): expr for expr in model}
        self._init_from_dict(model)

    def __len__(self):
        """
        :return: the number of dependent variables for this model.
        """
        return len(self.model_dict)

    def __getitem__(self, var):
        """
        Returns the expression belonging to a given dependent variable.

        :param var: Instance of ``Variable``
        :type var: ``Variable``
        :return: The expression belonging to ``var``
        """
        return self.model_dict[var]

    def __iter__(self):
        """
        :return: iterable over self.model_dict
        """
        return iter(self.model_dict)

    def __eq__(self, other):
        """
        ``Model``'s are considered equal when they have the same dependent variables,
        and the same expressions for those dependent variables. The same is defined here
        as passing sympy == for the vars themselves, and as expr1 - expr2 == 0 for the
        expressions. For more info check the `sympy docs <https://github.com/sympy/sympy/wiki/Faq>`_.

        :param other: Instance of ``Model``.
        :return: bool
        """
        if len(self) is not len(other):
            return False
        else:
            for var_1, var_2 in zip(self, other):
                if var_1 != var_2:
                    return False
                else:
                    if not self[var_1].expand() == other[var_2].expand():
                        return False
            else:
                return True

    def __neg__(self):
        """
        :return: new model with opposite sign. Does not change the model in-place,
            but returns a new copy.
        """
        new_model_dict = self.model_dict.copy()
        for key in new_model_dict:
            new_model_dict[key] *= -1
        return self.__class__(new_model_dict)

    def _init_from_dict(self, model_dict):
        """
        Initiate self from a model_dict to make sure attributes such as vars, params are available.

        Creates lists of alphabetically sorted independent vars, dependent vars, sigma vars, and parameters.
        Finally it creates a signature for this model so it can be called nicely. This signature only contains
        independent vars and params, as one would expect.

        :param model_dict: dict of (dependent_var, expression) pairs.
        """
        sort_func = lambda symbol: symbol.name
        self.model_dict = OrderedDict(sorted(model_dict.items(),
                                             key=lambda i: sort_func(i[0])))
        # Everything at the bottom of the toposort is independent, at the top
        # dependent, and the rest interdependent.
        ordered = list(toposort(self.connectivity_mapping))
        independent = ordered.pop(0)
        self.dependent_vars = sorted(ordered.pop(-1), key=sort_func)
        self.interdependent_vars = sorted(
            [item for items in ordered for item in items],
            key=sort_func
        )
        # `independent` contains both params and vars, needs to be separated
        self.params = sorted(
            [s for s in independent if isinstance(s, Parameter)],
            key=sort_func
        )
        self.independent_vars = sorted(
            [s for s in independent
             if not isinstance(s, Parameter) and not s in self],
            key=sort_func
        )

        # Make Variable object corresponding to each depedent var.
        self.sigmas = {var: Variable(name='sigma_{}'.format(var.name))
                       for var in self.dependent_vars}

    @cached_property
    def vars_as_functions(self):
        """
        :return: Turn the keys of this model into :mod:`~sympy.Function`
            objects. This is done recursively so the chain rule can be applied
            correctly. This is done on the basis of `connectivity_mapping`.

            Example: for ``{y: a * x}`` this returns ``{y: y(x, a)}``.
        """
        functions = {}
        # vars first, then params, and alphabetically within each group
        key = lambda arg: [isinstance(arg, Parameter), str(arg)]
        for symbol in self.ordered_symbols:
            if symbol in self.connectivity_mapping:
                connections = self.connectivity_mapping[symbol]
                # Replace the connection by it's function if possible
                connections = [functions.get(connection, connection)
                               for connection in connections]
                connections = sorted(connections, key=key)
                functions[symbol] = sympy.Function(symbol.name)(*(connections))
        return functions

    @cached_property
    def function_dict(self):
        """
        Equivalent to ``self.model_dict``, but with all variables replaced by
        functions if applicable. Sorted by the evaluation order according to
        ``self.ordered_symbols``, not alphabetical like ``self.model_dict``!
        """
        func_tuples = []
        for var, func in self.vars_as_functions.items():
            expr = self.model_dict[var].xreplace(self.vars_as_functions)
            func_tuples.append((func, expr))
        return OrderedDict(func_tuples)

    @cached_property
    def connectivity_mapping(self):
        """
        :return: This property returns a mapping of the interdepencies between
            variables. This is essentially the dict representation of a
            connectivity graph, because working with this dict results in
            cleaner code. Treats variables and parameters on the same footing.
        """
        connectivity = {}
        for var, expr in self.items():
            vars, params = seperate_symbols(expr)
            connectivity[var] = set(vars + params)
        return connectivity

    @property
    def ordered_symbols(self):
        """
        :return: list of all symbols in this model, topologically sorted so they
            can be evaluated in the correct order.

            Within each group of equal priority symbols, we sort by the order of
            the derivative.
        """
        key_func = lambda s: [isinstance(s, sympy.Derivative),
                           isinstance(s, sympy.Derivative) and s.derivative_count]
        symbols = []
        for d in toposort(self.connectivity_mapping):
            symbols.extend(sorted(d, key=key_func))

        return symbols

    @cached_property
    def vars(self):
        """
        :return: Returns a list of dependent, independent and sigma variables, in that order.
        """
        return self.independent_vars + self.dependent_vars + [self.sigmas[var] for var in self.dependent_vars]

    @property
    def bounds(self):
        """
        :return: List of tuples of all bounds on parameters.
        """
        bounds = []
        for p in self.params:
            if p.fixed:
                if p.value >= 0.0:
                    bounds.append([np.nextafter(p.value, 0), p.value])
                else:
                    bounds.append([p.value, np.nextafter(p.value, 0)])
            else:
                bounds.append([p.min, p.max])
        return bounds

    @property
    def shared_parameters(self):
        """
        :return: bool, indicating if parameters are shared between the vector
            components of this model.
        """
        if len(self) == 1:  # Not a vector
            return False
        else:
            params_thusfar = []
            for component in self.values():
                vars, params = seperate_symbols(component)
                if set(params).intersection(set(params_thusfar)):
                    return True
                else:
                    params_thusfar += params
            else:
                return False

    @property
    def free_params(self):
        """
        :return: ordered list of the subset of variable params
        """
        return [p for p in self.params if not p.fixed]

    def __str__(self):
        """
        Printable representation of a Mapping model.

        :return: str
        """
        template = "{}({}; {}) = {}"
        parts = []
        for var, expr in self.items():
            params_sorted = sorted((x for x in self.connectivity_mapping[var]
                                    if isinstance(x, Parameter)),
                                   key=lambda x: x.name)
            vars_sorted = sorted((x for x in self.connectivity_mapping[var]
                                  if x not in params_sorted),
                                 key=lambda x: x.name)
            parts.append(template.format(
                    var,
                    ', '.join([x.name for x in vars_sorted]),
                    ', '.join([x.name for x in params_sorted]),
                    expr
                )
            )
        return '[{}]'.format(",\n ".join(parts))

    def __getstate__(self):
        # Remove cached_property values from the state, they need to be
        # re-calculated after pickle.
        state = self.__dict__.copy()
        del state['__signature__']
        for key in self.__dict__:
            if key.startswith(cached_property.base_str):
                del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__signature__ = self._make_signature()


class BaseNumericalModel(BaseModel):
    """
    ABC for Numerical Models. These are models whose components are generic
    python callables.
    """
    @keywordonly(connectivity_mapping=None)
    def __init__(self, model, independent_vars=None, params=None, **kwargs):
        """
        :param model: dict of ``callable``, where dependent variables are the
            keys. If instead of a dict a (sequence of) ``callable`` is provided,
            it will be turned into a dict automatically.
        :param independent_vars: The independent variables of the  model.
            (Deprecated, use ``connectivity_mapping`` instead.)
        :param params: The parameters of the model.
            (Deprecated, use ``connectivity_mapping`` instead.)
        :param connectivity_mapping: Mapping indicating the dependencies of
            every variable in the model. For example, a model_dict
            {y: a * x + b} has a connectivity_mapping {y: {x, a, b}}. Note that
            the values of this dict have to be sets.
        """
        connectivity_mapping = kwargs.pop('connectivity_mapping')
        if connectivity_mapping is None and \
                independent_vars is not None and params is not None:
            # Make model into a mapping if needed.
            if not isinstance(model, Mapping):
                try:
                    iter(model)
                except TypeError:
                    model = [model]  # make model iterable

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = {Variable(): expr for expr in model}
            warnings.warn(DeprecationWarning(
                '`independent_vars` and `params` have been deprecated.'
                ' Use `connectivity_mapping` instead.'
            ))
            self.independent_vars = sorted(independent_vars, key=str)
            self.params = sorted(params, key=str)
            self.connectivity_mapping = {var: set(independent_vars + params)
                                         for var in model}
        elif connectivity_mapping:
            self.connectivity_mapping = connectivity_mapping
        else:
            raise TypeError('Provide either `connectivity_mapping` (preferred) '
                            'or `independent_vars` and `params` (deprecated).')
        super(BaseNumericalModel, self).__init__(model)

    @property
    def connectivity_mapping(self):
        return self._connectivity_mapping

    @connectivity_mapping.setter
    def connectivity_mapping(self, value):
        self._connectivity_mapping = value

    def __eq__(self, other):
        raise NotImplementedError(
            'Equality checking for {} is ambiguous.'.format(self.__class__.__name__)
        )

    def __neg__(self):
        """
        :return: new model with opposite sign. Does not change the model in-place,
            but returns a new copy.
        """
        new_model_dict = {}
        for key, callable_expr in self.model_dict.values():
            new_model_dict[key] = lambda *args, **kwargs: - callable_expr(*args, **kwargs)
        return self.__class__(new_model_dict)

    @property
    def shared_parameters(self):
        """
        BaseNumericalModel's cannot infer if parameters are shared.
        """
        raise NotImplementedError(
            'Shared parameters can not be inferred for {}'.format(self.__class__.__name__)
        )


class BaseCallableModel(BaseModel):
    """
    Baseclass for callable models. A callable model is expected to have
    implemented a `__call__` method which evaluates the model.
    """
    def eval_components(self, *args, **kwargs):
        """
        :return: evaluated lambda functions of each of the components in
            model_dict, to be used in numerical calculation.
        """
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        kwargs = bound_arguments.arguments  # Only work with kwargs
        components = dict(zip(self, self.numerical_components))
        # Evaluate the variables in topological order.
        for symbol in self.ordered_symbols:
            if symbol.name not in kwargs:
                dependencies = self.connectivity_mapping[symbol]
                dependencies_kwargs = {d.name: kwargs[d.name]
                                       for d in dependencies}
                kwargs[symbol.name] = components[symbol](**dependencies_kwargs)

        return [np.atleast_1d(kwargs[var.name]) for var in self]

    def numerical_components(self):
        """
        :return: A list of callables corresponding to each of the components
            of the model.
        """
        raise NotImplementedError(
            ('No `numerical_components` is defined for object of type {}. '
             'Implement either `numerical_components`, or change '
             '`eval_components` so it no longer calls '
             '`numerical_components`.').format(self.__class__)
        )

    def _make_signature(self):
        # Handle args and kwargs according to the allowed names.
        parameters = [
            # Note that these are inspect_sig.Parameter's, not symfit parameters!
            inspect_sig.Parameter(arg.name,
                                  inspect_sig.Parameter.POSITIONAL_OR_KEYWORD)
            for arg in self.independent_vars + self.params
        ]
        return inspect_sig.Signature(parameters=parameters)

    def _init_from_dict(self, model_dict):
        super(BaseCallableModel, self)._init_from_dict(model_dict)
        self.__signature__ = self._make_signature()

    def __call__(self, *args, **kwargs):
        """
        Evaluate the model for a certain value of the independent vars and parameters.
        Signature for this function contains independent vars and parameters, NOT dependent and sigma vars.

        Can be called with both ordered and named parameters. Order is independent vars first, then parameters.
        Alphabetical order within each group.

        :param args:
        :param kwargs:
        :return: A namedtuple of all the dependent vars evaluated at the desired point. Will always return a tuple,
            even for scalar valued functions. This is done for consistency.
        """
        Ans = variabletuple('Ans', self)
        return Ans(*self.eval_components(*args, **kwargs))


class BaseGradientModel(BaseCallableModel):
    """
    Baseclass for models which have a gradient. Such models are expected to
    implement an `eval_jacobian` function.

    Any subclass of this baseclass which does not implement its own
    `eval_jacobian` will inherit a finite difference gradient.
    """
    @keywordonly(dx=1e-8)
    def finite_difference(self, *args, **kwargs):
        """
        Calculates a numerical approximation of the Jacobian of the model using
        the sixth order central finite difference method. Accepts a `dx`
        keyword to tune the relative stepsize used.
        Makes 6*n_params calls to the model.

        :return: A numerical approximation of the Jacobian of the model as a
                 list with length n_components containing numpy arrays of shape
                 (n_params, n_datapoints)
        """
        # See also: scipy.misc.derivative. It might be convinced to work, but
        # it will make way too many function evaluations
        dx = kwargs.pop('dx')
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        var_vals = [bound_arguments.arguments[var.name] for var in self.independent_vars]
        param_vals = [bound_arguments.arguments[param.name] for param in self.params]
        param_vals = np.array(param_vals, dtype=float)
        f = partial(self, *var_vals)
        # See also: scipy.misc.central_diff_weights
        factors = np.array((3/2., -3/5., 1/10.))
        orders = np.arange(1, len(factors) + 1)
        out = []
        # TODO: Dark numpy magic. Needs an extra dimension in out, and a sum
        #       over the right axis at the end.

        # We can't make the output arrays yet, since we don't know the size of
        # the components. So put a sentinel value.
        out = None

        for param_idx, param_val in enumerate(param_vals):
            for order, factor in zip(orders, factors):
                h = np.zeros(len(self.params))
                # Note: stepsize (h) depends on the parameter values...
                h[param_idx] = dx * order
                if abs(param_val) >= 1e-7:
                    # ...but it'd better not be (too close to) 0.
                    h[param_idx] *= param_val
                up = f(*(param_vals + h))
                down = f(*(param_vals - h))
                if out is None:
                    # Initialize output arrays. Now that we evaluated f, we
                    # know the size of our data.
                    out = []
                    # out is a list  of length Ncomponents with numpy arrays of
                    # shape (Nparams, Ndata). Part of our misery comes from the
                    # fact that the length of the data may be different for all
                    # the components. Numpy doesn't like ragged arrays, so make
                    # a list of arrays.
                    for comp_idx in range(len(self)):
                        try:
                            len(up[comp_idx])
                        except TypeError:  # output[comp_idx] is a number
                            data_shape = (1,)
                        else:
                            data_shape = up[comp_idx].shape
                        # Initialize at 0 so we can += all the contributions
                        param_grad = np.zeros([len(self.params)] + list(data_shape), dtype=float)
                        out.append(param_grad)
                for comp_idx in range(len(self)):
                    diff = up[comp_idx] - down[comp_idx]
                    out[comp_idx][param_idx, :] += factor * diff / (2 * h[param_idx])
        return out

    def eval_jacobian(self, *args, **kwargs):
        """
        :return: The jacobian matrix of the function.
        """
        Ans = variabletuple('Ans', self)
        return Ans(*self.finite_difference(*args, **kwargs))


class CallableNumericalModel(BaseCallableModel, BaseNumericalModel):
    """
    Callable model, whose components are callables provided by the user.
    This allows the user to provide the components directly.

    Example::

        x, y = variables('x, y')
        a, b = parameters('a, b')
        numerical_model = CallableNumericalModel(
            {y: lambda x, a, b: a * x + b},
            independent_vars=[x],
            params=[a, b]
        )

    This is identical in functionality to the more traditional::

        x, y = variables('x, y')
        a, b = parameters('a, b')
        model = CallableModel({y: a * x + b})

    but allows power-users a lot more freedom while still interacting
    seamlessly with the :mod:`symfit` API.

    .. note:: All of the callables must accept all of the ``independent_vars``
        and  ``params`` of the model as arguments, even if not all of them are
        used by every callable.
    """
    @cached_property
    def numerical_components(self):
        return [expr if not isinstance(expr, sympy.Expr) else
                sympy_to_py(expr, self.connectivity_mapping[var], [])
                for var, expr in self.items()]


class CallableModel(BaseCallableModel):
    """
    Defines a callable model. The usual rules apply to the ordering of the
    arguments:

    * first independent variables, then dependent variables, then parameters.
    * within each of these groups they are ordered alphabetically.
    """
    @cached_property
    def numerical_components(self):
        """
        :return: lambda functions of each of the analytical components in
            model_dict, to be used in numerical calculation.
        """
        Ans = variabletuple('Ans', self.keys())
        # All components must feature the independent vars and params, that's
        # the API convention. But for those components which also contain
        # interdependence, we add those vars
        components = []
        for var, expr in self.items():
            dependencies = self.connectivity_mapping[var]
            # vars first, then params, and alphabetically within each group
            key = lambda arg: [isinstance(arg, Parameter), str(arg)]
            ordered = sorted(dependencies, key=key)
            components.append(sympy_to_py(expr, ordered, []))
        return Ans(*components)


class GradientModel(CallableModel, BaseGradientModel):
    """
    Analytical model which has an analytically computed Jacobian.
    """
    def __init__(self, *args, **kwargs):
        super(GradientModel, self).__init__(*args, **kwargs)
        self.jacobian_model = jacobian_from_model(self)

    @cached_property
    def jacobian(self):
        """
        :return: Jacobian filled with the symbolic expressions for all the
            partial derivatives. Partial derivatives are of the components of
            the function with respect to the Parameter's, not the independent
            Variable's. The return shape is a list over the models components,
            filled with tha symbolical jacobian for that component, as a list.
        """
        jac = []
        for var, expr in self.items():
            jac.append([])
            for param in self.params:
                partial_dv = D(var, param)
                jac[-1].append(self.jacobian_model[partial_dv])
        return jac

    def eval_jacobian(self, *args, **kwargs):
        """
        :return: Jacobian evaluated at the specified point.
        """
        eval_jac_dict = self.jacobian_model(*args, **kwargs)._asdict()
        # Take zero for component which are not present, happens for Constraints
        jac = [[eval_jac_dict.get(D(var, param), 0)
                for param in self.params]
            for var in self
        ]

        # Use numpy to broadcast these arrays together and then stack them along
        # the parameter dimension. We do not include the component direction in
        # this, because the components can have independent shapes.
        for idx, comp in enumerate(jac):
            jac[idx] = np.stack(np.broadcast_arrays(*comp))

        Ans = variabletuple('Ans', self.keys())
        return Ans(*jac)


class Model(GradientModel):
    """
    Model represents a symbolic function and all it's derived properties such as sum of squares, jacobian etc.
    Models can be initiated from several objects::

        a = Model({y: x**2})
        b = Model(y=x**2)

    Models are callable. The usual rules apply to the ordering of the arguments:

    * first independent variables, then dependent variables, then parameters.
    * within each of these groups they are ordered alphabetically.

    Models are also iterable, behaving as their internal model_dict. In the example above,
    a[y] returns x**2, len(a) == 1, y in a == True, etc.
    """
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.hessian_model = hessian_from_model(self)

    @property
    def hessian(self):
        """
        :return: Hessian filled with the symbolic expressions for all the
            second order partial derivatives. Partial derivatives are taken with
            respect to the Parameter's, not the independent Variable's.
        """
        return [[[sympy.diff(partial_dv, param) for param in self.params]
                 for partial_dv in comp] for comp in self.jacobian]

    def eval_hessian(self, *args, **kwargs):
        """
        :return: Hessian evaluated at the specified point.
        """
        # Evaluate the hessian model and use the resulting Ans namedtuple as a
        # dict. From this, take the relevant components.
        eval_hess_dict = self.hessian_model(*args, **kwargs)._asdict()
        hess = [[[eval_hess_dict.get(D(var, p1, p2), 0)
                    for p2 in self.params]
                for p1 in self.params]
            for var in self
        ]
        # Use numpy to broadcast these arrays together and then stack them along
        # the parameter dimension. We do not include the component direction in
        # this, because the components can have independent shapes.
        for idx, comp in enumerate(hess):
            hess[idx] = np.stack(np.broadcast_arrays(*comp))

        Ans = variabletuple('Ans', self.keys())
        return Ans(*hess)


class TaylorModel(Model):
    """
    A first-order Taylor expansion of a model around given parameter values (:math:`p_0`).
    Is used by NonLinearLeastSquares. Currently only a first order expansion is implemented.
    """
    def __init__(self, model):
        """
        Make a first order Taylor expansion of ``model``.

        :param model: Instance of ``Model``
        """
        params_0 = OrderedDict(
            [(p, Parameter(name='{}_0'.format(p.name))) for p in model.params]
        )
        model_dict = {}
        for (var, component), jacobian_vec in zip(model.items(), model.jacobian):
            linear = component.subs(params_0.items())
            # params_0 is assumed OrderedDict!
            for (p, p0), jac in zip(params_0.items(), jacobian_vec):
                linear += jac.subs(params_0.items()) * (p - p0)
            model_dict[var] = linear
        self.params_0 = params_0
        super(TaylorModel, self).__init__(model_dict)
        self.model_dict_orig = copy.copy(self.model_dict)

    @property
    def params(self):
        """
        params returns only the `free` parameters. Strictly speaking, the expression for a
        ``TaylorModel`` contains both the parameters :math:`\\vec{p}` and :math:`\\vec{p_0}`
        around which to expand, but params should only give :math:`\\vec{p}`. To get a
        mapping to the :math:`\\vec{p_0}`, use ``.params_0``.
        """
        return [p for p in self._params if p not in self.params_0.values()]

    @params.setter
    def params(self, items):
        self._params = items

    @property
    def p0(self):
        """
        Property of the :math:`p_0` around which to expand. Should be set by the names of
        the parameters themselves.

        Example::

            a = Parameter()
            x, y = variables('x, y')
            model = TaylorModel({y: sin(a * x)})

            model.p0 = {a: 0.0}

        """
        return self._p0

    @p0.setter
    def p0(self, expand_at):
        self._p0 = {self.params_0[p]: float(value) for p, value in expand_at.items()}
        for var in self.model_dict_orig:
            self.model_dict[var] = self.model_dict_orig[var].subs(self.p0.items())

    def __str__(self):
        """
        When printing a TaylorModel, the point around which the expansion took place is included.

        For example, a Taylor expansion of {y: sin(w * x)} at w = 0 would be printed as::

            @{w: 0.0} -> y(x; w) = w*x
        """
        sup = super(TaylorModel, self).__str__()
        return '@{} -> {}'.format(self.p0, sup)

    def eval_jacobian(self, *args, **kwargs):
        """
        See :meth:`~symfit.core.fit.Model.eval_jacobian`.
        """
        kwargs.update(self.p0)
        return super(TaylorModel, self).eval_jacobian(*args, **key2str(kwargs))

    def eval_hessian(self, *args, **kwargs):
        """
        See :meth:`~symfit.core.fit.Model.eval_hessian`.
        """
        kwargs.update(self.p0)
        return super(TaylorModel, self).eval_hessian(*args, **key2str(kwargs))


class Constraint(Model):
    """
    Constraints are a special type of model in that they have a type: >=, == etc.
    They are made to have lhs - rhs == 0 of the original expression.

    For example, Eq(y + x, 4) -> Eq(y + x - 4, 0)

    Since a constraint belongs to a certain model, it has to be initiated with knowledge of it's parent model.
    This is important because all ``numerical_`` methods are done w.r.t. the parameters and variables of the parent
    model, not the constraint! This is because the constraint might not have all the parameter or variables that the
    model has, but in order to compute for example the Jacobian we still want to derive w.r.t. all the parameters,
    not just those present in the constraint.
    """
    @keywordonly(params=None)
    def __init__(self, constraint, model=None, **kwargs):
        """
        :param constraint: constraint that model should be subjected to.
        :param model: A constraint needs to be initiated with all the parameters
            of the corresponiding model, either by providing the model or the
            parameters.
        :param params: list of parameters in case no ``model`` is provided.
        """
        params = kwargs.pop('params')
        if isinstance(constraint, Relational):
            self.constraint_type = type(constraint)
            if model is None and params is not None:
                pass
            elif isinstance(model, BaseModel):
                self.model = model
                params = self.model.params
            else:
                raise TypeError('The model argument must be of type Model.')
            super(Constraint, self).__init__(constraint.lhs - constraint.rhs)

            # Update the signature to accept all vars and parms of the model
            self.params = params
            self.__signature__ = self._make_signature()
            # Update the jacobian and hessian model as well
            self.jacobian_model.params = params
            self.hessian_model.params = params
            self.jacobian_model.__signature__ = self.jacobian_model._make_signature()
            self.hessian_model.__signature__ = self.hessian_model._make_signature()
        else:
            raise TypeError('Constraints have to be initiated with a subclass of sympy.Relational')

    def __neg__(self):
        """
        :return: new model with opposite sign. Does not change the model in-place,
            but returns a new copy.
        """
        new_constraint = self.constraint_type( - self.model_dict[self.dependent_vars[0]])
        return self.__class__(new_constraint, self.model)


class TakesData(object):
    """
    An base class for everything that takes data. Most importantly, it takes care
    of linking the provided data to variables. The allowed variables are extracted
    from the model.
    """
    @keywordonly(absolute_sigma=None)
    def __init__(self, model, *ordered_data, **named_data):
        """
        :param model: (dict of) sympy expression or ``Model`` object.
        :param bool absolute_sigma: True by default. If the sigma is only used
            for relative weights in your problem, you could consider setting it to
            False, but if your sigma are measurement errors, keep it at True.
            Note that curve_fit has this set to False by default, which is wrong in
            experimental science.
        :param ordered_data: data for dependent, independent and sigma variables. Assigned in
            the following order: independent vars are assigned first, then dependent
            vars, then sigma's in dependent vars. Within each group they are assigned in
            alphabetical order.
        :param named_data: assign dependent, independent and sigma variables data by name.

        Standard deviation can be provided to any variable. They have to be prefixed
        with sigma\_. For example, let x be a Variable. Then sigma_x will give the
        stdev in x.
        """
        absolute_sigma = named_data.pop('absolute_sigma')
        if isinstance(model, BaseModel):
            self.model = model
        else:
            self.model = Model(model)

        # Handle ordered_data and named_data according to the allowed names.
        var_names = [var.name for var in self.model.vars]
        parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
            # inspect_sig.Parameter(name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD, default=1 if name.startswith('sigma_') else None)
            inspect_sig.Parameter(name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD, default=None)
                for name in var_names
        ]

        signature = inspect_sig.Signature(parameters=parameters)
        try:
            bound_arguments = signature.bind(*ordered_data, **named_data)
        except TypeError as err:
            for var in self.model.vars:
                if var.name.startswith(Variable._argument_name):
                    raise type(err)(str(err) + '. Some of your Variable\'s are unnamed. That might be the cause of this Error: make sure you use e.g. x = Variable(\'x\')')
            else:
                raise err
        # Include default values in bound_argument object
        for param in signature.parameters.values():
            if param.name not in bound_arguments.arguments:
                bound_arguments.arguments[param.name] = param.default

        original_data = bound_arguments.arguments   # ordereddict of the data
        self.data = OrderedDict((var, original_data[var.name]) for var in self.model.vars)
        self.data.update({var: None for var in self.model.interdependent_vars})
        # Change the type to array if no array operations are supported.
        # We don't want to break duck-typing, hence the try-except.
        for var, dataset in self.data.items():
            try:
                dataset**2
            except TypeError:
                if dataset is not None:
                    self.data[var] = np.array(dataset)
        self.sigmas_provided = any(value is not None for value in self.sigma_data.values())

        # Replace sigmas that are constant by an array of that constant
        for var, sigma in self.model.sigmas.items():
            try:
                iter(self.data[sigma])
            except TypeError:
                if self.data[var] is None and self.data[sigma] is None:
                    if len(self.data_shapes[1]) == 1:
                        # The shape of the dependent vars is unique across dependent vars.
                        # This means we can just assume this shape.
                        self.data[sigma] = np.ones(self.data_shapes[1][0])
                    else: pass # No stdevs can be calculated
                if self.data[var] is not None and self.data[sigma] is None:
                    self.data[sigma] = np.ones(self.data[var].shape)
                elif self.data[var] is not None:
                    self.data[sigma] *= np.ones(self.data[var].shape)

        # If user gives a preference, use that. Otherwise, use True if at least one sigma is
        # given, False if no sigma is given.
        if absolute_sigma is not None:
            self.absolute_sigma = absolute_sigma
        else:
            for sigma in self.sigma_data:
                # Check if the user provided sigmas in the original data.
                # If so, interpret sigmas as measurement errors
                if original_data[sigma.name] is not None:
                    self.absolute_sigma = True
                    break
            else:
                self.absolute_sigma = False

    @property
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var])
                           for var in self.model.dependent_vars)

    @property
    def independent_data(self):
        """
        Read-only Property

        :return: Data belonging to each independent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var]) for var in self.model.independent_vars)

    @property
    def sigma_data(self):
        """
        Read-only Property

        :return: Data belonging to each sigma variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        sigmas = self.model.sigmas
        return OrderedDict((sigmas[var], self.data[sigmas[var]])
                           for var in self.model.dependent_vars)

    @property
    def data_shapes(self):
        """
        Returns the shape of the data. In most cases this will be the same for
        all variables of the same type, if not this raises an Exception.

        Ignores variables which are set to None by design so we know that those
        None variables can be assumed to have the same shape as the other in
        calculations where this is needed, such as the covariance matrix.

        :return: Tuple of all independent var shapes, dependent var shapes.
        """
        independent_shapes = []
        for var, data in self.independent_data.items():
            if data is not None:
                independent_shapes.append(data.shape)

        dependent_shapes = []
        for var, data in self.dependent_data.items():
            if data is not None:
                dependent_shapes.append(data.shape)

        return list(set(independent_shapes)), list(set(independent_shapes))

    @property
    def initial_guesses(self):
        """
        :return: Initial guesses for every parameter.
        """
        return np.array([param.value for param in self.model.params])


class BaseFit(TakesData):
    """
    Abstract base class for all fitting objects.
    """
    def execute(self, *args, **kwargs):
        """
        Every fit object has to define an execute method.
        Any * and ** arguments will be passed to the fitting module that is being wrapped, e.g. leastsq.

        :args kwargs:
        :return: Instance of FitResults
        """
        raise NotImplementedError('Every subclass of BaseFit must have an execute method.')

    def error_func(self, *args, **kwargs):
        """
        Every fit object has to define an error_func method, giving the function to be minimized.
        """
        raise NotImplementedError('Every subclass of BaseFit must have an error_func method.')

    def eval_jacobian(self, *args, **kwargs):
        """
        Every fit object has to define an eval_jacobian method, giving the jacobian of the
        function to be minimized.
        """
        raise NotImplementedError('Every subclass of BaseFit must have an eval_jacobian method.')


class HasCovarianceMatrix(TakesData):
    """
    Mixin class for calculating the covariance matrix for any model that has a
    well-defined Jacobian :math:`J`. The covariance is then approximated as
    :math:`J^T W J`, where W contains the weights of each data point.

    Supports vector valued models, but is unable to estimate covariances for
    those, just variances. Therefore, take the result with a grain of salt for
    vector models.
    """
    def covariance_matrix(self, best_fit_params):
        """
        Given best fit parameters, this function finds the covariance matrix.
        This matrix gives the (co)variance in the parameters.

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        :return: covariance matrix.
        """
        if not hasattr(self.model, 'eval_jacobian'):
            return None
        if any(element is None for element in self.sigma_data.values()):
            # If one of the sigma's was explicitly set to None, we are unable
            # to determine the covariances.
            return np.array(
                [[float('nan') for p in self.model.params] for p in self.model.params]
            )
        try:
            if isinstance(self.objective, LogLikelihood):
                # Loglikelihood is a special case that needs to be considered
                # separately, see #138
                hess = self.objective.eval_hessian(**key2str(best_fit_params))
                cov_mat = np.linalg.inv(hess)
                return cov_mat
            if len(set(arr.shape for arr in self.sigma_data.values())) == 1:
                # Shapes of all sigma data identical
                return self._cov_mat_equal_lenghts(best_fit_params=best_fit_params)
            else:
                return self._cov_mat_unequal_lenghts(best_fit_params=best_fit_params)
        except np.linalg.LinAlgError:
            return None

    def _reduced_residual_ss(self, best_fit_params, flatten=False):
        """
        Calculate the residual Sum of Squares divided by the d.o.f..

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        :param flatten: when `True`, return the total sum of squares (SS).
            If `False`, return the componentwise SS.
        :return: The reduced residual sum of squares.
        """
        # popt = [best_fit_params[p.name] for p in self.model.params]
        # Rescale the covariance matrix with the residual variance
        if isinstance(self.objective, (VectorLeastSquares, LeastSquares)):
            ss_res = self.objective(flatten_components=flatten,
                                    **key2str(best_fit_params))
        else:
            ss_res = self.objective(**key2str(best_fit_params))

        if isinstance(self.objective, VectorLeastSquares):
            ss_res = np.sum(ss_res**2)

        degrees_of_freedom = 0 if flatten else []
        for data in self.dependent_data.values():
            if flatten:
                if data is not None:
                    degrees_of_freedom += np.product(data.shape)
            else:
                if data is not None:
                    degrees_of_freedom.append(np.product(data.shape))
                    # degrees_of_freedom = np.product(data.shape) - len(popt)
                    # break
                else: # the correspoding component in ss_res will be 0 so it is ok to add any non-zero number.
                    degrees_of_freedom.append(len(best_fit_params) + 1)
        degrees_of_freedom = np.array(degrees_of_freedom)
        s_sq = ss_res / (degrees_of_freedom - len(best_fit_params))
        # s_sq = ss_res / degrees_of_freedom

        return s_sq

    def _cov_mat_equal_lenghts(self, best_fit_params):
        """
        If all the data arrays are of equal size, use this method. This will
        typically be the case, and this method is a lot faster because it allows
        for numpy magic.

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        """
        # Stack in a new dimension, and make this the first dim upon indexing.
        sigma = np.concatenate([arr[np.newaxis, ...] for arr in self.sigma_data.values()], axis=0)

        # Weight matrix. Since it should be a diagonal matrix, we just remember
        # this and multiply it elementwise for efficiency.
        # It is also rescaled by the reduced residual ss in case of absolute_sigma==False
        if self.absolute_sigma:
            W = 1/sigma**2
        else:
            s_sq = self._reduced_residual_ss(best_fit_params, flatten=False)
            W = 1/sigma**2/s_sq[:, np.newaxis]

        kwargs = key2str(best_fit_params)
        kwargs.update(key2str(self.independent_data))

        jac = self.model.eval_jacobian(**kwargs)
        # Drop the axis which correspond to dependent vars which have been
        # set to None. jac also contains interdependent_vars, hence the .get
        mask = [self.dependent_data.get(var, None) is not None
                for var in self.model.dependent_vars]
        jac = np.array([comp for comp, relevant in zip(jac, mask) if relevant])
        W = W[mask]
        if jac.shape[0] == 0:
            return None

        # Order jacobian as param, component, datapoint
        jac = np.swapaxes(jac, 0, 1)
        if not self.independent_data:
            jac = jac * np.ones_like(W)
        # Dot away all but the parameter dimension!
        try:
            cov_matrix_inv = np.tensordot(W*jac, jac, (range(1, jac.ndim), range(1, jac.ndim)))
        except ValueError as err:
            # If this fails because the shape of the jacobian could not be
            # properly estimated, then we remedy this. If not, the error is reraised.
            if jac.shape[-1] == 1:
                # Take the shape of the dependent data
                dependent_shape = self.data_shapes[1][0]
                # repeat the object along the last axis to match the shape
                new_jac = np.repeat(jac, np.product(dependent_shape), -1)
                jac = new_jac.reshape((jac.shape[0], jac.shape[1]) + dependent_shape)
                cov_matrix_inv = np.tensordot(W * jac, jac, (range(1, jac.ndim), range(1, jac.ndim)))
            else:
                raise err
        cov_matrix = np.linalg.inv(cov_matrix_inv)
        return cov_matrix

    def _cov_mat_unequal_lenghts(self, best_fit_params):
        """
        If the data arrays are of unequal size, use this method. Less efficient
        but more general than the method for equal size datasets.
        """
        sigma = list(self.sigma_data.values())
        # Weight matrix. Since it should be a diagonal matrix, we just remember
        # this and multiply it elementwise for efficiency.
        if self.absolute_sigma:
            W = [1/s**2 for s in sigma]
        else:
            s_sq = self._reduced_residual_ss(best_fit_params, flatten=False)
            # W = 1/sigma**2/s_sq[:, np.newaxis]
            W = [1/s**2/res for s, res in zip(sigma, s_sq)]

        kwargs = key2str(best_fit_params)
        kwargs.update(key2str(self.independent_data))

        jac = self.model.eval_jacobian(**kwargs)
        data_len = max(j.shape[1] for j in jac)
        data_len = max(data_len, max(len(w) for w in W))
        W_full = np.zeros((len(W), data_len), dtype=float)
        jac_full = np.zeros((len(jac), jac[0].shape[0], data_len), dtype=float)
        for idx, (j, w) in enumerate(zip(jac, W)):
            if not self.independent_data:
                j = j * np.ones_like(w)
            jac_full[idx, :, :j.shape[1]] = j
            W_full[idx, :len(w)] = w
        jac = jac_full
        W = W_full
        # Order jacobian as param, component, datapoint
        jac = np.swapaxes(jac, 0, 1)
        # Weigh each component with its respective weight.

        # Build the inverse cov_matrix.
        cov_matrix_inv = []
        cov_matrix_inv = np.tensordot(W*jac, jac, (range(1, jac.ndim), range(1, jac.ndim)))
        cov_matrix = np.linalg.inv(cov_matrix_inv)
        return cov_matrix


class LinearLeastSquares(BaseFit):
    """
    Experimental. Solves the linear least squares problem analytically. Involves no iterations
    or approximations, and therefore gives the best possible fit to the data.

    The ``Model`` provided has to be linear.

    Currently, since this object still has to mature, it suffers from the following limitations:

    * It does not check if the model can be linearized by a simple substitution.
      For example, exp(a * x) -> b * exp(x). You will have to do this manually.
    * Does not use bounds or guesses on the ``Parameter``'s. Then again, it doesn't have to,
      since you have an exact solution. No guesses required.
    * It only works with scalar functions. This is strictly enforced.

    .. _Blobel: http://www.desy.de/~blobel/blobel_leastsq.pdf
    .. _Wiki: https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)
    """
    def __init__(self, *args, **kwargs):
        """
        :raises: ``ModelError`` in case of a non-linear model or when a vector
            valued function is provided.
        """
        super(LinearLeastSquares, self).__init__(*args, **kwargs)
        if not self.is_linear(self.model):
            raise ModelError('This Model is non-linear. Please use NonLinearLeastSquares instead.')
        elif len(self.model) > 1:
            raise ModelError('Currently only scalar valued functions are supported.')
        self.leastsquares_model = leastsquares_from_model(self.model)

    @staticmethod
    def is_linear(model):
        """
        Test whether model is of linear form in it's parameters.

        Currently this function does not recognize if a model can be considered linear
        by a simple substitution, such as exp(k x) = k' exp(x).

        .. _Wiki: https://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)

        :param model: ``Model`` instance
        :return: True or False
        """
        terms = {}
        for var, expr in model.items():
            terms.update(sympy.collect(sympy.expand(expr), model.params, evaluate=False))
        difference = set(terms.keys()) ^ set(model.params) # Symmetric difference
        return not difference or difference == {1}  # Either no difference or it still contains O(1) terms

    def best_fit_params(self):
        """
        Fits to the data and returns the best fit parameters.

        :return: dict containing parameters and their best-fit values.
        """
        terms_per_component = []
        for expr in self.leastsquares_model.jacobian[0]:
            # Collect terms linear in the parameters. Returns a dict with parameters and
            # their prefactor as function of variables. Includes O(1)
            terms = sympy.collect(sympy.expand(expr), self.model.params, evaluate=False)
            # Evaluate every term separately and 'sum out' the variables. This results in
            # a system that is very easy to solve.
            for param in terms:
                terms[param] = np.sum(terms[param](**key2str(self.data)))

            terms_per_component.append(terms)

        # Reconstruct the linear system.
        system = [sum(factor*param for param, factor in terms.items()) for terms in terms_per_component]
        sol = sympy.solve(system, self.model.params, dict=True)
        try:
            assert len(sol) == 1 # Future Homer should think about what to do with multiple/no solutions
        except AssertionError:
            raise Exception('Got an unexpected number of solutions:', len(sol))
        return sol[0] # Dict of param: value pairs.

    def covariance_matrix(self, best_fit_params):
        """
        Given best fit parameters, this function finds the covariance matrix.
        This matrix gives the (co)variance in the parameters.

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        :return: covariance matrix.
        """
        # The rest of this method is concerned with determining the covariance matrix
        # Weight matrix. Diagonal matrix for now.
        sigma = list(self.sigma_data.values())[0]
        W = np.diag(1/sigma.flatten()**2)

        # Calculate the covariance matrix from the Jacobian X @ best_params.
        # In X, we do NOT sum over the components by design. This is because
        # it has to be contracted with W, the weight matrix.
        kwargs = {p.name: float(value) for p, value in best_fit_params.items()}
        kwargs.update(self.independent_data)
        X = np.atleast_2d(self.model.eval_jacobian(**key2str(kwargs))[0])
        X = np.ones(sigma.shape[0]) * X
        cov_matrix = np.linalg.inv(X.dot(W).dot(X.T))
        if not self.absolute_sigma:
            kwargs.update(self.data)
            # Sum of squared residuals. To be honest, I'm not sure why ss_res does not give the
            # right result but by using the chi_squared the results are compatible with curve_fit.
            S = np.sum(self.leastsquares_model(**key2str(kwargs)), dtype=float) / (len(W) - len(self.model.params))
            cov_matrix *= S

        return cov_matrix

    def execute(self):
        """
        Execute an analytical (Linear) Least Squares Fit. This object works by symbolically
        solving when :math:`\\nabla \\chi^2 = 0`.

        To perform this task the expression of :math:`\\nabla \\chi^2` is
        determined, ignoring that :math:`\\chi^2` involves summing over all
        terms. Then the sum is performed by substituting the variables by their
        respective data and summing all terms, while leaving the parameters
        symbolic.

        The resulting system of equations is then easily solved with
        ``sympy.solve``.

        :return: ``FitResult``
        """
        # Obtain the best fit params first.
        best_fit_params = self.best_fit_params()
        cov_matrix = self.covariance_matrix(best_fit_params=best_fit_params)

        self._fit_results = FitResults(
            model=self.model,
            popt=[best_fit_params[param] for param in self.model.params],
            covariance_matrix=cov_matrix,
            infodic={'nfev': 0},
            mesg='',
            ier=0,
        )
        self._fit_results.gof_qualifiers['r_squared'] = \
            r_squared(self.model, self._fit_results, self.data)
        return self._fit_results


class NonLinearLeastSquares(BaseFit):
    """
    Experimental.
    Implements non-linear least squares [wiki_nllsq]_. Works by a two step process:
    First the model is linearised by doing a first order taylor expansion
    around the guesses for the parameters.
    Then a LinearLeastSquares fit is performed. This is iterated until
    a fit of sufficient quality is obtained.

    Sensitive to good initial guesses. Providing good initial guesses is a must.

    .. [wiki_nllsq] https://en.wikipedia.org/wiki/Non-linear_least_squares
    """
    def __init__(self, *args, **kwargs):
        super(NonLinearLeastSquares, self).__init__(*args, **kwargs)
        # Make an approximation of model at the initial guesses
        # self.model_appr = self.linearize(self.model, {p: p.value for p in self.model.params})
        self.model_appr = TaylorModel(self.model)
        # Set initial expansion point
        self.model_appr.p0 = {
            param: value for param, value in zip(self.model_appr.params, self.initial_guesses)
        }

    def execute(self, relative_error=1e-8, max_iter=500):
        """
        Perform a non-linear least squares fit.

        :param relative_error: Relative error between the sum of squares
            of subsequent itterations. Once smaller than the value specified,
            the fit is considered complete.
        :param max_iter: Maximum number of iterations before giving up.
        :return: Instance of ``FitResults``.
        """
        fit = LinearLeastSquares(self.model_appr,
                                 absolute_sigma=self.absolute_sigma,
                                 **key2str(self.data))

        kwargs = self.data.copy()
        kwargs.update(dict(zip(self.model.params, self.initial_guesses)))
        i = 0
        S_k1 = np.sum(
            fit.leastsquares_model(**key2str(kwargs))
        )
        while i < max_iter:
            fit_params = fit.best_fit_params()
            kwargs.update(fit_params)
            S_k2 = np.sum(
                fit.leastsquares_model(**key2str(kwargs))
            )
            if not S_k1 < 0 and np.abs(S_k2 - S_k1) <= relative_error * np.abs(S_k1):
                break
            else:
                S_k1 = S_k2
                # Update the model with a better approximation
                self.model_appr.p0 = fit_params
                fit.leastsquares_model = leastsquares_from_model(self.model_appr)
                i += 1

        cov_matrix = fit.covariance_matrix(best_fit_params=fit_params)

        self._fit_results = FitResults(
            model=self.model,
            popt=[float(fit_params[param]) for param in self.model.params],
            covariance_matrix=cov_matrix,
            infodic={'nfev': i},
            mesg='',
            ier=0,
        )
        self._fit_results.gof_qualifiers['r_squared'] = \
            r_squared(self.model, self._fit_results, self.data)
        return self._fit_results

class Fit(HasCovarianceMatrix):
    """
    Your one stop fitting solution! Based on the nature of the input, this
    object will attempt to select the right fitting type for your problem.

    If you need very specific control over how the problem is solved, you can
    pass it the minimizer or objective function you would like to use.

    Example usage::

        a, b = parameters('a, b')
        x, y = variables('x, y')

        model = {y: a * x + b}

        # Fit will use its default settings
        fit = Fit(model, x=xdata, y=ydata)
        fit_result = fit.execute()

        # Use Nelder-Mead instead
        fit = Fit(model, x=xdata, y=ydata, minimizer=NelderMead)
        fit_result = fit.execute()

        # Use Nelder-Mead to get close, and BFGS to polish it off
        fit = Fit(model, x=xdata, y=ydata, minimizer=[NelderMead, BFGS])
        fit_result = fit.execute(minimizer_kwargs=[dict(xatol=0.1), {}])
    """

    @keywordonly(objective=None, minimizer=None, constraints=None)
    def __init__(self, model, *ordered_data, **named_data):
        """

        :param model: (dict of) sympy expression(s) or ``Model`` object.
        :param constraints: iterable of ``Relation`` objects to be used as
            constraints.
        :param bool absolute_sigma: True by default. If the sigma is only used
            for relative weights in your problem, you could consider setting it
            to False, but if your sigma are measurement errors, keep it at True.
            Note that curve_fit has this set to False by default, which is
            wrong in experimental science.
        :param objective: Have Fit use your specified objective. Can be one of
            the predefined `symfit` objectives or any callable which accepts fit
            parameters and returns a scalar.
        :param minimizer: Have Fit use your specified
            :class:`symfit.core.minimizers.BaseMinimizer`. Can be a
            :class:`~collections.abc.Sequence` of :class:`symfit.core.minimizers.BaseMinimizer`.
        :param ordered_data: data for dependent, independent and sigma
            variables. Assigned in the following order: independent vars are
            assigned first, then dependent vars, then sigma's in dependent
            vars. Within each group they are assigned in alphabetical order.
        :param named_data: assign dependent, independent and sigma variables
            data by name.
        """
        objective = named_data.pop('objective')
        minimizer = named_data.pop('minimizer')
        constraints = named_data.pop('constraints')
        super(Fit, self).__init__(model, *ordered_data, **named_data)

        # List of Constraint objects
        self.constraints = self._init_constraints(constraints=constraints)

        if objective is None:
            # Param only scalar Model -> the model is the objective.
            if len(self.model.independent_vars) == 0 and len(self.model) == 1:
                # No data provided means a simple minimization of the Model parameters
                # is requested, not a fit.
                if all(value is None for value in self.data.values()):
                    objective = MinimizeModel
            elif minimizer is MINPACK:
                # MINPACK is considered a special snowflake, as its API has to be
                # considered seperately and has its own non standard objective function.
                objective = VectorLeastSquares

        if objective is None:
            objective = LeastSquares
        elif objective == LogLikelihood or isinstance(objective, LogLikelihood):
            if self.sigmas_provided:
                raise NotImplementedError(
                    'LogLikelihood fitting does not currently support data '
                    'weights.'
                )
        # Initialise the objective if it's not initialised already
        if isinstance(objective, BaseObjective):
            self.objective = objective
        else:
            self.objective = objective(self.model, self.data)

        # Select the minimizer on the basis of the provided information.
        if minimizer is None:
            minimizer = self._determine_minimizer()

        # Initialise the minimizer
        if isinstance(minimizer, Sequence):
            minimizers = [self._init_minimizer(mini) for mini in minimizer]
            self.minimizer = self._init_minimizer(ChainedMinimizer, minimizers=minimizers)
        else:
            self.minimizer = self._init_minimizer(minimizer)

    def _determine_minimizer(self):
        """
        Determine the most suitable minimizer by the presence of bounds or
        constraints.
        :return: a subclass of `BaseMinimizer`.
        """
        if self.constraints:
            return SLSQP
        elif any([bound is not None for pair in self.model.bounds for bound in pair]):
            # If any bound is set
            return LBFGSB
        else:
            return BFGS

    def _init_minimizer(self, minimizer, **minimizer_options):
        """
        Takes a :class:`~symfit.core.minimizers.BaseMinimizer` and instantiates
        it, passing the jacobian and constraints as appropriate for the
        minimizer.

        :param minimizer: :class:`~symfit.core.minimizers.BaseMinimizer` to
            instantiate.
        :param **minimizer_options: Further options to be passed to the
            minimizer on instantiation.
        :returns: instance of :class:`~symfit.core.minimizers.BaseMinimizer`.
        """

        if isinstance(minimizer, BaseMinimizer):
            return minimizer
        if issubclass(minimizer, BasinHopping):
            minimizer_options['local_minimizer'] = self._init_minimizer(
                self._determine_minimizer()
            )
        if issubclass(minimizer, GradientMinimizer):
            # If an analytical version of the Jacobian exists we should use
            # that, otherwise we let the minimizer estimate it itself.
            # Hence the check of numerical_jacobian, as this is the
            # py function version of the analytical jacobian.
            if hasattr(self.model, 'eval_jacobian') and hasattr(self.objective, 'eval_jacobian'):
                minimizer_options['jacobian'] = self.objective.eval_jacobian

        if issubclass(minimizer, ConstrainedMinimizer):
            # set the constraints as MinimizeModel. The dependent vars of the
            # constraint are set to None since their value is irrelevant.
            constraint_objectives = []
            for constraint in self.constraints:
                data = self.data  # No copy, share state
                data.update({var: None for var in constraint.dependent_vars})
                constraint_objectives.append(MinimizeModel(constraint, data))
            minimizer_options['constraints'] = constraint_objectives
        return minimizer(self.objective, self.model.params, **minimizer_options)

    def _init_constraints(self, constraints):
        """
        Takes the user provided constraints and converts them to a list of
        :class:`~symfit.core.fit.Constraint` objects.

        :param constraints: iterable of :class:`~sympy.core.relational.Relation`
            objects.
        :return: list of :class:`~symfit.core.fit.Constraint` objects.
        """
        con_models = []
        if constraints:
            for constraint in constraints:
                if isinstance(constraint, Constraint):
                    con_models.append(constraint)
                else:
                    con_models.append(Constraint(constraint, self.model))
            # Check if the type of each constraint is allowed.
            allowed_types = [sympy.Eq, sympy.Ge, sympy.Le]
            for index in range(len(con_models)):
                constraint = con_models[index]
                if constraint.constraint_type not in allowed_types:
                    raise ModelError(
                        'Only constraints of the type {} are allowed. A constraint'
                        ' of type {} was provided.'.format(allowed_types,
                                                           constraint.constraint_type)
                    )
                elif constraint.constraint_type is sympy.Le:
                    assert len(constraint) == 1
                    for var in constraint:
                        component = constraint[var]
                        con_models[index] = Constraint(
                            sympy.Ge(- component, 0),
                            model=constraint.model
                        )
        return con_models

    def execute(self, **minimize_options):
        """
        Execute the fit.

        :param minimize_options: keyword arguments to be passed to the specified
            minimizer.
        :return: FitResults instance
        """
        minimizer_ans = self.minimizer.execute(**minimize_options)
        try: # to build covariance matrix
            cov_matrix = minimizer_ans.covariance_matrix
        except AttributeError:
            cov_matrix = self.covariance_matrix(dict(zip(self.model.params, minimizer_ans._popt)))
        else:
            if cov_matrix is None:
                cov_matrix = self.covariance_matrix(dict(zip(self.model.params, minimizer_ans._popt)))
        finally:
            minimizer_ans.covariance_matrix = cov_matrix
        # Overwrite the DummyModel with the current model
        minimizer_ans.model = self.model
        minimizer_ans.gof_qualifiers['r_squared'] = r_squared(self.model, minimizer_ans, self.data)
        return minimizer_ans


def r_squared(model, fit_result, data):
    """
    Calculates the coefficient of determination, R^2, for the fit.

    (Is not defined properly for vector valued functions.)

    :param model: Model instance
    :param fit_result: FitResults instance
    :param data: data with which the fit was performed.
    """
    # First filter out the dependent vars
    y_is = [data[var] for var in model if var in data]
    x_is = [value for var, value in data.items() if var.name in model.__signature__.parameters]
    y_bars = [np.mean(y_i) if y_i is not None else None for y_i in y_is]
    f_is = model(*x_is, **fit_result.params)
    SS_res = np.sum([np.sum((y_i - f_i)**2) for y_i, f_i in zip(y_is, f_is) if y_i is not None])
    SS_tot = np.sum([np.sum((y_i - y_bar)**2) for y_i, y_bar in zip(y_is, y_bars) if y_i is not None])
    return 1 - SS_res/SS_tot

class ODEModel(BaseGradientModel):
    """
    Model build from a system of ODEs. When the model is called, the ODE is
    integrated using the LSODA package.
    """
    def __init__(self, model_dict, initial, *lsoda_args, **lsoda_kwargs):
        """
        :param model_dict: Dictionary specifying ODEs. e.g.
            model_dict = {D(y, x): a * x**2}
        :param initial: ``dict`` of initial conditions for the ODE.
            Must be provided! e.g.
            initial = {y: 1.0, x: 0.0}
        :param lsoda_args: args to pass to the lsoda solver.
            See `scipy's odeint <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html>`_
            for more info.
        :param lsoda_kwargs: kwargs to pass to the lsoda solver.
        """
        self.initial = initial
        self.lsoda_args = lsoda_args
        self.lsoda_kwargs = lsoda_kwargs

        sort_func = lambda symbol: str(symbol)
        # Mapping from dependent vars to their derivatives
        self.dependent_derivatives = {d: list(d.free_symbols - set(d.variables))[0] for d in model_dict}
        self.dependent_vars = sorted(
            self.dependent_derivatives.values(),
            key=sort_func
        )
        self.independent_vars = sorted(set(d.variables[0] for d in model_dict), key=sort_func)
        self.interdependent_vars = []  # Not yet supported for ODEModels
        if not len(self.independent_vars):
            raise ModelError('ODEModel can only have one independent variable.')

        self.model_dict = OrderedDict(
            sorted(
                model_dict.items(),
                key=lambda i: sort_func(self.dependent_derivatives[i[0]])
            )
        )
        # Extract all the params and vars as a sorted, unique list.
        expressions = model_dict.values()
        model_params = set([])

        # Only the once's that have a Parameter as initial parameter.
        # self.initial_params = {value for var, value in self.initial.items()
        #                        if isinstance(value, Parameter)}

        for expression in expressions:
            vars, params = seperate_symbols(expression)
            model_params.update(params)
            # self.independent_vars.update(vars)
        # Although unique now, params and vars should be sorted alphabetically to prevent ambiguity
        self.params = sorted(model_params, key=sort_func)
        # self.params = sorted(self.model_params | self.initial_params, key=sort_func)
        # self.model_params = sorted(self.model_params, key=sort_func)
        # self.initial_params = sorted(self.initial_params, key=sort_func)

        # Make Variable object corresponding to each sigma var.
        self.sigmas = {var: Variable(name='sigma_{}'.format(var.name)) for var in self.dependent_vars}

        self.__signature__ = self._make_signature()

    def __str__(self):
        """
        Printable representation of this model.

        :return: str
        """
        template = "{}; {}) = {}"
        parts = []
        for var, expr in self.model_dict.items():
            parts.append(template.format(
                    str(var)[:-1],
                    ", ".join(arg.name for arg in self.params),
                    expr
                )
            )
        return "\n".join(parts)

    def __getitem__(self, dependent_var):
        """
        Gives the function defined for the derivative of ``dependent_var``.
        e.g. :math:`y' = f(y, t)`, model[y] -> f(y, t)

        :param dependent_var:
        :return:
        """
        for d, f in self.model_dict.items():
            if dependent_var == self.dependent_derivatives[d]:
                return f

    def __iter__(self):
        """
        :return: iterable over self.model_dict
        """
        return iter(self.dependent_vars)

    def __neg__(self):
        """
        :return: new model with opposite sign. Does not change the model in-place,
            but returns a new copy.
        """
        new_model_dict = self.model_dict.copy()
        for key in new_model_dict:
            new_model_dict[key] *= -1
        return self.__class__(new_model_dict, initial=self.initial)

    @cached_property
    def _ncomponents(self):
        """
        :return: The `numerical_components` for an ODEModel. This differs from
            the traditional `numerical_components`, in that these component can
            also contain dependent variables, not just the independent ones.

            Each of these components does not correspond to e.g. `y(t) = ...`,
            but to `D(y, t) = ...`. The system spanned by these component
            therefore still needs to be integrated.
        """
        return [sympy_to_py(expr, self.independent_vars + self.dependent_vars, self.params)
                for expr in self.values()]

    @cached_property
    def _njacobian(self):
        """
        :return: The `numerical_jacobian` of the components of the ODEModel with
            regards to the dependent variables. This is not to be confused with
            the jacobian of the model as a whole, which is 2D and computed with
            regards to the dependent vars and the fit parameters, and the
            ODEModel still needs to integrated to compute that.

            Instead, this function is used by the ODE integrator, and is not
            meant for human consumption.
        """
        return [
            [sympy_to_py(
                    sympy.diff(expr, var), self.independent_vars + self.dependent_vars, self.params
                ) for var in self.dependent_vars
            ] for _, expr in self.items()
        ]

    def eval_components(self, *args, **kwargs):
        """
        Numerically integrate the system of ODEs.

        :param args: Ordered arguments for the parameters and independent
          variables
        :param kwargs:  Keyword arguments for the parameters and independent
          variables
        :return:
        """
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        t_like = bound_arguments.arguments[self.independent_vars[0].name]

        # System of functions to be integrated
        f = lambda ys, t, *a: [c(t, *(list(ys) + list(a))) for c in self._ncomponents]
        Dfun = lambda ys, t, *a: [[c(t, *(list(ys) + list(a))) for c in row] for row in self._njacobian]

        initial_dependent = [self.initial[var] for var in self.dependent_vars]
        t_initial = self.initial[self.independent_vars[0]] # Assuming there's only one

        # Check if the time-like data includes the initial value, because integration should start there.
        try:
            t_like[0]
        except (TypeError, IndexError): # Python scalar gives TypeError, numpy scalars IndexError
            t_like = np.array([t_like]) # Allow evaluation at one point.

        # The strategy is to split the time axis in a part above and below the
        # initial value, and to integrate those seperately. At the end we rejoin them.
        # np.flip is needed because odeint wants the first point to be t_initial
        # and so t_smaller is a declining series.
        if t_initial in t_like:
            t_bigger = t_like[t_like >= t_initial]
            t_smaller = t_like[t_like <= t_initial][::-1]
        else:
            t_bigger = np.concatenate(
                (np.array([t_initial]), t_like[t_like > t_initial])
            )
            t_smaller = np.concatenate(
                (np.array([t_initial]), t_like[t_like < t_initial][::-1])
            )
        # Properly ordered time axis containing t_initial
        t_total = np.concatenate((t_smaller[::-1][:-1], t_bigger))

        ans_bigger = odeint(
            f,
            initial_dependent,
            t_bigger,
            args=tuple(
                bound_arguments.arguments[param.name] for param in self.params),
            Dfun=Dfun,
            *self.lsoda_args, **self.lsoda_kwargs
        )
        ans_smaller = odeint(
            f,
            initial_dependent,
            t_smaller,
            args=tuple(
                bound_arguments.arguments[param.name] for param in self.params),
            Dfun=Dfun,
            *self.lsoda_args, **self.lsoda_kwargs
        )

        ans = np.concatenate((ans_smaller[1:][::-1], ans_bigger))
        if t_initial in t_like:
            # The user also requested to know the value at t_initial, so keep it.
            return ans.T
        else:
            # The user didn't ask for the value at t_initial, so exclude it.
            # (t_total contains all the t-points used for the integration,
            # and so is t_like with t_initial inserted at the right position).
            return ans[t_total != t_initial].T

    def __call__(self, *args, **kwargs):
        """
        Evaluate the model for a certain value of the independent vars and parameters.
        Signature for this function contains independent vars and parameters, NOT dependent and sigma vars.

        Can be called with both ordered and named parameters. Order is independent vars first, then parameters.
        Alphabetical order within each group.

        :param args: Ordered arguments for the parameters and independent
          variables
        :param kwargs:  Keyword arguments for the parameters and independent
          variables
        :return: A namedtuple of all the dependent vars evaluated at the desired point. Will always return a tuple,
            even for scalar valued functions. This is done for consistency.
        """
        Ans = variabletuple('Ans', self)
        ans = Ans(*self.eval_components(*args, **kwargs))
        return ans


def jacobian_from_model(model, as_functions=False):
    """
    Build a :class:`~symfit.core.fit.CallableModel` representing the Jacobian of
    ``model``.

    This function make sure the chain rule is correctly applied for
    interdependent variables.

    :param model: Any symbolical model-type.
    :param as_functions: If `True`, the result is returned using
        :class:`sympy.Function` where needed, e.g. {y(x, a): a * x} instead of
        {y: a * x}.
    :return: :class:`~symfit.core.fit.CallableModel` representing the Jacobian
        of ``model``.
    """
    def partial_diff(var, *params):
        """
        Sympy does not handle repeated partial derivation correctly, e.g.
        D(D(y, a), a) = D(y, a, a) but D(D(y, a), b) = 0.
        Use this function instead to prevent evaluation to zero.
        """
        if isinstance(var, sympy.Derivative):
            return sympy.Derivative(var.expr, *(var.variables + params))
        else:
            return D(var, *params)

    def partial_subs(func, func2vars):
        """
        Partial-bug proof substitution. Works by making the substitutions on
        the expression inside the derivative first, and then rebuilding the
        derivative safely without evaluating it using `partial_diff`.
        """
        if isinstance(func, sympy.Derivative):
            new_func = func.expr.xreplace(func2vars)
            new_variables = tuple(var.xreplace(func2vars)
                                  for var in func.variables)
            return partial_diff(new_func, *new_variables)
        else:
            return func.xreplace(func2vars)

    # Inverse dict so we can turn functions back into vars in the end
    functions_as_vars = dict((v, k) for k, v in model.vars_as_functions.items())
    # Create the jacobian components. The `vars` here in the model_dict are
    # always of the type D(y, a), but the righthand-side might still contain
    # functions instead of vars depending on the value of `as_functions`.
    jac = {}
    for func, expr in model.function_dict.items():
        for param in model.params:
            target = D(func, param)
            dfdp = expr.diff(param)
            if as_functions:
                jac[partial_subs(target, functions_as_vars)] = dfdp
            else:
                # Turn Function objects back into Variables.
                dfdp = dfdp.subs(functions_as_vars, evaluate=False)
                jac[partial_subs(target, functions_as_vars)] = dfdp
    # Next lines are needed for the Hessian, where the components of model still
    # contain functions instead of vars.
    if as_functions:
        jac.update(model)
    else:
        jac.update({y: expr.subs(functions_as_vars, evaluate=False)
                    for y, expr in model.items()})
    return CallableModel(jac)

def hessian_from_model(model):
    """
    Build a :class:`~symfit.core.fit.CallableModel` representing the Hessian of
    ``model``.

    This function make sure the chain rule is correctly applied for
    interdependent variables.

    :param model: Any symbolical model-type.
    :return: :class:`~symfit.core.fit.CallableModel` representing the Hessian
        of ``model``.
    """
    jac_model = jacobian_from_model(model, as_functions=True)
    print(jac_model)
    return jacobian_from_model(jac_model)

def leastsquares_from_model(model):
    """
    Creates an analytical Least-Squares model for ``model``, which can be used
    to solve analytical least-squares problems.

    .. note: Currently only for use by :class:`LinearLeastSquares`, as no
        analytical sum is included.

    :param model: Any symbolical model-type.
    :return: :class:`~symfit.core.fit.GradientModel` representing the :math:`\chi^2`
        of ``model``.
    """
    chi2_expr = sum(((model[y] - y)**2 / model.sigmas[y]**2)
                    for y in model.dependent_vars)
    chi2 = Variable()
    chi2_dict = {chi2: chi2_expr}
    # Update it with any leftover interdependent variables we might need.
    chi2_dict.update({var: expr for var, expr in model.items()
                     if var not in model.dependent_vars})
    return GradientModel(chi2_dict)