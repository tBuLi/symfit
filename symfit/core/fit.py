from collections import namedtuple, Mapping, OrderedDict
import copy
import sys
import warnings
from abc import abstractmethod

import sympy
from sympy.core.relational import Relational
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint

from symfit.core.argument import Parameter, Variable
from symfit.core.support import seperate_symbols, keywordonly, sympy_to_py, cache, key2str, deprecated
from symfit.core.leastsqbound import leastsqbound

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig


class ModelError(Exception):
    """
    Raised when a problem occurs with a model.
    """
    pass


class ParameterDict(object):
    """
    Container for all the parameters and their (co)variances.
    Behaves mostly like an OrderedDict: can be **-ed, allowing the sexy syntax where a model is
    called with values for the Variables and **params. However, under iteration
    it behaves like a list! In other words, it preserves order in the params.
    """
    def __init__(self, params, popt, pcov, *args, **kwargs):
        super(ParameterDict, self).__init__(*args, **kwargs)
        self.__params = params  # list of Parameter instances
        self.__params_dict = dict([(p.name, p) for p in params])
        # popt and pstdev are dicts with parameter names: value pairs.
        self.__popt = dict([(p.name, value) for p, value in zip(params, popt)])
        if pcov is not None:
            # Can be None.
            stdevs = np.sqrt(np.diagonal(pcov))
        else:
            stdevs = [None for _ in params]
        self.__pstdev = dict([(p.name, s) for p, s in zip(params, stdevs)])
        # Covariance matrix
        self.__pcov = pcov

    def __len__(self):
        """
        Length gives the number of ``Parameter`` instances.

        :return: len(self.__params)
        """
        return len(self.__params)

    def __iter__(self):
        """
        Iteration over the ``Parameter`` instances.
        :return: iterator
        """
        return iter(self.__params)

    def __getitem__( self, param_name):
        """
        This method allows this object to be addressed as a dict. This allows for the ** unpacking.
        Therefore return the value of the best fit parameter, as this is what the user expects.

        :param param_name: Name of the ``Parameter`` whose value your interested in.
        :return: the value of the best fit parameter with name 'key'.
        """
        return getattr(self, param_name)

    def keys(self):
        """
        :return: All ``Parameter`` names.
        """
        return self.__params_dict.keys()

    def __getattr__(self, name):
        """
        A user can access the value of a parameter directly through this object.

        :param name: Name of a ``Parameter``.
            Naming convention:
            let a = Parameter(). Then:
            .a gives the value of the parameter.
            .a_stdev gives the standard deviation.
        """
        # If a parameter with this name exists, return it immediately
        try:
            return self.__popt[name]
        except KeyError:
            param_name = name
            # Expand this if statement if in the future we allow more suffixes
            if name.endswith('_stdev'):
                param_name = name[:-len('_stdev')]  # everything but the suffix
                try:
                    return self.__pstdev[param_name]
                except KeyError:
                    pass
        raise AttributeError('No Parameter by the name {}.'.format(param_name))

    @deprecated(replacement='value')
    def get_value(self, param):
        """
        Deprecated.
        :param param: ``Parameter`` instance.
        :return: returns the numerical value of param
        :raises: DeprecationWarning
        """
        return self.value(param)

    @deprecated(replacement='stdev')
    def get_stdev(self, param):
        """
        Deprecated.
        :param param: ``Parameter`` instance.
        :return: returns the standard deviation of param
        :raises: DeprecationWarning
        """
        return self.stdev(param)

    def value(self, param):
        """
        :param param: ``Parameter`` instance.
        :return: returns the numerical value of param
        """
        return self.__popt[param.name]

    def stdev(self, param):
        """
        :param param: ``Parameter`` instance.
        :return: returns the standard deviation of param
        """
        return self.__pstdev[param.name]

    @property
    def covariance_matrix(self):
        """
        Returns the covariance matrix.
        """
        return self.__pcov


class FitResults(object):
    """
    Class to display the results of a fit in a nice and unambiguous way.
    All things related to the fit are available on this class, e.g.
    - parameters + stdev
    - R squared (Regression coefficient.)
    - fitting status message

    This object is made to behave entirely read-only. This is a bit unnatural
    to enforce in Python but I feel it is necessary to guarantee the integrity
    of the results.
    """
    __params = None  # Private property.
    __infodict = None
    __status_message = None
    __iterations = None
    __ydata = None
    __sigma = None

    def __init__(self, params, popt, pcov, infodic, mesg, ier, ydata=None, sigma=None):
        """
        Excuse the ugly names of most of these variables, they are inherited from scipy. Will be changed.

        :param params: list of ``Parameter``'s.
        :param popt: best fit parameters, same ordering as in params.
        :param pcov: covariance matrix.
        :param infodic: dict with fitting info.
        :param mesg: Status message.
        :param ier: Number of iterations.
        :param ydata:
        """
        # Validate the types in rough way
        self.__infodict = infodic
        self.__status_message = mesg
        self.__iterations = ier
        self.__ydata = ydata
        self.__params = ParameterDict(params, popt, pcov)
        self.__sigma = sigma

    def __str__(self):
        """
        Pretty print the results as a table.
        """
        res = '\nParameter Value        Standard Deviation\n'
        for p in self.params:
            value = self.params.value(p)
            value_str = '{:e}'.format(value) if value is not None else 'None'
            stdev = self.params.stdev(p)
            stdev_str = '{:e}'.format(stdev) if stdev is not None else 'None'
            res += '{:10}{} {}\n'.format(p.name, value_str, stdev_str, width=20)

        res += 'Fitting status message: {}\n'.format(self.status_message)
        res += 'Number of iterations:   {}\n'.format(self.infodict['nfev'])
        res += 'Regression Coefficient: {}\n'.format(self.r_squared)
        return res

    @property
    def r_squared(self):
        """
        r_squared Property.

        :return: Regression coefficient.
        """
        if self._r_squared is not None:
            return self._r_squared
        else:
            return float('nan')

    @r_squared.setter
    def r_squared(self, value):
        self._r_squared = value

    #
    # READ-ONLY Properties
    # What follows are all the read-only properties of this object.
    # Their definitions are mostly trivial, but necessary to make sure that
    # FitResults can't be changed.
    #

    @property
    def infodict(self):
        """
        Read-only Property.
        """
        return self.__infodict

    @property
    def status_message(self):
        """
        Read-only Property.
        """
        return self.__status_message

    @property
    def iterations(self):
        """
        Read-only Property.
        """
        return self.__iterations

    @property
    def params(self):
        """
        Read-only Property.
        """
        return self.__params

    def stdev(self, param):
        """
        Return the standard deviation in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Standard deviation of ``param``.
        """
        return self.params.stdev(param)

    def value(self, param):
        """
        Return the value in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Value of ``param``.
        """
        return self.params.value(param)

    def variance(self, param):
        """
        Return the variance in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Variance of ``param``.
        """
        param_number = list(self.params).index(param)
        return self.params.covariance_matrix[param_number, param_number]

    def covariance(self, param_1, param_2):
        """
        Return the covariance between param_1 and param_2.

        :param param_1: ``Parameter`` Instance.
        :param param_2: ``Parameter`` Instance.
        :return: Covariance of the two params.
        """
        param_1_number = list(self.params).index(param_1)
        param_2_number = list(self.params).index(param_2)
        return self.params.covariance_matrix[param_1_number, param_2_number]

class BaseModel(Mapping):
    """
    ABC for ``Model``'s. Makes sure models are iterable.
    Models can be initiated from Mappings or Iterables of Expressions, or from an expression directly.
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
                enum = enumerate(model)
            except TypeError:
                enum = enumerate([model])

            model = {sympy.Dummy('y_{}'.format(index + 1)): expr for index, expr in enum}
            # model = {Variable('dummy_{}'.format(index + 1)): expr for index, expr in enumerate(model)}

        self._init_from_dict(model)

    def __len__(self):
        """
        :return: the number of dependent variables for this model.
        """
        return len(self.model_dict)

    def __getitem__(self, dependent_var):
        """
        Returns the expression belonging to a given dependent variable.

        :param dependent_var: Instance of ``Variable``
        :type dependent_var: ``Variable``
        :return: The expression belonging to ``dependent_var``
        """
        return self.model_dict[dependent_var]

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
        expressions. For more info check the `sympy docs<https://github.com/sympy/sympy/wiki/Faq>`_.

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
                    if not self[var_1].expand() - other[var_2].expand() == 0:
                        return False
            else:
                return True

    def _init_from_dict(self, model_dict):
        """
        Initiate self from a model_dict to make sure attributes such as vars, params are available.

        Creates lists of alphabetically sorted independent vars, dependent vars, sigma vars, and parameters.
        Finally it creates a signature for this model so it can be called nicely. This signature only contains
        independent vars and params, as one would expect.

        :param model_dict: dict of (dependent_var, expression) pairs.
        """
        sort_func = lambda symbol: str(symbol)
        self.model_dict = OrderedDict(sorted(model_dict.items(), key=lambda i: sort_func(i[0])))
        self.dependent_vars = sorted(model_dict.keys(), key=sort_func)

        # Extract all the params and vars as a sorted, unique list.
        expressions = model_dict.values()
        _params, self.independent_vars = set([]), set([])
        for expression in expressions:
            vars, params = seperate_symbols(expression)
            _params.update(params)
            self.independent_vars.update(vars)
        # Although unique now, params and vars should be sorted alphabetically to prevent ambiguity
        self.params = sorted(_params, key=sort_func)
        self.independent_vars = sorted(self.independent_vars, key=sort_func)

        # Make Variable object corresponding to each var.
        self.sigmas = {var: Variable(name='sigma_{}'.format(var.name)) for var in self.dependent_vars}

    @property
    @cache
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
        return [(np.nextafter(p.value, 0), p.value) if p.fixed else (p.min, p.max) for p in self.params]

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


class CallableModel(BaseModel):
    """
    Defines a callable model. The usual rules apply to the ordering of the arguments:

    * first independent variables, then dependent variables, then parameters.
    * within each of these groups they are ordered alphabetically.
    """
    @abstractmethod
    def eval_components(self, *args, **kwargs):
        """
        Evaluate the components of the model with the given data.
        Used for numerical evaluation.
        """
        pass

    def _make_signature(self):
        # Handle args and kwargs according to the allowed names.
        parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
            inspect_sig.Parameter(arg.name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD)
                for arg in self.independent_vars + self.params
        ]
        return inspect_sig.Signature(parameters=parameters)

    def _init_from_dict(self, model_dict):
        super(CallableModel, self)._init_from_dict(model_dict)
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
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        Ans = namedtuple('Ans', [var.name for var in self])
        # return Ans(*[component(**bound_arguments.arguments) for component in self.numerical_components])
        return Ans(*self.eval_components(**bound_arguments.arguments))


class Model(CallableModel):
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
    def __str__(self):
        """
        Printable representation of this model.

        :return: str
        """
        template = "{}({}; {}) = {}"
        parts = []
        for var, expr in self.items():
            parts.append(template.format(
                    var,
                    ", ".join(arg.name for arg in self.independent_vars),
                    ", ".join(arg.name for arg in self.params),
                    expr
                )
            )
        return "\n".join(parts)

    @property
    # @cache
    def jacobian(self):
        """
        :return: Jacobian 'Matrix' filled with the symbolic expressions for all the partial derivatives.
        Partial derivatives are of the components of the function with respect to the Parameter's,
        not the independent Variable's.
        """
        return [[sympy.diff(expr, param) for param in self.params] for expr in self.values()]

    @property
    # @cache
    def chi_squared(self):
        """
        :return: Symbolic :math:`\\chi^2`
        """
        return sum(((f - y)/self.sigmas[y])**2 for y, f in self.items())

    @property
    # @cache
    def chi(self):
        """
        :return: Symbolic Square root of :math:`\\chi^2`. Required for MINPACK optimization only. Denoted as :math:`\\sqrt(\\chi^2)`
        """
        return sympy.sqrt(self.chi_squared)

    @property
    # @cache
    def chi_jacobian(self):
        """
        Return a symbolic jacobian of the :math:`\\sqrt(\\chi^2)` function.
        Vector of derivatives w.r.t. each parameter. Not a Matrix but a vector! This is because that's what leastsq needs.
        """
        jac = []
        for param in self.params:
            # Differentiate to every param
            f = sympy.diff(self.chi, param)
            jac.append(f)
        return jac

    @property
    # @cache
    def chi_squared_jacobian(self):
        """
        Return a symbolic jacobian of the :math:`\\chi^2` function.
        Vector of derivatives w.r.t. each parameter. Not a Matrix but a vector!
        """
        jac = []
        for param in self.params:
            # Differentiate to every param
            f = sympy.diff(self.chi_squared, param)
            jac.append(f)
        return jac

    # @property
    # # @cache
    # def ss_res(self):
    #     """
    #     :return: Residual sum of squares. Similar to chi_squared, but without considering weights.
    #     """
    #     return sum((y - f)**2 for y, f in self.items())

    @property
    # @cache
    def numerical_components(self):
        """
        :return: lambda functions of each of the components in model_dict, to be used in numerical calculation.
        """
        return [sympy_to_py(expr, self.independent_vars, self.params) for expr in self.values()]

    @property
    # @cache
    def numerical_jacobian(self):
        """
        :return: lambda functions of the jacobian matrix of the function, which can be used in numerical optimization.
        """
        return [[sympy_to_py(partial, self.independent_vars, self.params) for partial in row] for row in self.jacobian]

    @property
    # @cache
    def numerical_chi_squared(self):
        """
        :return: lambda function of the ``.chi_squared`` method, to be used in numerical optimisation.
        """
        return sympy_to_py(self.chi_squared, self.vars, self.params)

    @property
    # @cache
    def numerical_chi(self):
        """
        :return: lambda function of the ``.chi`` method, to be used in MINPACK optimisation.
        """
        return sympy_to_py(self.chi, self.vars, self.params)

    @property
    # @cache
    def numerical_chi_jacobian(self):
        """
        :return: lambda functions of the jacobian of the ``.chi`` method, which can be used in numerical optimization.
        """
        return [sympy_to_py(component, self.vars, self.params) for component in self.chi_jacobian]

    @property
    # @cache
    def numerical_chi_squared_jacobian(self):
        """
        :return: lambda functions of the jacobian of the ``.chi_squared`` method.
        """
        return [sympy_to_py(component, self.vars, self.params) for component in self.chi_squared_jacobian]

    def eval_components(self, *args, **kwargs):
        """
        :return: lambda functions of each of the components in model_dict, to be used in numerical calculation.
        """
        return [expr(*args, **kwargs) for expr in self.numerical_components]
        # return [sympy_to_py(expr, self.independent_vars, self.params)(*args, **kwargs) for expr in self.values()]

    def eval_jacobian(self, *args, **kwargs):
        """
        :return: lambda functions of the jacobian matrix of the function, which can be used in numerical optimization.
        """
        return [
            [partial(*args, **kwargs) for partial in row ] for row in self.numerical_jacobian
        ]
        # return [[partial(*args, **kwargs) for partial in row] for row in self.jacobian]


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
            for (p, p0), jac in zip(params_0.items(), jacobian_vec): # params_0 is assumed OrderedDict!
                linear += jac.subs(params_0.items()) * (p - p0)
            model_dict[var] = linear
        self.params_0 = params_0
        super(TaylorModel, self).__init__(model_dict)
        # super(TaylorModel, self).__init__(**key2str(model_dict))
        self.model_dict_orig = copy.copy(self.model_dict)

    @property
    def params(self):
        """
        params returns only the `free` parameters. Strictly speaking, the expression for a
        ``TaylorModel`` contains both the parameters :math:`\vec{p}` and :math:`\vec{p_0}`
        around which to expand, but params should only give :math:`\vec{p}`. To get a
        mapping to the :math:`\vec{p_0}`, use ``.params_0``.
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
    constraint_type = sympy.Eq

    def __init__(self, constraint, model):
        """
        :param constraint: constraint that model should be subjected to.
        :param model: A constraint is always tied to a model.
        """
        # raise Exception(model)
        if isinstance(constraint, Relational):
            self.constraint_type = type(constraint)
            if isinstance(model, Model):
                self.model = model
            else:
                raise TypeError('The model argument must be of type Model.')
            super(Constraint, self).__init__(constraint.lhs - constraint.rhs)
        else:
            raise TypeError('Constraints have to be initiated with a subclass of sympy.Relational')

    @property
    # @cache
    def jacobian(self):
        """
        :return: Jacobian 'Matrix' filled with the symbolic expressions for all the partial derivatives.
            Partial derivatives are of the components of the function with respect to the Parameter's,
            not the independent Variable's.
        """
        return [[sympy.diff(expr, param) for param in self.model.params] for expr in self.values()]

    @property
    # @cache
    def numerical_components(self):
        """
        :return: lambda functions of each of the components in model_dict, to be used in numerical calculation.
        """
        return [sympy_to_py(expr, self.model.vars, self.model.params) for expr in self.values()]

    @property
    # @cache
    def numerical_jacobian(self):
        """
        :return: lambda functions of the jacobian matrix of the function, which can be used in numerical optimization.
        """
        return [[sympy_to_py(partial, self.model.vars, self.model.params) for partial in row] for row in self.jacobian]

    def _make_signature(self):
        # Handle args and kwargs according to the allowed names.
        parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
            inspect_sig.Parameter(arg.name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD)
                for arg in self.model.vars + self.model.params
        ]
        return inspect_sig.Signature(parameters=parameters)


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
        :param absolute_sigma bool: True by default. If the sigma is only used
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
        with sigma_. For example, let x be a Variable. Then sigma_x will give the
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
        bound_arguments = signature.bind(*ordered_data, **named_data)
        # Include default values in bound_argument object
        for param in signature.parameters.values():
            if param.name not in bound_arguments.arguments:
                bound_arguments.arguments[param.name] = param.default

        self.data = copy.copy(bound_arguments.arguments)   # ordereddict of the data. Only copy the dict, not the data.
        # self.sigmas = {name: self.data.pop(name) for name in var_names if name.startswith('sigma_')}
        self.sigmas = {name: self.data[name] for name in var_names if name.startswith('sigma_')}

        # Replace sigmas that are constant by an array of that constant
        for var, sigma in self.model.sigmas.items():
            try:
                iter(self.data[sigma.name])
            except TypeError:
                # if self.data[var.name] is None and self.data[sigma.name] is None:
                #     pass # This is the way it's supposed to be.
                if self.data[var.name] is not None and self.data[sigma.name] is None:
                    self.data[sigma.name] = np.ones(self.data[var.name].shape)
                elif self.data[var.name] is not None:
                    self.data[sigma.name] *= np.ones(self.data[var.name].shape)

        # If user gives a preference, use that. Otherwise, use True if at least one sigma is
        # given, False if no sigma is given.
        if absolute_sigma is not None:
            self.absolute_sigma = absolute_sigma
        else:
            for name, value in self.sigmas.items():
                if value is not None:
                    self.absolute_sigma = True
                    break
            else:
                self.absolute_sigma = False

    @property
    @cache
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable.
        :rtype: OrderedDict with variable names as key, data as value.
        """
        return OrderedDict((var.name, self.data[var.name]) for var in self.model)

    @property
    @cache
    def independent_data(self):
        """
        Read-only Property

        :return: Data belonging to each independent variable.
        :rtype: OrderedDict with variable names as key, data as value.
        """
        return OrderedDict((var.name, self.data[var.name]) for var in self.model.independent_vars)

    @property
    @cache
    def sigma_data(self):
        """
        Read-only Property

        :return: Data belonging to each sigma variable.
        :rtype: OrderedDict with variable names as key, data as value.
        """
        sigmas = self.model.sigmas
        return OrderedDict((sigmas[var].name, self.data[sigmas[var].name]) for var in self.model)

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


class NumericalLeastSquares(BaseFit):
    """
    Solves least squares numerically using leastsqbounds. Gives results consistent with MINPACK except
    when borders are provided.
    """
    def execute(self, *options, **kwoptions):
        """
        :param options: Any postional arguments to be passed to leastsqbound
        :param kwoptions: Any named arguments to be passed to leastsqbound
        """
        if hasattr(self.model, 'numerical_jacobian'):
            Dfun = self.eval_jacobian
        else:
            Dfun = None

        try:
            popt, cov_x, infodic, mesg, ier = leastsqbound(
                self.error_func,
                Dfun=Dfun,
                args=(self.independent_data, self.dependent_data, self.sigma_data,),
                x0=self.initial_guesses,
                bounds=self.model.bounds,
                full_output=True,
                *options,
                **kwoptions
            )
        except ValueError as err:
            # The exact Jacobian can contain nan's, causing the fit to fail. In such cases, try again without providing an exact jacobian.
            popt, cov_x, infodic, mesg, ier = leastsqbound(
                self.error_func,
                args=(self.independent_data, self.dependent_data, self.sigma_data,),
                x0=self.initial_guesses,
                bounds=self.model.bounds,
                full_output=True,
                *options,
                **kwoptions
            )

        if self.absolute_sigma:
            s_sq = 1
        else:
            # Rescale the covariance matrix with the residual variance
            ss_res = np.sum(infodic['fvec']**2)
            for data in self.dependent_data.values():
                if data is not None:
                    degrees_of_freedom = np.product(data.shape) - len(popt)
                    break

            s_sq = ss_res / degrees_of_freedom

        pcov = cov_x * s_sq if cov_x is not None else None

        self._fit_results = FitResults(
            params=self.model.params,
            popt=popt,
            pcov=pcov,
            infodic=infodic,
            mesg=mesg,
            ier=ier,
            # ydata=list(self.data.values())[0] if len(self.model.dependent_vars) == 1 else None,
            # sigma=self.sigma,
        )
        self._fit_results.r_squared = r_squared(self.model, self._fit_results, self.data)
        return self._fit_results


    def error_func(self, p, independent_data, dependent_data, sigma_data, flatten=True):
        """
        Returns the value of the square root of :math:`\\chi^2`, summing over the components.

        This function now supports setting variables to None. Needs mathematical rigor!

        :param p: array of parameter values.
        :param independent_data: Data to provide to the independent variables.
        :param dependent_data:
        :param sigma_data:
        :param flatten: If True, summing is performed over the data indices (default).
        :return: :math:`\\sqrt(\\chi^2)`
        """
        # import matplotlib.pyplot as plt

        result = []
        jac_args = list(independent_data.values()) + list(p)

        # zip together the dependent vars and evaluated component
        for y, ans in zip(self.model, self.model(*jac_args)):
            # if 'd' in y.name:
            #     plt.plot(jac_args[0], ans, label=str(p))
            if dependent_data[y.name] is not None:
                result.append(((dependent_data[y.name] - ans)/sigma_data[self.model.sigmas[y].name])**2)
                if flatten:
                    result[-1] = result[-1].flatten()
        return np.sqrt(sum(result))

    def eval_jacobian(self, p, independent_data, dependent_data, sigma_data):
        chi = self.error_func(p, independent_data, dependent_data, sigma_data, flatten=False)
        jac_args = list(independent_data.values()) + list(p)
        result = len(self.model.params) * [0.0]
        for ans, y, row in zip(self.model(*jac_args), self.model, self.model.numerical_jacobian):
            if dependent_data[y.name] is not None:
                for index, component in enumerate(row):
                    result[index] += (1/chi) * component(*jac_args) * ((dependent_data[y.name] - ans)/sigma_data[self.model.sigmas[y].name]**2)
        result = [item.flatten() for item in result]
        return - np.array(result).T


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
        for expr in self.model.chi_squared_jacobian:
            # Collect terms linear in the parameters. Returns a dict with parameters and
            # their prefactor as function of variables. Includes O(1)
            terms = sympy.collect(sympy.expand(expr), self.model.params, evaluate=False)
            # Evaluate every term separately and 'sum out' the variables. This results in
            # a system that is very easy to solve.
            for param in terms:
                terms[param] = np.sum(terms[param](**self.data))

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
        # kwargs.update(self.data)
        X = np.atleast_2d([
            (np.ones(sigma.shape[0]) * comp(**kwargs)).flatten()
            for comp in self.model.numerical_jacobian[0]
        ])

        cov_matrix = np.linalg.inv(X.dot(W).dot(X.T))
        if not self.absolute_sigma:
            kwargs.update(self.data)
            # Sum of squared residuals. To be honest, I'm not sure why ss_res does not give the
            # right result but by using the chi_squared the results are compatible with curve_fit.
            S = np.sum(self.model.chi_squared(**kwargs), dtype=float) / (len(W) - len(self.model.params))
            cov_matrix *= S

        return cov_matrix

    def execute(self):
        """
        Execute an analytical (Linear) Least Squares Fit. This object works by symbolically
        solving when :math:`\\nabla \\chi^2 = 0`.

        To perform this task the expression of :math:`\\nabla \\chi^2` is determined, ignoring that
        :math:`\\chi^2` involves summing over all terms. Then the sum is performed by substituting
        the variables by their respective data and summing all terms, while leaving the parameters
        symbolic.

        The resulting system of equations is then easily solved with ``sympy.solve``.

        :return: ``FitResult``
        """
        # Obtain the best fit params first.
        best_fit_params = self.best_fit_params()
        cov_matrix = self.covariance_matrix(best_fit_params=best_fit_params)

        self._fit_results = FitResults(
            params=self.model.params,
            popt=[best_fit_params[param] for param in self.model.params],
            pcov=cov_matrix,
            infodic={'nfev': 0},
            mesg='',
            ier=0,
        )
        self._fit_results.r_squared = r_squared(self.model, self._fit_results, self.data)
        return self._fit_results


class NonLinearLeastSquares(BaseFit):
    """
    Experimental.
    Implements non-linear least squares. Works by a two step process:
    First the model is linearised by doing a first order taylor expansion
    around the guesses for the parameters.
    Then a LinearLeastSquares fit is performed. This is iterated until
    a fit of sufficient quality is obtained.

    Sensitive to good initial guesses. Providing good initial guesses is a must.

    .. [wiki] https://en.wikipedia.org/wiki/Non-linear_least_squares
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
        fit = LinearLeastSquares(self.model_appr, absolute_sigma=self.absolute_sigma, **self.data)

        # if fit.is_linear(self.model):
        #     return fit.execute()
        # else:
        i = 0
        S_k1 = np.sum(
            self.model.numerical_chi_squared(
                *self.data.values(),
                **{p.name: float(value) for p, value in zip(self.model.params, self.initial_guesses)}
            )
        )
        while i < max_iter:
            fit_params = fit.best_fit_params()
            S_k2 = np.sum(
                self.model.numerical_chi_squared(
                    *self.data.values(),
                    **{p.name: float(value) for p, value in fit_params.items()}
                )
            )
            if not S_k1 < 0 and np.abs(S_k2 - S_k1) <= relative_error * np.abs(S_k1):
                break
            else:
                S_k1 = S_k2
                # Update the model with a better approximation
                self.model_appr.p0 = fit_params
                i += 1

        cov_matrix = fit.covariance_matrix(best_fit_params=fit_params)

        self._fit_results = FitResults(
            params=self.model.params,
            popt=[float(fit_params[param]) for param in self.model.params],
            pcov=cov_matrix,
            infodic={'nfev': i},
            mesg='',
            ier=0,
        )
        self._fit_results.r_squared = r_squared(self.model, self._fit_results, self.data)
        return self._fit_results


class Fit(object):
    """
    Your one stop fitting solution! Based on the nature of the input, this
    object will attempt to select the right fitting type for your problem.

    If you need very specific control over how the problem is solved, please use
    one of the available fitting objects directly.

    Currently `Fit` will select between `NumericalLeastSquares` and
    `ConstrainedNumericalLeastSquares`.
    """
    def __init__(self, *args, **kwargs):
        if 'constraints' in kwargs:
            self.fit = ConstrainedNumericalLeastSquares(*args, **kwargs)
        else:
            init = TakesData(*args, **kwargs)
            # self.model = init.model
            if not all(min is None and max is None for min, max in init.model.bounds):
                # Bounds have been set.
                self.fit = ConstrainedNumericalLeastSquares(*args, **kwargs)
            else:
                if init.model.shared_parameters:
                    self.fit = ConstrainedNumericalLeastSquares(*args, **kwargs)
                else:
                    self.fit = NumericalLeastSquares(*args, **kwargs)

    def execute(self, *options, **kwoptions):
        """
        Execute ``Fit``, giving any ``options`` and ``kwoptions`` to the
        fitting object.
        """
        return self.fit.execute(*options, **kwoptions)

    @property
    def model(self):
        """ Property which returns self.fit.model """
        return self.fit.model

    @model.setter
    def model(self, new_model):
        self.fit.model = new_model


class Minimize(BaseFit):
    """
    Minimize a model subject to constraints. A wrapper for ``scipy.optimize.minimize``.

    ``Minimize`` itself doesn't work when data is provided to its Variables, use
    one of its subclasses for that.
    """
    @keywordonly(constraints=None)
    def __init__(self, model, *args, **kwargs):
        """
        Because in a lot of use cases for Minimize no data is supplied to variables,
        all the empty variables are replaced by an empty np array.

        :constraints: constraints the minimization is subject to.
        :type constraints: list
        """
        constraints = kwargs.pop('constraints')
        super(Minimize, self).__init__(model, *args, **kwargs)

        self.constraints = []
        if constraints:
            for constraint in constraints:
                if isinstance(constraint, Constraint):
                    self.constraints.append(constraint)
                else:
                    self.constraints.append(Constraint(constraint, self.model))


    def error_func(self, p, independent_data, dependent_data, sigma_data):
        """
        The function to be optimized. Scalar valued models are assumed. For
        Minimize the thing to minimize is simply `self.model` directly.

        :param p: array of floats for the parameters.
        :param data: data to be provided to ``Variable``'s.
        """
        jac_args = list(independent_data.values()) + list(p)
        ans, = self.model(*jac_args)
        return ans

    def eval_jacobian(self, p, independent_data, dependent_data, sigma_data):
        """
        Takes partial derivatives of model w.r.t. each ``Parameter``.

        :param p: array of floats for the parameters.
        :param data: data to be provided to ``Variable``'s.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        jac_args = list(independent_data.values()) + list(p)
        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(partial_derivative(*jac_args).flatten())
        else:
            return np.array(ans)

    def execute(self, method='SLSQP', *args, **kwargs):
        ans = minimize(
            self.error_func,
            self.initial_guesses,
            method=method,
            args=(self.independent_data, self.dependent_data, self.sigma_data,),
            bounds=self.model.bounds,
            constraints=self.scipy_constraints,
            jac=self.eval_jacobian if hasattr(self.model, 'numerical_jacobian') else None,
            # jac=self.eval_jacobian,
            *args,
            **kwargs
        )

        # Build infodic
        infodic = {
            'fvec': ans.fun,
            'nfev': ans.nfev,
        }
        # s_sq = (infodic['fvec'] ** 2).sum() / (len(self.ydata) - len(popt))
        # pcov = cov_x * s_sq if cov_x is not None else None

        self.__fit_results = FitResults(
            params=self.model.params,
            popt=ans.x,
            pcov=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit,
        )
        try:
            self.__fit_results.r_squared = r_squared(self.model, self.__fit_results, self.data)
        except ValueError:
            self.__fit_results.r_squared = float('nan')
        return self.__fit_results

    @property
    def scipy_constraints(self):
        """
        Read-only Property of all constraints in a scipy compatible format.

        :return: dict of scipy compatible statements.
        """
        cons = []
        types = { # scipy only distinguishes two types of constraint.
            sympy.Eq: 'eq', sympy.Gt: 'ineq', sympy.Ge: 'ineq', sympy.Ne: 'ineq', sympy.Lt: 'ineq', sympy.Le: 'ineq'
        }

        for key, constraint in enumerate(self.constraints):
            # jac = make_jac(c, p)
            cons.append({
                'type': types[constraint.constraint_type],
                # Assume the lhs is the equation.
                'fun': lambda p, x, c: c(*(list(x.values()) + list(p)))[0],
                # Assume the lhs is the equation.
                'jac' : lambda p, x, c: [component(*(list(x.values()) + list(p))) for component in c.numerical_jacobian[0]],
                'args': (self.data, constraint)
            })
        cons = tuple(cons)
        return cons

# class Minimize(BaseFit):
#     def __init__(self, model, xdata=None, ydata=None, constraints=None, *args, **kwargs):
#         """
#         :model: Model to minimize
#         :constraints: constraints the minimization is subject to
#         :xdata:
#         :ydata: data the minimization is subject to.
#         """
#         super(Minimize, self).__init__(model)
#         self.xdata = xdata if xdata is not None else np.array([])
#         self.ydata = ydata if ydata is not None else np.array([])
#         self.constraints = constraints if constraints else []
#
#     def error(self, p, func, x, y):
#         if x != np.array([]) and y != np.array([]):
#             return func(x, p) - y
#         else:
#             return func(x, p)
#
#     def get_initial_guesses(self):
#         return super(Minimize, self).get_initial_guesses()
#
#     def execute(self, method='SLSQP', *args, **kwargs):
#         ans = minimize(
#             self.error,
#             self.get_initial_guesses(),
#             args=(self.scipy_func, self.xdata, self.ydata),
#             method=method,
#             # method='L-BFGS-B',
#             bounds=self.get_bounds(),
#             constraints = self.get_constraints(),
#             jac=self.eval_jacobian,
#             options={'disp': True},
#         )
#
#         # Build infodic
#         infodic = {
#             'fvec': ans.fun,
#             'nfev': ans.nfev,
#         }
#         # s_sq = (infodic['fvec'] ** 2).sum() / (len(self.ydata) - len(popt))
#         # pcov = cov_x * s_sq if cov_x is not None else None
#         self.__fit_results = FitResults(
#             params=self.model.params,
#             popt=ans.x,
#             pcov=None,
#             infodic=infodic,
#             mesg=ans.message,
#             ier=ans.nit,
#             ydata=self.ydata,  # Needed to calculate R^2
#         )
#         return self.__fit_results
#
#     def get_constraints(self):
#         """
#             Turns self.constraints into a scipy compatible format.
#             :return: dict of scipy compatile statements.
#             """
#         from sympy import Eq, Gt, Ge, Ne, Lt, Le
#
#         cons = []
#         types = {
#             Eq: 'eq', Gt: 'ineq', Ge: 'ineq', Ne: 'ineq', Lt: 'ineq', Le: 'ineq'
#         }
#
#         def make_jac(constraint_lhs, p, x):
#             """
#             :param constraint_lhs: equation of a constraint. The lhs is assumed to be an eq, rhs a number.
#             :param p: current value of the parameters to be evaluated.
#             :return: numerical jacobian.
#             """
#             sym_jac = []
#             for param in self.model.params:
#                 sym_jac.append(sympy.diff(constraint_lhs, param))
#             ans = np.array(
#                 [sympy_to_scipy(jac, self.model.vars, self.model.params)(x, p) for jac in
#                  sym_jac]
#             )
#             return ans
#
#         for key, constraint in enumerate(self.constraints):
#             # jac = make_jac(c, p)
#             cons.append({
#                 'type': types[constraint.__class__],
#                 # Assume the lhs is the equation.
#                 'fun': lambda p, x, c: sympy_to_scipy(c.lhs, self.model.vars, self.model.params)(x, p),
#                 # Assume the lhs is the equation.
#                 'jac' : lambda p, x, c: make_jac(c.lhs, p, x),
#                 'args': (self.xdata, constraint)
#             })
#         cons = tuple(cons)
#         return cons



# class Maximize(Minimize):
#     def error(self, p, func, x, y):
#         """ Change the sign in order to maximize. """
#         return - super(Maximize, self).error(p, func, x, y)
#
#     def eval_jacobian(self, p, func, x, y):
#         """ Change the sign in order to maximize. """
#         return - super(Maximize, self).eval_jacobian(p, func, x, y)

class Maximize(Minimize):
    """
    Maximize a model subject to constraints.
    Simply flips the sign on error_func and eval_jacobian in order to maximize.
    """
    def error_func(self, p, independent_data, dependent_data, sigma_data):
        return - super(Maximize, self).error_func(p, independent_data, dependent_data, sigma_data)

    def eval_jacobian(self, p, independent_data, dependent_data, sigma_data):
        return - super(Maximize, self).eval_jacobian(p, independent_data, dependent_data, sigma_data)

class Likelihood(Maximize):
    """
    Fit using a Maximum-Likelihood approach. This object maximizes the
    log-likelihood function.
    """
    # def __init__(self, model, *args, **kwargs):
    #     """
    #     :param model: sympy expression.
    #     :param x: xdata to fit to.  Nx1
    #     """
    #     super(Likelihood, self).__init__(model, *args, **kwargs)

    # def execute(self, method='SLSQP', *args, **kwargs):
    #     # super(Likelihood, self).execute(*args, **kwargs)
    #     ans = minimize(
    #         self.error,
    #         self.initial_guesses,
    #         args=(self.scipy_func, self.xdata, self.ydata),
    #         method=method,
    #         bounds=self.get_bounds(),
    #         constraints = self.get_constraints(),
    #         # jac=self.eval_jacobian, # If I find a meaning to jac I'll let you know.
    #         options={'disp': True},
    #     )
    #
    #     # Build infodic
    #     infodic = {
    #         'fvec': ans.fun,
    #         'nfev': ans.nfev,
    #     }
    #
    #
    #
    #     self.__fit_results = FitResults(
    #         params=self.model.params,
    #         popt=ans.x,
    #         pcov=None,
    #         infodic=infodic,
    #         mesg=ans.message,
    #         ier=ans.nit,
    #         ydata=self.ydata,  # Needed to calculate R^2
    #     )
    #     return self.__fit_results

    def error_func(self, p, independent_data, dependent_data, sigma_data):
        """
        Error function to be maximised(!) in the case of log-likelihood fitting.

        :param p: guess params
        :param data: xdata
        :return: scalar value of log-likelihood
        """
        jac_args = list(independent_data.values()) + list(p)
        ans = - np.nansum(np.log(self.model(*jac_args)))
        return ans

    def eval_jacobian(self, p, independent_data, dependent_data, sigma_data):
        """
        Jacobian for log-likelihood is defined as :math:`\\nabla_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param p: guess params
        :param data: data for the variables.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        jac_args = list(independent_data.values()) + list(p)
        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(
                    - np.nansum(
                        partial_derivative(*jac_args).flatten() / self.model(*jac_args)
                    )
                )
        else:
            return np.array(ans)

class HasCovarianceMatrix(object):
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
        if any(element is None for element in self.sigma_data.values()):
            return np.array(
                [[float('nan') for p in self.model.params] for p in self.model.params]
            )
        if len(set(arr.shape for arr in self.sigma_data.values())) == 1:
            # All shapes identical
            return self._cov_mat_equal_lenghts(best_fit_params=best_fit_params)
        else:
            return self._cov_mat_unequal_lenghts(best_fit_params=best_fit_params)

    def _reduced_residual_ss(self, best_fit_params, flatten=True):
        """
        Calculate the residual Sum of Squares divided by the d.o.f..
        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        :param flatten: when `True`, return the total sum of squares (SS).
            If `False`, return the componentwise SS.
        :return: The reduced residual sum of squares.
        """
        popt = [best_fit_params[p.name] for p in self.model.params]
        # Rescale the covariance matrix with the residual variance
        ss_res = self.error_func(
            [best_fit_params[p.name] for p in self.model.params],
            self.independent_data,
            self.dependent_data,
            self.sigma_data,
            flatten_components=flatten
        )

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
                    degrees_of_freedom.append(len(popt) + 1)
        degrees_of_freedom = np.array(degrees_of_freedom)
        s_sq = ss_res / (degrees_of_freedom - len(popt))
        # s_sq = ss_res / degrees_of_freedom

        return s_sq

    def _cov_mat_equal_lenghts(self, best_fit_params):
        """
        If all the data arrays are of equal size, use this method. This will
        typically be the case, and this method is a lot faster because it allows
        for numpy magic.

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        """
        sigma = np.vstack(list(self.sigma_data.values()))
        # Weight matrix. Since it should be a diagonal matrix, we just remember
        # this and multiply it elementwise for efficiency.
        # It is also rescaled by the reduced residual ss in case of absolute_sigma==False
        if self.absolute_sigma:
            W = 1/sigma**2
        else:
            s_sq = self._reduced_residual_ss(best_fit_params, flatten=False)
            W = 1/sigma**2/s_sq[:, np.newaxis]

        kwargs = {p.name: best_fit_params[p.name] for p in self.model.params}
        kwargs.update(self.independent_data)
        jac = np.atleast_2d([
            [
                np.ones(sigma.shape[1]) * comp(**kwargs) for comp in row
            ] for row in self.model.numerical_jacobian
        ])
        # Order jacobian as param, component, datapoint
        jac = np.swapaxes(jac,0,1)

        # Dot away all but the parameter dimension!
        cov_matrix_inv = np.tensordot(W*jac, jac, (range(1, jac.ndim), range(1, jac.ndim)))
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

        kwargs = {p.name: best_fit_params[p.name] for p in self.model.params}
        kwargs.update(self.independent_data)
        jac = [
            [
                np.ones(s.shape) * comp(**kwargs) for comp in row
            ] for row, s in zip(self.model.numerical_jacobian, sigma)
        ]

        # Order jacobian as param, component, datapoint
        jac = np.swapaxes(jac,0,1)
        # Weigh each component with its respective weight.
        jac_weighed = [[j * w for j, w in zip(row, W)] for row in jac]

        # Buil the inverse cov_matrix.
        cov_matrix_inv = []
        # iterate along the parameters first
        for index, jac_w_p in enumerate(jac_weighed):
            cov_matrix_inv.append([])
            for jac_p in jac:
                # Now we have to dot product these guys properly.
                dot = np.sum([np.sum(a * b) for a, b in zip(jac_w_p, jac_p)])
                cov_matrix_inv[index].append(dot)

        cov_matrix = np.linalg.inv(cov_matrix_inv)
        return cov_matrix

class ConstrainedNumericalLeastSquares(Minimize, HasCovarianceMatrix):
    """
    This object performs :math:`\chi^2` minimization, subject to constraints.
    As an example, we could imagine fitting the angles of a triangle::

        a, b, c = parameters('a, b, c')
        a_i, b_i, c_i = variables('a_i, b_i, c_i')

        model = {a_i: a, b_i: b, c_i: c}

        data = np.array([
            [10.1, 9., 10.5, 11.2, 9.5, 9.6, 10.],
            [102.1, 101., 100.4, 100.8, 99.2, 100., 100.8],
            [71.6, 73.2, 69.5, 70.2, 70.8, 70.6, 70.1],
        ])

        fit = ConstrainedNumericalLeastSquares(
            model=model,
            a_i=data[0],
            b_i=data[1],
            c_i=data[2],
            constraints=[Equality(a + b + c, 180)]
        )
        fit_result = fit.execute()

    Unlike `NumericalLeastSquares`, it also supports vector components of unequal
    length and is therefore preferred for Global Fitting problems.

    In order to perform minimization, this object is a subclass of `Minimize`,
    and the output might therefore deviate slightly from the MINPACK result given
    by the more traditional `NumericalLeastSquares` object.
    """
    def error_func(self, p, independent_data, dependent_data, sigma_data, flatten_components=True):
        """
        Returns :math:`\\chi^2`, summing over all the vector components and
        data indices.

        This function now supports setting variables to None. Needs mathematical rigor!

        :param p: array of floats for the parameters.
        :param data: data to be provided to ``Variable``'s.
        :param flatten_components: If ``True``, :math:`\\chi^2` is returned.
            If ``False``, the :math:`\\chi^2` per vector component is returned.
        """
        jac_args = list(independent_data.values()) + list(p)
        evaluated_func = self.model(*jac_args)

        chi2 = [0 for _ in evaluated_func]
        for index, (dep_var_name, dep_var_value) in enumerate(evaluated_func._asdict().items()):
            dep_data = dependent_data[dep_var_name]
            if dep_data is not None:
                sigma = sigma_data['sigma_{}'.format(dep_var_name)] # Should be changed with #41
                chi2[index] += np.sum((dep_var_value - dep_data)**2/sigma**2)
                # chi2 += np.sum((dep_var_value - dep_data)**2/sigma**2)
        chi2 = np.sum(chi2) if flatten_components else chi2
        return chi2

    def eval_jacobian(self, p, independent_data, dependent_data, sigma_data):
        """
        Evaluates the jacobian of :math:`\\chi^2`, summing over all the vector
        components and data indices.

        :return: array of len(self.model.params) containing the components of the Jacobian.
        """
        jac_args = list(independent_data.values()) + list(p)
        evaluated_func = self.model(*jac_args)
        result = [0.0 for _ in self.model.params]

        for ans, y, row in zip(evaluated_func, self.model, self.model.numerical_jacobian):
            dep_data = dependent_data[y.name]
            if dep_data is not None:
                sigma = sigma_data['sigma_{}'.format(y.name)] # Should be changed with #41
                for index, component in enumerate(row):
                    result[index] += np.sum(
                        component(*jac_args) * ((dep_data - ans)/sigma**2)
                    )
        return - np.array(result).T

    @keywordonly(tol=1e-9)
    def execute(self, *args, **kwargs):
        """
        This wraps the execute of 'Minimize' with the calculation of the
        covariance matrix. Read `Minimize.execute` for a more general
        description.
        """
        fit_result = super(ConstrainedNumericalLeastSquares, self).execute(*args, **kwargs)
        # Extract the best fit parameters. Replace by fit_result.params.values() if #45 is fixed.
        popt = [fit_result.value(p) for p in self.model.params]

        try:
            cov_matrix = self.covariance_matrix(fit_result.params)
        except np.linalg.linalg.LinAlgError:
            return fit_result  # Return without covariance matrix since it is singular
        else:
            if len(self.model) > 1:
                # For vector-models, make all off-diagonal values nan
                cov_matrix[~np.eye(*cov_matrix.shape, dtype=bool)] = float('nan')

            results = FitResults(
                params=self.model.params,
                popt=popt,
                pcov=cov_matrix,
                infodic=fit_result.infodict,
                mesg=fit_result.status_message,
                ier=fit_result.iterations,
            )
            results.r_squared = fit_result.r_squared
            return results

# class LagrangeMultipliers:
#     """
#     Class to analytically solve a function subject to constraints using Karush Kuhn Tucker.
#     http://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions
#     """
#
#     def __init__(self, model, constraints):
#         self.model = model
#         # Seperate the constraints into equality and inequality constraint of the type <=.
#         self.equalities, self.lesser_thans = self.seperate_constraints(constraints)
#         self.model.vars, self.model.params = seperate_symbols(self.model)
#
#     @property
#     @cache
#     def lagrangian(self):
#         L = self.model
#
#         # Add equility constraints to the Lagrangian
#         for constraint, l_i in zip(self.equalities, self.l_params):
#             L += l_i * constraint
#
#         # Add inequility constraints to the Lagrangian
#         for constraint, u_i in zip(self.lesser_thans, self.u_params):
#             L += u_i * constraint
#
#         return L
#
#     @property
#     @cache
#     def l_params(self):
#         """
#         :return: Lagrange multipliers for every constraint.
#         """
#         return [Parameter(name='l_{}'.format(index)) for index in range(len(self.equalities))]
#
#     @property
#     @cache
#     def u_params(self):
#         """
#         :return: Lagrange multipliers for every inequality constraint.
#         """
#         return [Parameter(name='u_{}'.format(index)) for index in range(len(self.lesser_thans))]
#
#     @property
#     @cache
#     def all_params(self):
#         """
#         :return: All parameters. The convention is first the model parameters,
#         then lagrange multipliers for equality constraints, then inequility.
#         """
#         return self.model.params + self.l_params + self.u_params
#
#     @property
#     @cache
#     def extrema(self):
#         """
#         :return: list namedtuples of all extrema of self.model, where value = f(x1, ..., xn).
#         """
#         # Prepare the Extremum namedtuple for this number of variables.
#         field_names = [p.name for p in self.model.params] + ['value']
#         Extremum = namedtuple('Extremum', field_names)
#
#         # Calculate the function value at each solution.
#         values = [self.model.subs(sol) for sol in self.solutions]
#
#         # Build the output list of namedtuples
#         extrema_list = []
#         for value, solution in zip(values, self.solutions):
#             # Prepare an Extrumum tuple for every extremum.
#             ans = {'value': value}
#             for param in self.model.params:
#                 ans[param.name] = solution[param]
#             extrema_list.append(Extremum(**ans))
#         return extrema_list
#
#     @property
#     @cache
#     def solutions(self):
#         """
#         Do analytical optimization. This finds ALL solutions for the system.
#         Nomenclature: capital L is the Lagrangian, l the Lagrange multiplier.
#         :return: a list of dicts containing the values for all parameters,
#         including the Lagrange multipliers l_i and u_i.
#         """
#         # primal feasibility; pretend they are all equality constraints.
#         grad_L = [sympy.diff(self.lagrangian, p) for p in self.all_params]
#         solutions = sympy.solve(grad_L, self.all_params, dict=True)
#         print(grad_L, solutions, self.all_params)
#
#         if self.u_params:
#             # The smaller than constraints also have trivial solutions when u_i == 0.
#             # These are not automatically found by sympy in the previous process.
#             # Therefore we must now evaluate the gradient for these points manually.
#             u_zero = dict((u_i, 0) for u_i in self.u_params)
#             # We need to consider all combinations of u_i == 0 possible, of all lengths possible.
#             for number_of_zeros in range(1, len(u_zero) + 1):
#                 for zeros in itertools.combinations(u_zero.items(), number_of_zeros):  # zeros is a tuple of (Symbol, 0) tuples.
#                     # get a unique set of symbols.
#                     symbols = set(self.all_params) - set(symbol for symbol, _ in zeros)
#                     # differentiate w.r.t. these symbols only.
#                     relevant_grad_L = [sympy.diff(self.lagrangian, p) for p in symbols]
#
#                     solution = sympy.solve([grad.subs(zeros) for grad in relevant_grad_L], symbols, dict=True)
#                     for item in solution:
#                         item.update(zeros)  # include the zeros themselves.
#
#                     solutions += solution
#
#         return self.sanitise(solutions)
#
#     def sanitise(self, solutions):
#         """
#         Returns only solutions which are valid. This is an unfortunate consequence of the KKT method;
#         KKT parameters are not guaranteed to respect each other. However, it is easy to check this.
#         There are two things to check:
#         - all KKT parameters should be greater equal zero.
#         - all constraints should be met by the solutions.
#         :param solutions: a list of dicts, where each dict contains the coordinates of a saddle point of the lagrangian.
#         :return: bool
#         """
#         # All the inequality multipliers u_i must be greater or equal 0
#         final_solutions = []
#         for saddle_point in solutions:
#             for u_i in self.u_params:
#                 if saddle_point[u_i] < 0:
#                     break
#             else:
#                 final_solutions.append(saddle_point)
#
#         # we have to dubble check all if all our conditions are met because
#         # This is not garanteed with inequility constraints.
#         solutions = []
#         for solution in final_solutions:
#             for constraint in self.lesser_thans:
#                 test = constraint.subs(solution)
#                 if test > 0:
#                     break
#             else:
#                 solutions.append(solution)
#
#         return solutions
#
#
#
#     @staticmethod
#     def seperate_constraints(constraints):
#         """
#         We follow the definitions given here:
#         http://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions
#
#         IMPORTANT: <= and < are considered the same! The same goes for > and >=.
#         Strict inequalities of the type != are not currently supported.
#
#         :param constraints list: list of constraints.
#         :return: g_i are <= 0 constraints, h_j are equals 0 constraints.
#         """
#         equalities = []
#         lesser_thans = []
#         for constraint in constraints:
#             if isinstance(constraint, sympy.Eq):
#                 equalities.append(constraint.lhs - constraint.rhs)
#             elif isinstance(constraint, (sympy.Le, sympy.Lt)):
#                 lesser_thans.append(constraint.lhs - constraint.rhs)
#             elif isinstance(constraint, (sympy.Ge, sympy.Gt)):
#                 lesser_thans.append(-1 * (constraint.lhs - constraint.rhs))
#             else:
#                 raise TypeError('Constraints of type {} are not supported by this solver.'.format(type(constraint)))
#         return equalities, lesser_thans
#
# class ConstrainedFit(BaseFit):
#     """
#     Finds the analytical best fit parameters, combining data with LagrangeMultipliers
#     for the best result, if available.
#     """
#     def __init__(self, model, x, y, constraints=None, *args, **kwargs):
#         constraints = constraints if constraints is not None else []
#         value = Variable()
#         chi2 = (model - value)**2
#         self.analytic_fit = LagrangeMultipliers(chi2, constraints)
#         self.xdata = x
#         self.ydata = y
#         super(ConstrainedFit, self).__init__(chi2)
#
#     def execute(self):
#         print('here:', self.analytic_fit.solutions)
#         import inspect
#         for extremum in self.analytic_fit.extrema:
#             popt, pcov  = [], []
#             for param in self.model.params:
#                 # Retrieve the expression for this param.
#                 expr = getattr(extremum, param.name)
#                 py_expr = sympy_to_py(expr, self.model.vars, [])
#                 values = py_expr(*self.xdata)
#                 popt.append(np.average(values))
#                 pcov.append(np.var(values, ddof=len(self.model.vars)))
#             print(popt, pcov)
#
#             residuals = self.scipy_func(self.xdata, popt)
#
#             fit_results = FitResults(
#                 params=self.model.params,
#                 popt=popt,
#                 pcov=pcov,
#                 infodic={},
#                 mesg='',
#                 ier=0,
#                 r_squared=r_squared(residuals, self.ydata),
#             )
#             print(fit_results)
#         print(self.analytic_fit.extrema)
#
#     def error(self, p, func, x, y):
#         pass

def r_squared(model, fit_result, data):
    """
    Calculates the coefficient of determination, R^2, for the fit.

    (Is not defined properly for vector valued functions.)

    :param model: Model instance
    :param fit_result: FitResults instance
    :param data: data with which the fit was performed.
    """
    # First filter out the dependent vars
    y_is = [data[var.name] for var in model if var.name in data]
    x_is = [value for key, value in data.items() if key in model.__signature__.parameters]
    y_bars = [np.mean(y_i) for y_i in y_is if y_i is not None]
    f_is = model(*x_is, **fit_result.params)
    SS_res = np.sum([np.sum((y_i - f_i)**2) for y_i, f_i in zip(y_is, f_is) if y_i is not None])
    SS_tot = np.sum([np.sum((y_i - y_bar)**2) for y_i, y_bar in zip(y_is, y_bars) if y_i is not None])

    return 1 - SS_res/SS_tot

class ODEModel(CallableModel):
    """
    Model build from a system of ODEs. When the model is called, the ODE is
    integrated using the LSODA package.

    Currently the initial conditions are assumed to specify the
    first point to begin the integration from. This is enforced. In future
    versions one should be allowed to specify the initial value as a parameter.
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

    @property
    @cache
    def _ncomponents(self):
        return [sympy_to_py(expr, self.independent_vars + self.dependent_vars, self.params) for expr in self.values()]

    @property
    @cache
    def _njacobian(self):
        return [sympy_to_py(sympy.diff(expr, var), self.independent_vars + self.dependent_vars, self.params) for var, expr in self.items()]
        # return [sympy_to_py(sympy.diff(expr, var), self.independent_vars + self.dependent_vars, self.params) for var, expr in self.items()]

    def eval_components(self, *args, **kwargs):
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        t_like = bound_arguments.arguments[self.independent_vars[0].name]

        # System of functions to be integrated
        f = lambda ys, t, *a: [c(t, *(list(ys) + list(a))) for c in self._ncomponents]
        Dfun = lambda ys, t, *a: [c(t, *(list(ys) + list(a))) for c in self._njacobian]

        initial_dependent = [self.initial[var] for var in self.dependent_vars]
        initial_independent = self.initial[self.independent_vars[0]] # Assuming there's only one

        # Check if the time-like data includes the initial value, because integration should start there.
        # For fitting to make sence, it should probably not be in there though. Needs mathematical backing.
        try:
            t_like[0]
        except (TypeError, IndexError): # Python scalar gives TypeError, numpy scalars IndexError
            t_like = np.array([t_like]) # Allow evaluation at one point.

        if t_like[0] == initial_independent:
            start = 0
            warnings.warn("The initial point should probably not be included with your data points as this point will always be fitted perfectly.")
        elif t_like[0] < initial_independent:
            raise ModelError('ODEModel\'s can not be evaluated for values smaller than the initial value')
        else:
            assert len(t_like.shape) == 1
            t_like = np.hstack((np.array([initial_independent]), t_like))
            start = 1

        ans = odeint(
            f,
            initial_dependent,
            t_like,
            args=tuple(bound_arguments.arguments[param.name] for param in self.params),
            Dfun=Dfun,
            *self.lsoda_args, **self.lsoda_kwargs
        )
        return ans[start:].T

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
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        Ans = namedtuple('Ans', [var.name for var in self])
        ans = Ans(*self.eval_components(**bound_arguments.arguments))
        return ans

