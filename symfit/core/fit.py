import abc
from collections import namedtuple, defaultdict, Mapping, OrderedDict
import itertools
import functools
import copy
import sys

import sympy
from sympy.core.relational import Relational
import numpy as np
from scipy.optimize import minimize

from symfit.core.argument import Parameter, Variable
from symfit.core.support import seperate_symbols, keywordonly, sympy_to_py, cache, jacobian
from symfit.core.leastsqbound import leastsqbound

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig

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

    def get_value(self, param):
        """
        :param param: ``Parameter`` instance.
        :return: returns the numerical value of param
        """
        assert isinstance(param, Parameter)
        return self.__popt[param.name]

    def get_stdev(self, param):
        """
        :param param: ``Parameter`` instance.
        :return: returns the standard deviation of param
        """
        assert isinstance(param, Parameter)
        return self.__pstdev[param.name]


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
        Excuse the ugly names of most of these variables, they are inherited. Should be changed.
        from scipy.
        :param params: list of ``Parameter``'s.
        :param popt: best fit parameters, same ordering as in params.
        :param pcov: covariance matrix.
        :param infodic: dict with fitting info.
        :param mesg: Status message.
        :param ier: Number of iterations.
        :param ydata:
        """
        # Validate the types in rough way
        assert(type(infodic) == dict)
        self.__infodict = infodic
        assert(type(mesg) == str)
        self.__status_message = mesg
        assert(type(ier) == int)
        self.__iterations = ier
        # assert(type(ydata) == np.ndarray)
        self.__ydata = ydata
        self.__params = ParameterDict(params, popt, pcov)
        self.__sigma = sigma

    def __str__(self):
        """
        Pretty print the results as a table.
        :return:
        """
        res = '\nParameter Value        Standard Deviation\n'
        for p in self.params:
            value = self.params.get_value(p)
            value_str = '{:e}'.format(value) if value is not None else 'None'
            stdev = self.params.get_stdev(p)
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

class Model(object):
    """
    Model represents a symbolic function and all it's derived properties such as sum of squares, jacobian etc.
    Models can be initiated from several objects::

        a = Model.from_dict({y: x**2})
        b = Model(y=x**2)

    Models are callable. The usual rules apply to the ordering of the arguments:

    * first independent variables, then dependent variables, then parameters.
    * within each of these groups they are ordered alphabetically.
    """
    def __init__(self, *ordered_expressions, **named_expressions):
        """
        Initiate a Model from keyword arguments::

            b = Model(y=x**2)

        :param ordered_expressions: sympy Expr
        :param named_expressions: sympy Expr
        """
        model_dict = {sympy.Dummy('y_{}'.format(index + 1)): expr for index, expr in enumerate(ordered_expressions)}
        model_dict.update(
            {Variable(name=dep_var_name): expr for dep_var_name, expr in named_expressions.items()}
        )
        self._init_from_dict(model_dict)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Initiate a Model from a dict::

            a = Model.from_dict({y: x**2})

        Prefered syntax.

        :param model_dict: dict of ``Expr``, where dependent variables are the keys.
        """
        self = cls()
        self._init_from_dict(model_dict)

        return self

    def _init_from_dict(self, model_dict):
        """
        Initiate self from a model_dict to make sure attributes such as vars, params are available.

        Creates lists of alphabetically sorted independent vars, dependent vars, sigma vars, and parameters.
        Finally it creates a signature for this model so it can be called nicely. This signature only contains
        independent vars and params, as one would expect.

        :param model_dict: dict of (dependent_var, expression) pairs.
        """
        self.model_dict = model_dict
        self.dependent_vars = sorted(model_dict.keys(), key=lambda symbol: symbol.name)

        # Extract all the params and vars as a sorted, unique list.
        expressions = model_dict.values()
        self.params, self.independent_vars = set([]), set([])
        for expression in expressions:
            vars, params = seperate_symbols(expression)
            self.params.update(params)
            self.independent_vars.update(vars)
        # Although unique now, params and vars should be sorted alphabetically to prevent ambiguity
        self.params = sorted(self.params, key=lambda symbol: symbol.name)
        self.independent_vars = sorted(self.independent_vars, key=lambda symbol: symbol.name)
        # Make Variable object corresponding to each var.
        self.sigmas = {var: Variable(name='sigma_{}'.format(var.name)) for var in self.dependent_vars}

        self.__signature__ = self._make_signature()

    def _make_signature(self):
        # Handle args and kwargs according to the allowed names.
        parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
            inspect_sig.Parameter(arg.name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD)
                for arg in self.independent_vars + self.params
        ]
        return inspect_sig.Signature(parameters=parameters)

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
        print(self.__signature__)
        print(args,kwargs)
        bound_arguments = self.__signature__.bind(*args, **kwargs)
        Ans = namedtuple('Ans', [var.name for var in self.dependent_vars])
        return Ans(*[expression(**bound_arguments.arguments) for expression in self.numerical_components])

    def __str__(self):
        """
        Pretty print this model.

        :return: str
        """
        template = "{}({}; {}) = {}"
        parts = []
        for var in self.dependent_vars:
            parts.append(template.format(
                    var,
                    ", ".join(arg.name for arg in self.independent_vars),
                    ", ".join(arg.name for arg in self.params),
                    self.model_dict[var]
                )
            )
        return "\n".join(parts)

    @property
    @cache
    def chi_squared(self):
        """
        :return: Symbolic :math:`\\chi^2`
        """
        # return sum((y - f)**2/self.sigmas[y]**2 for y, f in self.model_dict.items())
        return sum(((f - y)/self.sigmas[y])**2 for y, f in self.model_dict.items())

    @property
    @cache
    def chi(self):
        """
        :return: Symbolic Square root of :math:`\\chi^2`. Required for MINPACK optimization only. Denoted as :math:`\\sqrt(\\chi^2)`
        """
        return sympy.sqrt(self.chi_squared)#.replace(sympy.Abs, sympy.Id)

    @property
    @cache
    def chi_jacobian(self):
        """
        Return a symbolic jacobian of the :math:`\\sqrt(\\chi^2)` function.
        Vector of derivatives w.r.t. each parameter. Not a Matrix but a vector! This is because that's what leastsq needs.
        """
        jac = []
        for param in self.params:
            # Differentiate to every param
            f = sympy.diff(self.chi, param)
            # f_denest = powdenest(f, force=True)
            # jac.append(f_denest.replace(sympy.Abs, sympy.Id))
            jac.append(f)
        return jac

    @property
    @cache
    def jacobian(self):
        """
        :return: Jacobian 'Matrix' filled with the symbolic expressions for all the partial derivatives.
        Partial derivatives are of the components of the function with respect to the Parameter's,
        not the independent Variable's.
        """
        return [[sympy.diff(self.model_dict[var], param) for param in self.params] for var in self.dependent_vars]

    @property
    @cache
    def ss_res(self):
        """
        :return: Residual sum of squares. Similar to chi_squared, but without considering weights.
        """
        return sum((y - f)**2 for y, f in self.model_dict.items())

    @property
    @cache
    def numerical_chi_squared(self):
        """
        :return: lambda function of the ``.chi_squared`` method, to be used in numerical optimisation.
        """
        return sympy_to_py(self.chi_squared, self.vars, self.params)

    @property
    @cache
    def numerical_components(self):
        """
        :return: lambda functions of each of the components in model_dict, to be used in numerical calculation.
        """
        return [sympy_to_py(self.model_dict[var], self.independent_vars, self.params) for var in self.dependent_vars]

    @property
    @cache
    def numerical_chi(self):
        """
        :return: lambda function of the ``.chi`` method, to be used in MINPACK optimisation.
        """
        return sympy_to_py(self.chi, self.vars, self.params)

    @property
    @cache
    def numerical_chi_jacobian(self):
        """
        :return: lambda functions of the jacobian of the ``.chi`` method, which can be used in numerical optimization.
        """
        return [sympy_to_py(component, self.vars, self.params) for component in self.chi_jacobian]

    @property
    @cache
    def numerical_jacobian(self):
        """
        :return: lambda functions of the jacobian matrix of the function, which can be used in numerical optimization.
        """
        return [[sympy_to_py(partial, self.independent_vars, self.params) for partial in row] for row in self.jacobian]
        # return [[sympy_to_py(partial, self.vars, self.params) for partial in row] for row in self.jacobian]

    # @property
    # @cache
    # def numerical_chi_jacobian(self):
    #     """
    #     :return: lambda function of the jacobian, which can be used in numerical optimization.
    #     """
    #     return [sympy_to_py(component, self.vars, self.params) for component in self.jacobian(self.chi, self.params)]

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
        :return:
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
    @cache
    def jacobian(self):
        """
        :return: Jacobian 'Matrix' filled with the symbolic expressions for all the partial derivatives.
            Partial derivatives are of the components of the function with respect to the Parameter's,
            not the independent Variable's.
        """
        return [[sympy.diff(self.model_dict[var], param) for param in self.model.params] for var in self.dependent_vars]

    @property
    @cache
    def numerical_components(self):
        """
        :return: lambda functions of each of the components in model_dict, to be used in numerical calculation.
        """
        return [sympy_to_py(self.model_dict[var], self.model.vars, self.model.params) for var in self.dependent_vars]

    @property
    @cache
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


class BaseFit(object):
    """
    Abstract Base Class for all fitting objects. Most importantly, it takes care of linking the provided data to variables.
    The allowed variables are extracted from the model.
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
        if isinstance(model, Mapping):
            self.model = Model.from_dict(model)
        elif isinstance(model, Model):
            self.model = model
        else:
            self.model = Model(model)

        # Handle ordered_data and named_data according to the allowed names.
        var_names = [var.name for var in self.model.vars]
        parameters = [  # Note that these are inspect_sig.Parameter's, not symfit parameters!
            inspect_sig.Parameter(name, inspect_sig.Parameter.POSITIONAL_OR_KEYWORD, default=1 if name.startswith('sigma_') else None)
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

        # Replace sigmas that are one by an array of ones
        # for var, sigma in self.model.sigmas.items():
        #     print(var, sigma)
        #     if bound_arguments.arguments[sigma.name] == 1:
        #         bound_arguments.arguments[sigma.name] = np.ones(self.data[var.name].shape)

        # If user gives a preference, use that. Otherwise, use True if at least one sigma is
        # given, False if no sigma is given.
        if absolute_sigma is not None:
            self.absolute_sigma = absolute_sigma
        else:
            for name, value in self.sigmas.items():
                if value is not 1:
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
        :rtype: dict with variable names as key, data as value.
        """
        return {var.name: self.data[var.name] for var in self.model.dependent_vars}

    @property
    @cache
    def independent_data(self):
        """
        Read-only Property

        :return: Data belonging to each independent variable.
        :rtype: dict with variable names as key, data as value.
        """
        return {var.name: self.data[var.name] for var in self.model.independent_vars}

    @property
    @cache
    def sigma_data(self):
        """
        Read-only Property

        :return: Data belonging to each sigma variable.
        :rtype: dict with variable names as key, data as value.
        """
        return {var.name: self.data[var.name] for var in self.model.sigmas}

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

    @property
    def initial_guesses(self):
        """
        :return: Initial guesses for every parameter.
        """
        return np.array([param.value for param in self.model.params])

class AnalyticalFit(BaseFit):
   def execute(self, *args, **kwargs):
       """
       Analitically solve the Least Squares optimisation.
       """
       k = sympy.symbols('k', cls=sympy.Idx)
       chi_squared_jac = jacobian(sympy.Sum(self.model.chi_squared, (k, 1, len(list(self.data.items())[0][1]))), self.model.params)
       print(self.model.chi_squared)
       print(chi_squared_jac, self.model.params)

       sol = sympy.solve(chi_squared_jac, self.model.params[1], quick=True)#, dict=True)
       return sol

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

        try:
            popt, cov_x, infodic, mesg, ier = leastsqbound(
                self.error_func,
                # lambda p, data: self.model.numerical_chi(*(list(data) + list(p))).flatten(), # This lambda unpacking is needed because scipy is an inconsistent mess.
                Dfun=self.eval_jacobian,
                # Dfun=lambda p, data: np.array([component(*(list(data) + list(p))).flatten() for component in self.model.numerical_chi_jacobian]).T,
                args=(self.data.values(),),
                x0=self.initial_guesses,
                bounds=self.model.bounds,
                full_output=True,
                *options,
                **kwoptions
            )
        except ValueError:
            # The exact Jacobian can contain nan's, causing the fit to fail. In such cases, try again without providing an exact jacobian.
            popt, cov_x, infodic, mesg, ier = leastsqbound(
                self.error_func,
                # lambda p, data: self.model.numerical_chi(*(list(data) + list(p))).flatten(),
                args=(self.data.values(),),
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
            # degrees_of_freedom = len(self.data) - len(popt)
            degrees_of_freedom = len(self.data[self.model.dependent_vars[0].name]) - len(popt)

            s_sq = ss_res / degrees_of_freedom

        pcov = cov_x * s_sq if cov_x is not None else None

        self.__fit_results = FitResults(
            params=self.model.params,
            popt=popt,
            pcov=pcov,
            infodic=infodic,
            mesg=mesg,
            ier=ier,
            # ydata=list(self.data.values())[0] if len(self.model.dependent_vars) == 1 else None,
            # sigma=self.sigma,
        )
        self.__fit_results.r_squared = r_squared(self.model, self.__fit_results, self.data)
        return self.__fit_results


    def error_func(self, p, data):
        return self.model.numerical_chi(*(list(data) + list(p))).flatten()

    def eval_jacobian(self, p, data):
        return np.array([component(*(list(data) + list(p))).flatten() for component in self.model.numerical_chi_jacobian]).T


class AnalyticalLeastSquares(object):
    def execute(self):
        sol = sympy.solve(self.model.chi_jacobian, self.model.params, dict=True)


class Fit(NumericalLeastSquares):
    """
    Wrapper for NumericalLeastSquares to give it a more appealing name. In the future I hope to make this object more
    intelligent so it can search out the best fitting object based on certain qualifiers and return that instead.
    """
    pass


class Minimize(BaseFit):
    """
    Minimize a model subject to constraints. A wrapper for ``scipy.optimize.minimize``.
    ``Minimize`` currently doesn't work when data is provided to Variables, and doesn't support vector functions.
    """
    @keywordonly(constraints=None)
    def __init__(self, model, *args, **kwargs):
        """
        Because in a lot of use cases for Minimize no data is supplied to variables,
        all the empty variables are replaced by an empty np array.

        :constraints: constraints the minimization is subject to.
        :type constraints: list
        """
        # constraints = kwargs.pop('constraints') if 'constraints' in kwargs else None
        constraints = kwargs.pop('constraints')
        super(Minimize, self).__init__(model, *args, **kwargs)
        for var, data in self.data.items():
            if data is None: # Replace None by an empty array
                # self.data[var] = np.array([])
                self.data[var] = 0

        try:
            assert len(self.model.dependent_vars) == 1
        except AssertionError:
            raise TypeError('Minimize (currently?) only works with scalar functions.')

        self.constraints = []
        if constraints:
            for constraint in constraints:
                if isinstance(model, Constraint):
                    self.constraints.append(constraint)
                else:
                    self.constraints.append(Constraint(constraint, self.model))


    def error_func(self, p, data):
        """
        The function to be optimized. Scalar valued models are assumed. For Minimize the thing to evaluate is simply
        self.model(*(list(data) + list(p)))

        :param p: array of floats for the parameters.
        :param data: data to be provided to ``Variable``'s.
        """
        # if self.dependent_data:
        #     ans = self.model.numerical_chi_squared(*(list(self.data.values()) + list(p)))
        # else:
        ans, = self.model(*(list(data) + list(p)))
        return ans

    def eval_jacobian(self, p, data):
        """
        Takes partial derivatives of model w.r.t. each ``Parameter``.

        :param p: array of floats for the parameters.
        :param data: data to be provided to ``Variable``'s.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(partial_derivative(*(list(data) + list(p))).flatten())
        # for row in self.partial_jacobian:
        #     for partial_derivative in row:
        #         ans.append(partial_derivative(**{param.name: value for param, value in zip(self.model.params, p)}))
        else:
            return np.array(ans)

    def execute(self, method='SLSQP', *args, **kwargs):
        ans = minimize(
            self.error_func,
            self.initial_guesses,
            method=method,
            args=([value for key, value in self.data.items() if key in self.model.__signature__.parameters],),
            bounds=self.model.bounds,
            constraints=self.scipy_constraints,
            jac=self.eval_jacobian,
            # options={'disp': True},
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
                # 'fun': lambda p, x, c: c.numerical_components[0](*(list(x.values()) + list(p))),
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
    def error_func(self, p, data):
        return - super(Maximize, self).error_func(p, data)

    def eval_jacobian(self, p, data):
        return - super(Maximize, self).eval_jacobian(p, data)

class Likelihood(Maximize):
    """
    Fit using a Maximum-Likelihood approach.
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

    def error_func(self, p, data):
        """
        Error function to be maximised(!) in the case of likelihood fitting.

        :param p: guess params
        :param data: xdata
        :return: scalar value of log-likelihood
        """
        ans = - np.nansum(np.log(self.model(*(list(data) + list(p)))))
        return ans

    def eval_jacobian(self, p, data):
        """
        Jacobian for likelihood is defined as :math:`\\nabla_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param p: guess params
        :param data: data for the variables.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(
                    - np.nansum(
                        partial_derivative(*(list(data) + list(p))).flatten() / self.model(*(list(data) + list(p)))
                    )
                )
        else:
            return np.array(ans)

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

    :param model: Model instance
    :param fit_result: FitResults instance
    :param data: data with which the fit was performed.
    """
    # First filter out the dependent vars
    y_is = [data[var.name] for var in model.dependent_vars if var.name in data]
    x_is = [value for key, value in data.items() if key in model.__signature__.parameters]
    # y_is = [value for key, value in data.items() if key in model.dependent_vars]
    y_bars = [np.mean(x) for x in y_is]
    f_is = model(*x_is, **fit_result.params)
    SS_res = np.sum([np.sum((y_i - f_i)**2) for y_i, f_i in zip(y_is, f_is)])
    SS_tot = np.sum([np.sum((y_i - y_bar)**2) for y_i, y_bar in zip(y_is, y_bars)])

    return 1 - SS_res/SS_tot

# def r_squared(residuals, ydata, sigma=None):
#     """
#     Calculate the squared regression coefficient from the given residuals and data.
#     :param residuals: array of residuals, f(x, p) - y.
#     :param ydata: y in the above equation.
#     :param sigma: sigma in the y_i
#     """
#     ss_err = np.sum(residuals ** 2)
#     if sigma is not None:
#         ss_tot = np.sum(((ydata - ydata.mean())/sigma) ** 2)
#     else:
#         ss_tot = np.sum((ydata - ydata.mean()) ** 2)
#
#     return 1 - (ss_err / ss_tot)