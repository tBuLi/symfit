# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from collections.abc import Sequence
import sys

import sympy
import numpy as np

from symfit.core.argument import Variable
from .support import keywordonly, key2str
from .minimizers import (
    BFGS, SLSQP, LBFGSB, BaseMinimizer, GradientMinimizer, HessianMinimizer,
    ConstrainedMinimizer, MINPACK, ChainedMinimizer, BasinHopping
)
from .objectives import (
    LeastSquares, BaseObjective, MinimizeModel, VectorLeastSquares,
    LogLikelihood, HessianObjectiveJacApprox
)
from .models import BaseModel, Model, BaseNumericalModel, CallableModel

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig

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
        signature = self._make_signature()
        try:
            bound_arguments = signature.bind(*ordered_data, **named_data)
        except TypeError as err:
            for var in self.model.vars:
                if var.name.startswith(Variable._argument_name):
                    raise type(err)(str(err) + '. Some of your Variable\'s are unnamed. That might be the cause of this Error: make sure you use e.g. x = Variable(\'x\')')
                elif isinstance(var, sympy.Derivative):
                    # Include a very strong warning with this error.
                    raise RuntimeWarning(
                        'The model contains derivatives in its definition. '
                        'Are you sure you don\'t mean to use `symfit.ODEModel`?'
                    )
            else:
                raise err
        # Include default values in bound_argument object
        for param in signature.parameters.values():
            if param.name not in bound_arguments.arguments:
                bound_arguments.arguments[param.name] = param.default

        original_data = bound_arguments.arguments   # ordereddict of the data
        self.data = original_data.copy()
        for var in self.model.vars:
            # Identify data by their Variable, not their variable names.
            # But anything that is not a part of model should not be thrown away
            if var.name in self.data:
                self.data[var] = self.data.pop(var.name)

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
        for var, sigma in zip(self.dependent_data, self.sigma_data):
            try:
                iter(self.data[sigma])
            except TypeError:  # not iterable
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

    def _make_signature(self):
        """
        Make a :class:`inspect.Signature` object corresponding to
        ``self.model``.

        :return: :class:`inspect.Signature` object corresponding to
            ``self.model``.
        """
        parameters = self._make_parameters(self.model)
        parameters = sorted(parameters, key=lambda p: p.default is None)
        return inspect_sig.Signature(parameters=parameters)

    @staticmethod
    def _make_parameters(model, none_allowed=None):
        """
        Based on a model, return the inspect.Parameter objects
        needed to satisfy all the variables of this model.

        :param model: instance of model
        :param none_allowed: If provided, this has to be a sequence of
            :class:`symfit.core.argument.Variable` whose values are set to
            ``None`` by default. If not provided, this will be set to sigma
            variables only.
        :return: list of  :class:`inspect.Parameter` corresponding to all the
            external variables of the model.
        """
        if none_allowed is None:
            none_allowed = model.sigmas.values()
        parameters = [
            inspect_sig.Parameter(
                var.name,
                kind=inspect_sig.Parameter.POSITIONAL_OR_KEYWORD,
                default=None if var in none_allowed else inspect_sig.Parameter.empty
            )
            for var in model.vars
        ]
        return parameters

    @property
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var])
                           for var in self.model.dependent_vars if var in self.data)

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
                           for var in self.model.dependent_vars if sigmas[var] in self.data)

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

        return list(set(independent_shapes)), list(set(dependent_shapes))

    @property
    def initial_guesses(self):
        """
        :return: Initial guesses for every parameter.
        """
        return np.array([param.value for param in self.model.params])


class HasCovarianceMatrix(TakesData):
    """
    Mixin class for calculating the covariance matrix for any model that has a
    well-defined Jacobian :math:`J`. The covariance is then approximated as
    :math:`J^T W J`, where W contains the weights of each data point.

    Supports vector valued models, but is unable to estimate covariances for
    those, just variances. Therefore, take the result with a grain of salt for
    vector models.
    """
    def _covariance_matrix(self, best_fit_params, objective):
        # Helper function for self.covariance_matrix.
        try:
            hess = objective.eval_hessian(**key2str(best_fit_params))
        except AttributeError:
            # Some models do not have an eval_hessian, in which case we give up
            return None
        else:
            if hess is None:
                return hess

        try:
            # The squeezing to a matrix is required for MinimizeModel objectives
            hess_inv = np.linalg.inv(np.atleast_2d(np.squeeze(hess)))
        except np.linalg.LinAlgError:
            return None

        if isinstance(objective, LeastSquares):
            # Calculate the covariance for a least squares method.
            # https://www8.cs.umu.se/kurser/5DA001/HT07/lectures/lsq-handouts.pdf
            # Residual sum of squares
            rss = 2 * objective(**key2str(best_fit_params))
            # Degrees of freedom
            raw_dof = np.sum([np.product(shape) for shape in self.data_shapes[1]])
            dof = raw_dof - len(self.model.params)
            if self.absolute_sigma:
                # When interpreting as measurement error, we do not rescale.
                s2 = 1
            else:
                s2 = rss / dof
            cov_mat = s2 * hess_inv
            return cov_mat
        else:
            # The inverse hessian is the covariance matrix for Loglikelihood and
            # also for objectives in general.
            return hess_inv

    def covariance_matrix(self, best_fit_params):
        """
        Given best fit parameters, this function finds the covariance matrix.
        This matrix gives the (co)variance in the parameters.

        :param best_fit_params: ``dict`` of best fit parameters as given by .best_fit_params()
        :return: covariance matrix.
        """
        cov_matrix = self._covariance_matrix(best_fit_params,
                                             objective=self.objective)
        if cov_matrix is None:
            # If the covariance matrix could not be computed we try again by
            # approximating the hessian with the jacobian.

            # VectorLeastSquares should be turned into a LeastSquares for
            # cov matrix calculation
            if self.objective.__class__ is VectorLeastSquares:
                base = LeastSquares
            else:
                base = self.objective.__class__

            class HessApproximation(base, HessianObjectiveJacApprox):
                """
                Class which impersonates ``base``, but which returns zeros
                for the models Hessian. This will effectively result in the
                calculation of the approximate Hessian by calculating
                outer(J.T, J) when calling ``base.eval_hessian``.
                """

            objective = HessApproximation(self.objective.model,
                                          self.objective.data)
            cov_matrix = self._covariance_matrix(best_fit_params,
                                                 objective=objective)

        return cov_matrix


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

    @keywordonly(objective=None, minimizer=None, constraints=None,
                 absolute_sigma=None)
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
        absolute_sigma = named_data.pop('absolute_sigma')
        # Should be a list of Constraint objects
        constraints = [] if constraints is None else constraints

        # Initiate self.model as an instance of BaseModel if it isn't already
        if isinstance(model, BaseModel):
            self.model = model
        else:
            self.model = Model(model)

        self.constraints = self._init_constraints(constraints=constraints,
                                                  model=self.model)

        # Bind as much as possible the provided arguments.
        signature = self._make_signature()
        bound_arguments = signature.bind_partial(*ordered_data, **named_data)

        # Select objective function to use. Has to be done before calling
        # super.__init__
        self.objective = self._determine_objective(
            self.model, objective=objective,
            minimizer=minimizer, bound_arguments=bound_arguments
        )

        super(Fit, self).__init__(self.model, absolute_sigma=absolute_sigma,
                                  **bound_arguments.arguments)

        # Update the data belonging to the constraints. We do this by checking
        # for the presence of data with the same name as one of the independent
        # variables of the constraint. If present, we start addressing them by
        # their Variable instead.
        for constraint in self.constraints:
            for var in constraint.vars:
                if var.name in self.data:
                    self.data[var] = self.data.pop(var.name)

        # Initialise the objective with data if it's not initialised already
        if not isinstance(self.objective, BaseObjective):
            self.objective = self.objective(self.model, self.data)

        # Select the minimizer on the basis of the provided information.
        if minimizer is None:
            minimizer = self._determine_minimizer()

        # Initialise the minimizer
        if isinstance(minimizer, Sequence):
            minimizers = [self._init_minimizer(mini) for mini in minimizer]
            self.minimizer = self._init_minimizer(ChainedMinimizer, minimizers=minimizers)
        else:
            self.minimizer = self._init_minimizer(minimizer)

    def _make_signature(self):
        parameters = self._make_parameters(self.model)
        # Extend the signature with the variables to the constraint. Since
        # constraints will be turned into MinimizeModel objectives, they only
        # need independent variables to be provided.
        for constraint in self.constraints:
            none_allowed = constraint.dependent_vars + list(constraint.sigmas.values())
            parameters.extend(
                self._make_parameters(
                    constraint, none_allowed=none_allowed
                )
            )

        # Make unique while preserving order, and sort by default value so
        # sigma variables end last
        unique_parameters = []
        for par in parameters:
            if par not in unique_parameters:
                unique_parameters.append(par)

        parameters = sorted(unique_parameters, key=lambda p: p.default is None)
        return inspect_sig.Signature(parameters=parameters)

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

    @staticmethod
    def _determine_objective(model, objective, minimizer, bound_arguments):
        """
        Determine the most suitable objective on the basis of the problem at
        hand. This could modify ``bound_arguments`` in place accordingly if
        required!

        :param model: :class:`symfit.core.models.BaseModel` under consideration.
        :param objective: objective provided to :class:`symfit.core.fit.Fit` by
            the user, or ``None``.
        :param minimizer: :class:`~symfit.core.minimizers.BaseMinimizer`
            provided by the user, or  ``None``
        :param bound_arguments: Instance of :class:`inspect.BoundArguments`.
        :return: a subclass of `BaseObjective`.
        """
        if objective is None:
            if minimizer is MINPACK:
                # MINPACK is considered a special snowflake, as its API has to
                # be considered separately and has its own non standard
                # objective function.
                objective = VectorLeastSquares
            elif (len(model) == 1 and len(model.independent_vars) == 0 and
                    model.dependent_vars[0].name not in bound_arguments.arguments):
                objective = MinimizeModel
            else:
                objective = LeastSquares

        # Check if the data is compatible with the objective
        if (objective is LogLikelihood or objective is MinimizeModel or
                isinstance(objective, (MinimizeModel, LogLikelihood))):
            # Set dependent vars and corresponding sigmas to None.
            for var in model.dependent_vars + list(model.sigmas.values()):
                if var.name not in bound_arguments.arguments:
                    bound_arguments.arguments[var.name] = None
                else:
                    raise TypeError(
                        'A value was provided for `{}`, however for {} '
                        'fitting the dependent variable cannot have a value '
                        'assigned to it.'.format(var.name, objective)
                    )
        return objective

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
            # Hence the check of jacobian_model, as this is the
            # py function version of the analytical jacobian.
            if hasattr(self.model, 'eval_jacobian') and hasattr(self.objective, 'eval_jacobian'):
                minimizer_options['jacobian'] = self.objective.eval_jacobian
        if issubclass(minimizer, HessianMinimizer):
            # If an analytical version of the Hessian exists we should use
            # that, otherwise we let the minimizer estimate it itself.
            # Hence the check of hessian_model, as this is the
            # py function version of the analytical hessian.
            if hasattr(self.model, 'eval_hessian') and hasattr(self.objective, 'eval_hessian'):
                minimizer_options['hessian'] = self.objective.eval_hessian

        if issubclass(minimizer, ConstrainedMinimizer):
            # set the constraints as MinimizeModel. The dependent vars of the
            # constraint are set to None since their value is irrelevant.
            constraint_objectives = []
            for constraint in self.constraints:
                data = self.data  # No copy, share state
                constraint_objectives.append(MinimizeModel(constraint, data))
            minimizer_options['constraints'] = constraint_objectives
        return minimizer(self.objective, self.model.params, **minimizer_options)

    def _init_constraints(self, constraints, model):
        """
        Takes the user provided constraints and converts them to a list of
        ``type(model)`` objects, which are extended to also have the
         parameters of ``model``.

        :param constraints: iterable of :class:`~sympy.core.relational.Relation`
            objects.
        :return: list of :class:`~symfit.core.models.BaseModel` objects. The
            exact type will depend on the type of ``model``.
        """
        con_models = []
        for constraint in constraints:
            if hasattr(constraint, 'constraint_type'):
                con_models.append(constraint)
            else:
                if isinstance(model, BaseNumericalModel):
                    # Numerical models need to be provided with a connectivity
                    # mapping, so we cannot use the type of model. Instead,
                    # use the bare minimum for an analytical model for the
                    # constraint. ToDo: once GradientNumericalModel etc are
                    # introduced, pick the corresponding analytical model for
                    # the constraint.
                    con_models.append(
                        CallableModel.as_constraint(constraint, model)
                    )
                else:
                    con_models.append(
                        model.__class__.as_constraint(constraint, model)
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
        minimizer_ans.covariance_matrix = self.covariance_matrix(
            dict(zip(self.model.params, minimizer_ans._popt))
        )
        # Overwrite the DummyModel with the current model
        minimizer_ans.model = self.model
        minimizer_ans.minimizer = self.minimizer
        return minimizer_ans
