"""
Objective functions are the functions which are minimized by the
:mod:`~symfit.core.minimizers`.
Famous examples are `least squares
<https://en.wikipedia.org/wiki/Least_squares>`_, `log-likelihood
<https://en.wikipedia.org/wiki/Likelihood_function>`_, or minimizing the model
itself.

``symfit`` provides objective functions for those cases by default. Custom
objectives can also be created, for example by inheriting from
:class:`~symfit.core.objectives.BaseObjective`,
:class:`~symfit.core.objectives.GradientObjective` or
:class:`~symfit.core.objectives.HessianObjective`.
"""

import abc
from collections import OrderedDict
from six import add_metaclass

import numpy as np

from .support import cached_property, keywordonly, key2str

@add_metaclass(abc.ABCMeta)
class BaseObjective(object):
    """
    ABC for objective functions. Implements basic data handling.
    """
    def __init__(self, model, data):
        """
        :param model: `symfit` style model.
        :param data: data for all the variables of the model.
        """
        self.model = model
        self.data = data
        # Compares the model with the data to see if they are compatible.
        self._sanity_checking()

    @cached_property
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var])
                           for var in self.model.dependent_vars)

    @cached_property
    def independent_data(self):
        """
        Read-only Property

        :return: Data belonging to each independent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var]) for var in
                           self.model.independent_vars)

    @cached_property
    def sigma_data(self):
        """
        Read-only Property

        :return: Data belonging to each sigma variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        sigmas = self.model.sigmas
        return OrderedDict(
            (sigmas[var], self.data[sigmas[var]]) for var in
            self.model.dependent_vars)

    @abc.abstractmethod
    def __call__(self, ordered_parameters=[], **parameters):
        """
        Evaluate the objective function for given parameter values.

        :param ordered_parameters: List of parameter, in alphabetical order.
            Typically provided by the minimizer.
        :param parameters: parameters as keyword arguments.
        :return: evaluated model.
        """
        # zip will stop when the shortest of the two is exhausted
        parameters.update(dict(zip(self.model.free_params, ordered_parameters)))
        parameters.update(self._invariant_kwargs)
        result = self.model(**key2str(parameters))._asdict()
        # Return only the components corresponding to the dependent data.
        return self._shape_of_dependent_data(
            [comp for var, comp in result.items()
             if var in self.model.dependent_vars]
        )

    def _shape_of_dependent_data(self, model_output, param_level=0):
        """
        In rare cases, the dependent data and the output of the model do not
        have the same shape. Think for example about :math:`y_i = a`. This
        function makes sure both sides of the equation have the same shape.

        :param model_output: Output of a call to model
        :param param_level: indicates how many parameter dimensions should be
            added to the shape. 0 for __call__, 1 for jac, 2 for hess.
        :return: ``model_output`` reshaped to ``dependent_data``'s shape, if
            possible.
        """
        shaped_result = []
        n_params = len(self.model.params)
        for dep_var, component in zip(self.model.dependent_vars, model_output):
            dep_data = self.dependent_data.get(dep_var, None)
            if dep_data is not None:
                if dep_data.shape == component.shape:
                    shaped_result.append(component)
                else:
                    # Add extra dimensions to the component if needed.
                    dim_diff = len(dep_data.shape) - len(component.shape[param_level:])
                    for _ in range(dim_diff):
                        component = np.expand_dims(component, -1)
                    # Let numpy deal with all the broadcasting
                    shape = param_level * [n_params] + list(dep_data.shape)
                    shaped_result.append(np.broadcast_to(component, shape))
            else:
                shaped_result.append(component)
        return shaped_result

    @cached_property
    def _invariant_kwargs(self):
        """
        Prepares the invariant kwargs to ``self.model`` which are not provided
        by the minimizers, and are the same for every iteration of the
        minimization. This means fixed parameters and data, matching the
        signature of ``self.model``.
        """
        kwargs = {p: p.value for p in self.model.params
                  if p not in self.model.free_params}
        data_by_name = key2str(self.independent_data)
        kwargs.update(
            {p: data_by_name[p] for p in
            self.model.__signature__.parameters if p in data_by_name}
        )
        return kwargs

    def __eq__(self, other):
        """
        Objectives are considered equal if they are of the same type, have the
        same model, and the same data.
        """
        # Class equality is enforced, even though this breaks subclassing.
        # This is to prevent false positives, which could be way worse. In the
        # case of subclassing, we leave it up to the subclasser to decide when
        # equality is achieved.
        if self.__class__ != other.__class__ or self.model != other.model:
            return False

        # Check if the data is also equivalent
        for key, value in self.data.items():
            try:
                equal = np.allclose(other.data[key], value)
            except TypeError:
                equal = other.data[key] == value
            finally:
                if not equal:
                    return False
        return True

    def _sanity_checking(self):
        """
        Check if the model and the provided data are compatible. Raises a
        TypeError when this is not the case.
        """
        # Simply checking for existence of these dicts will raise an error if
        # they cannot be built (because these are properties).
        self.dependent_data
        self.independent_data
        self.sigma_data


@add_metaclass(abc.ABCMeta)
class GradientObjective(BaseObjective):
    """
    ABC for objectives that support gradient methods.
    """
    @abc.abstractmethod
    def eval_jacobian(self, ordered_parameters=[], **parameters):
        """
        Evaluate the jacobian for given parameter values.

        :param ordered_parameters: List of parameter, in alphabetical order.
            Typically provided by the minimizer.
        :param parameters: parameters as keyword arguments.
        :return: evaluated jacobian
        """
        parameters.update(dict(zip(self.model.free_params, ordered_parameters)))
        parameters.update(self._invariant_kwargs)
        result = self.model.eval_jacobian(**key2str(parameters))._asdict()
        # Return only the components corresponding to the dependent data.
        return self._shape_of_dependent_data(
            [comp for var, comp in result.items()
             if var in self.model.dependent_vars],
            param_level=1
        )


@add_metaclass(abc.ABCMeta)
class HessianObjective(GradientObjective):
    """
    ABC for objectives that support hessian methods.
    """
    @abc.abstractmethod
    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        Evaluate the hessian for given parameter values.

        :param ordered_parameters: List of parameter, in alphabetical order.
            Typically provided by the minimizer.
        :param parameters: parameters as keyword arguments.
        :return: evaluated hessian
        """
        parameters.update(dict(zip(self.model.free_params, ordered_parameters)))
        parameters.update(self._invariant_kwargs)
        result = self.model.eval_hessian(**key2str(parameters))._asdict()
        # Return only the components corresponding to the dependent data.
        return self._shape_of_dependent_data(
            [comp for var, comp in result.items()
             if var in self.model.dependent_vars],
            param_level=2
        )


class VectorLeastSquares(GradientObjective):
    """
    Implemented for MINPACK only. Returns the residuals/sigma before squaring
    and summing, rather then chi2 itself.
    """
    @keywordonly(flatten_components=True)
    def __call__(self, ordered_parameters=[], **parameters):
        """
        Returns the value of the square root of :math:`\\chi^2`, summing over the components.

        This function now supports setting variables to None.

        :param flatten_components: If True, summing is performed over the data indices (default).
        :return: :math:`\\sqrt(\\chi^2)`
        """
        flatten_components = parameters.pop('flatten_components')
        evaluated_func = super(VectorLeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        result = []

        # zip together the dependent vars and evaluated component
        for y, ans in zip(self.model.dependent_vars, evaluated_func):
            dep_data = self.dependent_data.get(y, None)
            if dep_data is not None:
                result.append(((self.dependent_data[y] - ans) / self.sigma_data[self.model.sigmas[y]]) ** 2)
                if flatten_components: # Flattens *within* a component
                    result[-1] = result[-1].flatten()
        return np.sqrt(sum(result))

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        chi = self(ordered_parameters, flatten_components=False, **parameters)
        evaluated_func = super(VectorLeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(VectorLeastSquares, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = len(self.model.params) * [0.0]
        for ans, y, row in zip(evaluated_func, self.model.dependent_vars,
                               evaluated_jac):
            dep_data = self.dependent_data.get(y, None)
            if dep_data is not None:
                for index, component in enumerate(row):
                    result[index] += component * (
                        (self.dependent_data[y] - ans) / self.sigma_data[self.model.sigmas[y]] ** 2
                    )
        result *= (1 / chi)
        result = np.nan_to_num(result)
        result = [item.flatten() for item in result]
        return - np.array(result).T


class LeastSquares(HessianObjective):
    """
    Objective representing the least-squares deviation of a model, defined as
    :math:`S = \\frac{1}{2} \\sum_{i} \\sum_{x_i} \\frac{r_i(x_i, \\vec{p})^2}{\\sigma_i(x_i)^2}`,
    where :math:`i` ranges over all components of the model,
    :math:`r_i(x_i, \\vec{p})` is the residue of the :math:`i`-th component,
    :math:`x_i` indicates all the data associated with the :math:`i`-th
    component, and :math:`\\sigma_i(x_i)` indicates the associated standard deviations.

    The data for each component does not have to be the same, and it does not
    have to have the same shape. The only thing that matters is that within each
    component the shapes have to be compatible.
    """
    @keywordonly(flatten_components=True)
    def __call__(self, ordered_parameters=[], **parameters):
        """
        :param ordered_parameters: See ``parameters``.
        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`S` at.
        :param flatten_components: if `True`, return the total :math:`S`. If
            `False`, return the :math:`S` per component of the
            :class:`~symfit.core.models.BaseModel`.
        :return: scalar or list of scalars depending on the value of `flatten_components`.
        """
        flatten_components = parameters.pop('flatten_components')
        evaluated_func = super(LeastSquares, self).__call__(
            ordered_parameters, **parameters
        )

        chi2 = [0 for _ in evaluated_func]
        for index, (dep_var, dep_var_value) in enumerate(zip(self.model.dependent_vars, evaluated_func)):
            dep_data = self.dependent_data.get(dep_var, None)
            if dep_data is not None:
                sigma = self.sigma_data[self.model.sigmas[dep_var]]
                chi2[index] += np.sum(
                    (dep_var_value - dep_data) ** 2 / sigma ** 2
                )
        chi2 = np.sum(chi2) if flatten_components else chi2
        return chi2 / 2

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        """
        Jacobian of :math:`S` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p} S`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} S` at.
        :return: ``np.array`` of length equal to the number of parameters..
        """
        evaluated_func = super(LeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LeastSquares, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = 0
        for var, f, jac_comp in zip(self.model.dependent_vars, evaluated_func,
                                    evaluated_jac):
            y = self.dependent_data.get(var, None)
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                pre_sum = jac_comp * ((y - f) / sigma**2)[np.newaxis, ...]
                axes = tuple(range(1, len(pre_sum.shape)))
                result -= np.sum(pre_sum, axis=axes, keepdims=False)
        return np.atleast_1d(np.squeeze(np.array(result)))

    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        Hessian of :math:`S` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p}^2 S`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} S` at.
        :return: ``np.array`` of length equal to the number of parameters..
        """
        evaluated_func = super(LeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LeastSquares, self).eval_jacobian(
            ordered_parameters, **parameters
        )
        evaluated_hess = super(LeastSquares, self).eval_hessian(
            ordered_parameters, **parameters
        )

        result = 0
        for var, f, jac_comp, hess_comp in zip(self.model.dependent_vars,
                                               evaluated_func, evaluated_jac,
                                               evaluated_hess):
            y = self.dependent_data.get(var, None)
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                p1 = hess_comp * ((y - f) / sigma**2)[np.newaxis, np.newaxis, ...]
                # Outer product
                p2 = np.einsum('i...,j...->ij...', jac_comp, jac_comp)
                p2 = p2 / sigma[np.newaxis, np.newaxis, ...]**2
                # We sum away everything except the matrices in the axes 0 & 1.
                axes = tuple(range(2, len(p2.shape)))
                result += np.sum(p2 - p1, axis=axes, keepdims=False)
        return np.atleast_2d(np.squeeze(np.array(result)))


class HessianObjectiveJacApprox(HessianObjective):
    """
    This object should only be used as a Mixin for covariance matrix estimation.
    Since the covariance matrix for the least-squares method is proportional to
    the Hessian of :math:`S`, this function attempts to return the Hessian
    upon calculating ``eval_hessian``.

    However, if the model does not have a Hessian defined through
    ``eval_hessian``, we approximate the Hessian as :math:`J^{T}\cdot J`,
    where :math:`J` is the Jacobian of the model. This approximation is valid
    when, amongst other things, the residuals are sufficiently small. It can
    therefore only be used after fitting, not during.

    An objective which inherits from this object, will return zeros with the
    shape of the hessian of the model, when ``eval_hessian`` is called. This
    code injection will therefore result in the terms proportional to the
    hessian of the model dropping out, which leaves the famous
    :math:`J^{T}\cdot J` approximation.
    """
    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        :return: Zeros with the shape of the Hessian of the model.
        """
        result = super(HessianObjectiveJacApprox, self).__call__(
            ordered_parameters, **parameters
        )
        num_params = len(self.model.params)
        return [np.broadcast_to(
                    np.zeros_like(comp),
                    (num_params, num_params) + comp.shape
                ) for comp in result]


class BaseIndependentObjective(BaseObjective):
    """
    Some objective functions dependent only on independent variables, not
    dependent and sigma variables. In this case, sanity checking is greatly
    simplified.
    """
    @cached_property
    def dependent_data(self):
        """
        :return: Empty OrderedDict.
        :rtype: collections.OrderedDict
        """
        return OrderedDict()

    @cached_property
    def sigma_data(self):
        """
        :return: Empty OrderedDict.
        :rtype: collections.OrderedDict
        """
        return OrderedDict()


class LogLikelihood(HessianObjective, BaseIndependentObjective):
    """
    Error function to be minimized by a minimizer in order to *maximize*
    the log-likelihood.
    """
    def __call__(self, ordered_parameters=[], **parameters):
        """
        :param parameters: values for the fit parameters.
        :return: scalar value of log-likelihood
        """
        evaluated_func = super(LogLikelihood, self).__call__(
            ordered_parameters, **parameters
        )

        ans = - np.nansum(
            [np.nansum(np.log(component)) for component in evaluated_func]
        )
        return ans

    @keywordonly(apply_func=np.nansum)
    def eval_jacobian(self, ordered_parameters=[], **parameters):
        """
        Jacobian for log-likelihood is defined as :math:`\\nabla_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param parameters: values for the fit parameters.
        :param apply_func: Function to apply to each component before returning it.
            The default is to sum away along the datapoint dimension using `np.nansum`.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        apply_func = parameters.pop('apply_func')
        evaluated_func = super(LogLikelihood, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LogLikelihood, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = []
        for component, jac_comp in zip(evaluated_func, evaluated_jac):
            component_sums = []
            for df in jac_comp:
                component_sums.append(
                    - apply_func(
                        df / component
                    )
                )
            result.append(component_sums)
        result = np.sum(result, axis=0)
        return np.atleast_1d(np.squeeze(np.array(result)))

    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        Hessian for log-likelihood is defined as
        :math:`\\nabla^2_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param parameters: values for the fit parameters.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        evaluated_func = super(LogLikelihood, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LogLikelihood, self).eval_jacobian(
            ordered_parameters, **parameters
        )
        evaluated_hess = super(LogLikelihood, self).eval_hessian(
            ordered_parameters, **parameters
        )

        result = 0
        for f, jac_comp, hess_comp in zip(evaluated_func, evaluated_jac, evaluated_hess):
            # Outer product
            jac_outer_jac = np.einsum('i...,j...->ij...', jac_comp, jac_comp)
            dd_logf = - hess_comp / f[np.newaxis, np.newaxis, ...] + \
                      (1 / f**2)[np.newaxis, np.newaxis, ...] * jac_outer_jac
            # We sum away everything except the matrices in the axes 0 & 1.
            axes = tuple(range(2, len(dd_logf.shape)))
            result += np.sum(dd_logf, axis=axes, keepdims=False)

        return np.atleast_2d(np.squeeze(np.array(result)))


class MinimizeModel(HessianObjective, BaseIndependentObjective):
    """
    Objective to use when the model itself is the quantity that should be
    minimized. This is only supported for scalar models.
    """
    def __init__(self, model, *args, **kwargs):
        if len(model.dependent_vars) > 1:
            raise TypeError('Only scalar functions are supported by {}'.format(self.__class__))
        super(MinimizeModel, self).__init__(model, *args, **kwargs)

    def __call__(self, ordered_parameters=[], **parameters):
        evaluated_func = super(MinimizeModel, self).__call__(
            ordered_parameters, **parameters
        )
        return evaluated_func[0]

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        if hasattr(self.model, 'eval_jacobian'):
            evaluated_jac = super(MinimizeModel, self).eval_jacobian(
                ordered_parameters, **parameters
            )
            return np.array(evaluated_jac[0])
        else:
            return None

    def eval_hessian(self, ordered_parameters=[], **parameters):
        if hasattr(self.model, 'eval_hessian'):
            evaluated_hess = super(MinimizeModel, self).eval_hessian(
                ordered_parameters, **parameters
            )
            return np.array(evaluated_hess[0])
        else:
            return None