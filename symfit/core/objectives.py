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

    @cached_property
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var, self.data[var]) for var in self.model)

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
            self.model)

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
        parameters.update(self.invariant_kwargs)
        result = self.model(**key2str(parameters))
        return self._shape_dependent_data(result)

    def _shape_dependent_data(self, model_output, level=0):
        """
        In rare cases, the dependent data and the output of the model do not
        have the same shape. Think for example about :math:`y_i = a`. This
        function makes sure both sides of the equation have the same shape.

        :param model_output: Output of a call to model
        :return: ``model_output`` reshaped to ``dependent_data``'s shape, if
            possible.
        """
        shaped_result = []
        for dep_data, component in zip(self.dependent_data.values(), model_output):
            if dep_data is not None:
                if dep_data.shape == component.shape:
                    shaped_result.append(component)
                else:
                    for _ in range(level):
                        dep_data = np.expand_dims(dep_data, 0)
                    shaped_result.append(
                        np.broadcast_arrays(dep_data, component)[1]
                    )
            else:
                shaped_result.append(component)
        return shaped_result

    @cached_property
    def invariant_kwargs(self):
        """
        Prepares the invariant kwargs to ``self.model`` which are not provided
        by the minimizers, and are the same for every iteration of the
        minimization. This means fixed parameters and data, matching the
        signature of ``self.model``.
        """
        kwargs = {p: p.value for p in self.model.params
                  if p not in self.model.free_params}
        data_by_name = key2str(self.data)
        kwargs.update(
            {p: data_by_name[p] for p in
            self.model.__signature__.parameters if p in data_by_name}
        )
        return kwargs

    def _clear_cache(self):
        """
        Clear cached properties. Should preferably be called after every
        minimization precedure to clean the pallette.
        """
        del self.invariant_kwargs


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
        parameters.update(self.invariant_kwargs)
        result = self.model.eval_jacobian(**key2str(parameters))
        return self._shape_dependent_data(result, level=1)


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
        parameters.update(self.invariant_kwargs)
        result = self.model.eval_hessian(**key2str(parameters))
        return self._shape_dependent_data(result, level=2)


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
        for y, ans in zip(self.model, evaluated_func):
            if self.dependent_data[y] is not None:
                result.append(((self.dependent_data[y] - ans) / self.sigma_data[self.model.sigmas[y]]) ** 2)
                if flatten_components: # Flattens *within* a component
                    result[-1] = result[-1].flatten()
        return np.sqrt(sum(result))

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        chi = self(ordered_parameters, flatten=False, **parameters)
        evaluated_func = super(VectorLeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(VectorLeastSquares, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = len(self.model.params) * [0.0]
        for ans, y, row in zip(evaluated_func, self.model, evaluated_jac):
            if self.dependent_data[y] is not None:
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
    Objective representing the :math:`\chi^2` of a model.
    """
    @keywordonly(flatten_components=True)
    def __call__(self, ordered_parameters=[], **parameters):
        """
        :param ordered_parameters: See ``parameters``.
        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\chi^2` at.
        :param flatten_components: if `True`, return the total :math:`\chi^2`. If
            `False`, return the :math:`\chi^2` per component of the
            :class:`~symfit.core.fit.BaseModel`.
        :return: scalar or list of scalars depending on the value of `flatten_components`.
        """
        flatten_components = parameters.pop('flatten_components')
        evaluated_func = super(LeastSquares, self).__call__(
            ordered_parameters, **parameters
        )

        chi2 = [0 for _ in evaluated_func]
        for index, (dep_var, dep_var_value) in enumerate(zip(self.model, evaluated_func)):
            dep_data = self.dependent_data[dep_var]
            if dep_data is not None:
                sigma = self.sigma_data[self.model.sigmas[dep_var]]
                chi2[index] += np.sum(
                    (dep_var_value - dep_data) ** 2 / sigma ** 2
                )
        chi2 = np.sum(chi2) if flatten_components else chi2
        return chi2

    def eval_jacobian(self, ordered_parameters=[], **parameters):
        """
        Jacobian of :math:`\\chi^2` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p} \\chi^2`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} \\chi^2` at.
        :return: `np.array` of length equal to the number of parameters..
        """
        evaluated_func = super(LeastSquares, self).__call__(
            ordered_parameters, **parameters
        )
        evaluated_jac = super(LeastSquares, self).eval_jacobian(
            ordered_parameters, **parameters
        )

        result = 0
        for var, f, jac_comp in zip(self.model, evaluated_func, evaluated_jac):
            y = self.dependent_data[var]
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                pre_sum = jac_comp * ((y - f) / sigma**2)[np.newaxis, ...]
                axes = tuple(range(1, len(pre_sum.shape)))
                result -= 2 * np.sum(pre_sum, axis=axes, keepdims=False)
        return np.atleast_1d(np.squeeze(np.array(result)))

    def eval_hessian(self, ordered_parameters=[], **parameters):
        """
        Hessian of :math:`\\chi^2` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p}^2 \\chi^2`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} \\chi^2` at.
        :return: `np.array` of length equal to the number of parameters..
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
        for var, f, jac_comp, hess_comp in zip(self.model, evaluated_func,
                                               evaluated_jac, evaluated_hess):
            y = self.dependent_data[var]
            sigma_var = self.model.sigmas[var]
            if y is not None:
                sigma = self.sigma_data[sigma_var]
                p1 = hess_comp * ((y - f) / sigma**2)[np.newaxis, np.newaxis, ...]
                # Outer product
                p2 = np.einsum('i...,j...->ij...', jac_comp, jac_comp)
                p2 / sigma[np.newaxis, np.newaxis, ...]**2
                # We sum away everything except the matrices in the axes 0 & 1.
                axes = tuple(range(2, len(p2.shape)))
                result += 2 * np.sum(p2 - p1, axis=axes, keepdims=False)
        return np.atleast_2d(np.squeeze(np.array(result)))


class LogLikelihood(HessianObjective):
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

        ans = - np.nansum(np.log(evaluated_func))
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
        for jac_comp in evaluated_jac:
            for df in jac_comp:
                result.append(
                    - apply_func(
                        df.flatten() / evaluated_func
                    )
                )
        else:
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
        else:
            return np.atleast_2d(np.squeeze(np.array(result)))


class MinimizeModel(HessianObjective):
    """
    Objective to use when the model itself is the quantity that should be
    minimized. This is only supported for scalar models.
    """
    def __init__(self, model, *args, **kwargs):
        if len(model) > 1:
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