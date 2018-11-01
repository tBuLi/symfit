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
        return self.model(**key2str(parameters))

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
        return self.model.eval_jacobian(**key2str(parameters))


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


class LeastSquares(GradientObjective):
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
        result = [0.0 for _ in self.model.params]

        for ans, var, row in zip(evaluated_func, self.model, evaluated_jac):
            dep_data = self.dependent_data[var]
            sigma_var = self.model.sigmas[var]
            if dep_data is not None:
                sigma = self.sigma_data[sigma_var]  # Should be changed with #41
                for index, component in enumerate(row):
                    result[index] += np.sum(
                        component * ((dep_data - ans) / sigma ** 2)
                    )
        return - np.array(result).T


class LogLikelihood(GradientObjective):
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

        ans = []
        for row in evaluated_jac:
            for partial_derivative in row:
                ans.append(
                    - apply_func(
                        partial_derivative.flatten() / evaluated_func
                    )
                )
        else:
            return np.array(ans)


class MinimizeModel(GradientObjective):
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