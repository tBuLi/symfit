import abc
from collections import OrderedDict
from six import add_metaclass

import numpy as np

from .support import cache, keywordonly, key2str

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

    @property
    @cache
    def dependent_data(self):
        """
        Read-only Property

        :return: Data belonging to each dependent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict(
            (var.name, self.data[var.name]) for var in self.model)

    @property
    @cache
    def independent_data(self):
        """
        Read-only Property

        :return: Data belonging to each independent variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        return OrderedDict((var.name, self.data[var.name]) for var in
                           self.model.independent_vars)

    @property
    @cache
    def sigma_data(self):
        """
        Read-only Property

        :return: Data belonging to each sigma variable as a dict with
                 variable names as key, data as value.
        :rtype: collections.OrderedDict
        """
        sigmas = self.model.sigmas
        return OrderedDict(
            (sigmas[var].name, self.data[sigmas[var].name]) for var in
            self.model)

    @abc.abstractmethod
    def __call__(self, **parameters):
        """
        Evaluate the objective function for given parameter values.

        :param parameters:
        :return: float
        """
        pass

@add_metaclass(abc.ABCMeta)
class GradientObjective(BaseObjective):
    """
    ABC for objectives that support gradient methods.
    """
    @abc.abstractmethod
    def eval_jacobian(self, **parameters):
        """
        Evaluate the jacobian for given parameter values.

        :param parameters:
        :return: float
        """
        pass

class VectorLeastSquares(GradientObjective):
    """
    Implemented for MINPACK only. Returns the residuals/sigma before squaring
    and summing, rather then chi2 itself.
    """
    @keywordonly(flatten_components=True)
    def __call__(self, **parameters):
        """
        Returns the value of the square root of :math:`\\chi^2`, summing over the components.

        This function now supports setting variables to None.

        :param p: array of parameter values.
        :param flatten_components: If True, summing is performed over the data indices (default).
        :return: :math:`\\sqrt(\\chi^2)`
        """
        flatten_components = parameters.pop('flatten_components')
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)
        evaluated_func = self.model(**jac_kwargs)
        result = []

        # zip together the dependent vars and evaluated component
        for y, ans in zip(self.model, evaluated_func):
            if self.dependent_data[y.name] is not None:
                result.append(((self.dependent_data[y.name] - ans) / self.sigma_data[self.model.sigmas[y].name]) ** 2)
                if flatten_components: # Flattens *within* a component
                    result[-1] = result[-1].flatten()
        return np.sqrt(sum(result))

    def eval_jacobian(self, **parameters):
        chi = self(flatten=False, **parameters)
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)
        evaluated_func = self.model(**jac_kwargs)

        result = len(self.model.params) * [0.0]
        for ans, y, row in zip(evaluated_func, self.model, self.model.numerical_jacobian):
            if self.dependent_data[y.name] is not None:
                for index, component in enumerate(row):
                    result[index] += component(**jac_kwargs) * (
                        (self.dependent_data[y.name] - ans) / self.sigma_data[self.model.sigmas[y].name] ** 2
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
    def __call__(self, **parameters):
        """

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\chi^2` at.
        :param flatten_components: if `True`, return the total :math:`\chi^2`. If
            `False`, return the :math:`\chi^2` per component of the
            :class:`~symfit.core.fit.BaseModel`.
        :return: scalar or list of scalars depending on the value of `flatten_components`.
        """
        flatten_components = parameters.pop('flatten_components')
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)
        evaluated_func = self.model(**jac_kwargs)

        chi2 = [0 for _ in evaluated_func]
        for index, (dep_var_name, dep_var_value) in enumerate(evaluated_func._asdict().items()):
            dep_data = self.dependent_data[dep_var_name]
            if dep_data is not None:
                sigma = self.sigma_data['sigma_{}'.format(dep_var_name)]  # Should be changed with #41
                chi2[index] += np.sum(
                    (dep_var_value - dep_data) ** 2 / sigma ** 2)
                # chi2 += np.sum((dep_var_value - dep_data)**2/sigma**2)
        chi2 = np.sum(chi2) if flatten_components else chi2
        return chi2

    def eval_jacobian(self, **parameters):
        """
        Jacobian of :math:`\\chi^2` in the
        :class:`~symfit.core.argument.Parameter`'s (:math:`\\nabla_\\vec{p} \\chi^2`).

        :param parameters: values of the
            :class:`~symfit.core.argument.Parameter`'s to evaluate :math:`\\nabla_\\vec{p} \\chi^2` at.
        :return: `np.array` of length equal to the number of parameters..
        """
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)
        evaluated_func = self.model(**jac_kwargs)
        result = [0.0 for _ in self.model.params]

        for ans, var, row in zip(evaluated_func, self.model,
                               self.model.numerical_jacobian):
            dep_data = self.dependent_data[var.name]
            sigma_var = self.model.sigmas[var]
            if dep_data is not None:
                sigma = self.sigma_data[sigma_var.name]  # Should be changed with #41
                for index, component in enumerate(row):
                    result[index] += np.sum(
                        component(**jac_kwargs) * ((dep_data - ans) / sigma ** 2)
                    )
        return - np.array(result).T


class LogLikelihood(GradientObjective):
    """
    Error function to be minimized by a minimizer in order to *maximize*
    the log-likelihood.
    """
    def __call__(self, **parameters):
        """
        :param parameters: values for the fit parameters.
        :return: scalar value of log-likelihood
        """
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)

        ans = - np.nansum(np.log(self.model(**jac_kwargs)))
        return ans

    @keywordonly(apply_func=np.nansum)
    def eval_jacobian(self, **parameters):
        """
        Jacobian for log-likelihood is defined as :math:`\\nabla_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param parameters: values for the fit parameters.
        :param apply_func: Function to apply to each component before returning it.
            The default is to sum away along the datapoint dimension using `np.nansum`.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        apply_func = parameters.pop('apply_func')
        jac_kwargs = key2str(parameters)
        jac_kwargs.update(self.independent_data)

        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(
                    - apply_func(
                        partial_derivative(**jac_kwargs).flatten() / self.model(**jac_kwargs)
                    )
                )
        else:
            return np.array(ans)


class MinimizeModel(BaseObjective):
    """
    Objective to use when the model itself is the quantity that should be
    minimized. This is only supported for scalar models.
    """
    def __init__(self, model, *args, **kwargs):
        if len(model) > 1:
            raise TypeError('Only scalar functions are supported by {}'.format(self.__class__))
        super(MinimizeModel, self).__init__(model, *args, **kwargs)

    def __call__(self, **parameters):
        return self.model(**parameters)[0]

    def eval_jacobian(self, **parameters):
        if hasattr(self.model, 'numerical_jacobian'):
            ans = []
            for partial_derivative in self.model.numerical_jacobian[0]:
                ans.append(partial_derivative(**parameters))
            return np.array(ans)
        else:
            return None