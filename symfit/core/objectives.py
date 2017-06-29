import abc
from collections import OrderedDict
from six import add_metaclass

import numpy as np

from .support import cache, keywordonly

@add_metaclass(abc.ABCMeta)
class BaseObjective:
    def __init__(self, model, data):
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
        jac_kwargs = dict(**self.independent_data, **parameters)
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
        # jac_args = list(independent_data.values()) + list(p)
        evaluated_func = self.model(**self.independent_data, **parameters)
        result = [0.0 for _ in self.model.params]

    def eval_jacobian(self, **parameters):
        chi = self(flatten=False, **parameters)
        jac_kwargs = dict(**self.independent_data, **parameters)
        evaluated_func = self.model(**self.independent_data, **parameters)

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
    @keywordonly(flatten_components=True)
    def __call__(self, **parameters):
        flatten_components = parameters.pop('flatten_components')
        jac_kwargs = dict(**self.independent_data, **parameters)
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
        # jac_args = list(independent_data.values()) + list(p)
        evaluated_func = self.model(**self.independent_data, **parameters)
        result = [0.0 for _ in self.model.params]

        for ans, var, row in zip(evaluated_func, self.model,
                               self.model.numerical_jacobian):
            dep_data = self.dependent_data[var.name]
            sigma_var = self.model.sigmas[var]
            if dep_data is not None:
                sigma = self.sigma_data[sigma_var.name]  # Should be changed with #41
                for index, component in enumerate(row):
                    result[index] += np.sum(
                        component(**self.independent_data, **parameters) * ((dep_data - ans) / sigma ** 2)
                    )
        return - np.array(result).T


class LogLikelihood(GradientObjective):
    def __call__(self, **parameters):
        """
        Error function to be maximised(!) in the case of log-likelihood fitting.

        :param p: guess params
        :param data: xdata
        :return: scalar value of log-likelihood
        """
        jac_kwargs = dict(**parameters, **self.independent_data)
        ans = - np.nansum(np.log(self.model(**jac_kwargs)))
        return ans

    def eval_jacobian(self, **parameters):
        """
        Jacobian for log-likelihood is defined as :math:`\\nabla_{\\vec{p}}( \\log( L(\\vec{p} | \\vec{x})))`.

        :param p: guess params
        :param data: data for the variables.
        :return: array of length number of ``Parameter``'s in the model, with all partial derivatives evaluated at p, data.
        """
        jac_kwargs = dict(**parameters, **self.independent_data)
        ans = []
        for row in self.model.numerical_jacobian:
            for partial_derivative in row:
                ans.append(
                    - np.nansum(
                        partial_derivative(**jac_kwargs).flatten() / self.model(**jac_kwargs)
                    )
                )
        else:
            return np.array(ans)


class MinimizeModel(BaseObjective):
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