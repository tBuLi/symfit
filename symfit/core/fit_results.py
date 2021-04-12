# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict

import numpy as np

from symfit.core.objectives import (
    LeastSquares, VectorLeastSquares, LogLikelihood
)
from symfit.core.support import keywordonly

class FitResults(object):
    """
    Class to display the results of a fit in a nice and unambiguous way.
    All things related to the fit are available on this class, e.g.
    - parameter values + stdev
    - R squared (Regression coefficient.) or other fit quality qualifiers.
    - fitting status message
    - covariance matrix
    - objective and minimizer used.

    Contains the attribute `params`, which is an
    :class:`~collections.OrderedDict` containing all the parameter names and
    their optimized values. Can be `**` unpacked when evaluating
    :class:`~symfit.core.models.Model`'s.
    """
    @keywordonly(constraints=None)
    def __init__(self, model, popt, covariance_matrix, minimizer, objective, message, **minimizer_output):
        """
        :param model: :class:`~symfit.core.models.Model` that was fit to.
        :param popt: best fit parameters, same ordering as in model.params.
        :param covariance_matrix: covariance matrix.
        :param minimizer: Minimizer instance used.
        :param objective: Objective function which was optimized.
        :param message: Status message returned by the minimizer.
        :param \**minimizer_output: Raw output as given by the minimizer.
        """
        constraints = minimizer_output.pop('constraints')
        self.constraints = constraints if constraints is not None else []
        self.minimizer_output = minimizer_output
        self.model = model
        self.minimizer = minimizer
        self.objective = objective
        # LBFGSB returns with python<=3.6 a bit-string as message
        self.status_message = message.decode() if isinstance(message, bytes) else message

        self._popt = popt
        self.params = OrderedDict(
            [(p.name, value) for p, value in zip(self.model.params, popt)]
        )
        self.covariance_matrix = covariance_matrix
        self.gof_qualifiers = self._gof_qualifiers()

    @property
    def iterations(self):
        if 'nit' in self.minimizer_output:
            return self.minimizer_output['nit']
        else:
            return None

    def __str__(self):
        """
        Pretty print the results as a table.
        """
        res = '\nParameter Value        Standard Deviation\n'
        for p in self.model.params:
            value = self.value(p)
            value_str = '{:e}'.format(value) if value is not None else 'None'
            stdev = self.stdev(p)
            stdev_str = '{:e}'.format(stdev) if stdev is not None else 'None'
            res += '{:10}{} {}\n'.format(p.name, value_str, stdev_str, width=20)

        res += '{:<22} {}\n'.format('Status message', self.status_message)
        res += '{:<22} {}\n'.format('Number of iterations', self.iterations)
        res += '{:<22} {}\n'.format('Objective', self.objective)
        res += '{:<22} {}\n'.format('Minimizer', self.minimizer)

        res += '\nGoodness of fit qualifiers:\n'
        res += '\n'.join('{:<22} {}'.format(gof, value)
                         for gof, value in sorted(self.gof_qualifiers.items()))

        if self.constraints:
            res += '\n\nConstraints:\n'
            res += 20 * '-' + '\n'
            # res += '{:<22} {}\n'.format('Constraint', 'Value')
            for constraint in self.constraints:
                # Print the component and the value of the constraint
                res += 'Question: {} {} 0?\n'.format(
                    constraint.model[constraint.model.dependent_vars[0]],
                    constraint.model.constraint_type.rel_op
                )
                res += 'Answer:   {}\n\n'.format(constraint(**self.params)[0])

        return res

    def __getattr__(self, item):
        """
        Return the requested `item` if it can be found in the gof_qualifiers
        dict.

        :param item: Name of Goodness of Fit qualifier.
        :return: Goodness of Fit qualifier if present.
        """
        if 'gof_qualifiers' in vars(self):
            if item in self.gof_qualifiers:
                return self.gof_qualifiers[item]
        raise AttributeError

    def stdev(self, param):
        """
        Return the standard deviation in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Standard deviation of ``param``.
        """
        try:
            return np.sqrt(self.variance(param))
        except (AttributeError, TypeError):
            # This happens when variance returns None.
            return None

    def value(self, param):
        """
        Return the value in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Value of ``param``.
        """
        return self.params[param.name]

    def variance(self, param):
        """
        Return the variance in a given parameter as found by the fit.

        :param param: ``Parameter`` Instance.
        :return: Variance of ``param``.
        """
        param_number = self.model.params.index(param)
        try:
            return self.covariance_matrix[param_number, param_number]
        except TypeError:
            # covariance_matrix can be None
            return None

    def covariance(self, param_1, param_2):
        """
        Return the covariance between param_1 and param_2.

        :param param_1: ``Parameter`` Instance.
        :param param_2: ``Parameter`` Instance.
        :return: Covariance of the two params.
        """
        param_1_number = self.model.params.index(param_1)
        param_2_number = self.model.params.index(param_2)
        return self.covariance_matrix[param_1_number, param_2_number]

    @staticmethod
    def _array_safe_dict_eq(one_dict, other_dict):
        """
        Dicts containing arrays are hard to compare. This function uses
        numpy.allclose to compare arrays, and does normal comparison for all
        other types.

        This pretty mich defines FitResult equality, but because there are
        still some questions on how and if that should be defined, __eq__ has
        not been implemented.

        :param one_dict: __dict__ of a FitResults object
        :param other_dict: __dict__ of a FitResults object
        :return: bool
        """
        for key in one_dict:
            try:
                if key == 'minimizer':
                    assert one_dict[key].__class__ == other_dict[key].__class__
                elif key == 'minimizer_output':
                    # Ignore this, because it can contain unexpected terms and
                    # if all the derived attributes are correct I see no reason
                    # why this term shouldn't be at least close enough.
                    pass
                else:
                    assert one_dict[key] == other_dict[key]
            except ValueError as err:
                # When dealing with arrays, we need to use numpy for comparison
                if isinstance(one_dict[key], dict):
                    assert FitResults._array_safe_dict_eq(one_dict[key], other_dict[key])
                else:
                    assert np.allclose(one_dict[key], other_dict[key])
            except AssertionError:
                return False
        else: return True

    def __getstate__(self):
        state = self.__dict__.copy()

        if hasattr(state['minimizer'], 'minimizers'):  # ChainedMinimizer
            # ToDo: when py27 support is droppend at least this can be replaced
            #       with just pickling the instance, perhaps also for other
            #       minimizers.
            minimizer_cls = [type(state['minimizer'])]
            minimizer_cls.extend(
                [type(minimizer) for minimizer in state['minimizer'].minimizers]
            )
        else:
            minimizer_cls = type(state['minimizer'])
        state['minimizer'] = (minimizer_cls,
                              state['minimizer'].objective,
                              state['minimizer'].parameters)
        return state

    def __setstate__(self, state):
        min_class, objective, parameters = state['minimizer']
        try:
            # If min_class is iterable, initiate a ChainedMinimizer.
            minimizers = [cls(objective, parameters) for cls in min_class[1:]]
        except TypeError:
            state['minimizer'] = min_class(objective, parameters)
        else:
            state['minimizer'] = min_class[0](objective, parameters,
                                              minimizers=minimizers)
        self.__dict__.update(state)

    def _gof_qualifiers(self):
        """
        Based on the objective used, we can infer certain goodness of fit
        (g.o.f.) qualifiers.

        The ``objective_value`` itself always exists, and then depending on the
        objective we also get the following:

        - The coefficient of determination :math:`R^2` and :math:`\\chi^2` for
          :class:`~symfit.core.objectives.LeastSquares` and
          :class:`~symfit.core.objectives.VectorLeastSquares`.
        - Likelihood and log-likelihood for
          :class:`~symfit.core.objectives.LogLikelihood`.

        :return: ``dict`` containing goodness of fit qualifiers.
        """
        gof_qualifiers = {}

        gof_qualifiers['objective_value'] = self.minimizer_output['fun']
        if isinstance(self.objective, (LeastSquares, VectorLeastSquares)):
            R2 = r_squared(self.objective.model, fit_result=self,
                           data=self.objective.data)
            gof_qualifiers['r_squared'] = R2

        if isinstance(self.objective, VectorLeastSquares):
            # In this case the objective value is the residuals
            chi_squared = np.sum(gof_qualifiers['objective_value'] ** 2)
            gof_qualifiers['chi_squared'] = chi_squared
        elif isinstance(self.objective, LeastSquares):
            # Undo the normalization to get back chi^2.
            gof_qualifiers['chi_squared'] = 2 * gof_qualifiers['objective_value']
        elif isinstance(self.objective, LogLikelihood):
            # We undo the minus sign we have included to maximize likelihood
            gof_qualifiers['log_likelihood'] = - gof_qualifiers['objective_value']
            gof_qualifiers['likelihood'] = np.exp(gof_qualifiers['log_likelihood'])
        return gof_qualifiers


def r_squared(model, fit_result, data):
    """
    Calculates the coefficient of determination, R^2, for the fit.

    (Is not defined properly for vector valued functions.)

    :param model: Model instance
    :param fit_result: FitResults instance
    :param data: data with which the fit was performed.
    """
    # First filter out the dependent vars
    y_is = [data[var] for var in model.dependent_vars if var in data]
    x_is = [data[var] for var in model.independent_vars if var in data]
    y_bars = [np.mean(y_i) if y_i is not None else None for y_i in y_is]
    f_is = model(*x_is, **fit_result.params)._asdict()
    # f_is also contains the evaluated interdependent_vars, skip those.
    f_is = [f_is[var] for var in model.dependent_vars]
    SS_res = np.sum([np.sum((y_i - f_i)**2) for y_i, f_i in zip(y_is, f_is) if y_i is not None])
    SS_tot = np.sum([np.sum((y_i - y_bar)**2) for y_i, y_bar in zip(y_is, y_bars) if y_i is not None])
    return 1 - SS_res/SS_tot