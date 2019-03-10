from collections import OrderedDict

import numpy as np

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
    :class:`~symfit.core.fit.Model`'s.
    """
    def __init__(self, model, popt, covariance_matrix, infodict, mesg, ier, minimizer, objective, **gof_qualifiers):
        """
        :param model: :class:`~symfit.core.fit.Model` that was fit to.
        :param popt: best fit parameters, same ordering as in model.params.
        :param pcov: covariance matrix.
        :param infodict: dict with fitting info.
        :param mesg: Status message.
        :param ier: Number of iterations.
        :param minimizer: Minimizer instance used.
        :param objective: Objective function which was optimized.
        :param gof_qualifiers: Any remaining keyword arguments should be
          Goodness of fit (g.o.f.) qualifiers.
        """
        # Validate the types in rough way
        self.infodict = infodict
        self.status_message = mesg
        self.iterations = ier
        self.model = model
        self.gof_qualifiers = gof_qualifiers
        self.minimizer = minimizer
        self.objective = objective

        self._popt = popt
        self.params = OrderedDict([(p.name, value) for p, value in zip(self.model.params, popt)])
        self.covariance_matrix = covariance_matrix


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

        res += 'Fitting status message: {}\n'.format(self.status_message)
        res += 'Number of iterations:   {}\n'.format(self.infodict['nfev'])
        res += 'Objective:              {}\n'.format(self.objective)
        res += 'Minimizer               {}\n'.format(self.minimizer)
        try:
            res += 'Regression Coefficient: {}\n'.format(self.r_squared)
        except AttributeError:
            pass
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
        except AttributeError:
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

        :param one_dict:
        :param other_dict:
        :return: bool
        """
        for key in one_dict:
            try:
                if key == 'objective' or key == 'minimizer':
                    assert one_dict[key].__class__ == other_dict[key].__class__
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

    def __eq__(self, other):
        return FitResults._array_safe_dict_eq(self.__dict__, other.__dict__)