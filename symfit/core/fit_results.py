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

    Contains the attribute `params`, which is an
    :class:`~collections.OrderedDict` containing all the parameter names and
    their optimized values. Can be `**` unpacked when evaluating
    :class:`~symfit.core.fit.Model`'s.
    """
    def __init__(self, model, popt, covariance_matrix, infodic, mesg, ier, **gof_qualifiers):
        """
        Excuse the ugly names of most of these variables, they are inherited from scipy. Will be changed.

        :param model: :class:`~symfit.core.fit.Model` that was fit to.
        :param popt: best fit parameters, same ordering as in model.params.
        :param pcov: covariance matrix.
        :param infodic: dict with fitting info.
        :param mesg: Status message.
        :param ier: Number of iterations.
        :param gof_qualifiers: Any remaining keyword arguments should be
          Goodness of fit (g.o.f.) qualifiers.
        """
        # Validate the types in rough way
        self.infodict = infodic
        self.status_message = mesg
        self.iterations = ier
        self.model = model
        self.gof_qualifiers = gof_qualifiers

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
        if item in self.gof_qualifiers:
            return self.gof_qualifiers[item]
        else:
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
