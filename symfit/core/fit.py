import abc

import sympy
import numpy as np
from scipy.optimize import minimize

from symfit.core.argument import Parameter, Variable
from symfit.core.support import seperate_symbols, sympy_to_scipy, sympy_to_py
from symfit.core.leastsqbound import leastsqbound


class ParameterDict(object):
    """
    Behaves like a dict when **-ed, allowing the sexy syntax where a model is
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
        return len(self.__params)

    def __iter__(self):
        return iter(self.__params)

    def __getitem__( self, key):
        """
        Intended use for this is for use of ParameterDict as a kwarg.
        Therefore return the value of the param, as this is what the user
        expects.
        :return: getattr(self, key), the value of the param with name 'key'
        """
        return getattr(self, key)

    def keys(self):
        return self.__params_dict.keys()

    def __getattr__(self, name):
        """
        A user can access the value of a parameter directly through this object.
        :param name: Name of a param in __params.
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
        :param param: Parameter object.
        :return: returns the numerical value of param
        """
        assert isinstance(param, Parameter)
        return self.__popt[param.name]

    def get_stdev(self, param):
        """
        :param param: Parameter object.
        :return: returns the standard deviation of param
        """
        assert isinstance(param, Parameter)
        return self.__pstdev[param.name]


class FitResults(object):
    """
    Class to display the results of a fit in a nice and unambiguous way.
    All things related to the fit are available on this class, e.g.
    - paramameters + stdev
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

    def __init__(self, params, popt, pcov, infodic, mesg, ier, ydata):
        """
        Excuse the ugly names of most of these variables, they are inherited
        from scipy.
        :param params:
        :param popt:
        :param pcov:
        :param infodic:
        :param mesg:
        :param ier:
        :param ydata:
        :return:
        """
        # Validate the types in rough way
        assert(type(infodic) == dict)
        self.__infodict = infodic
        assert(type(mesg) == str)
        self.__status_message = mesg
        assert(type(ier) == int)
        self.__iterations = ier
        assert(type(ydata) == np.ndarray)
        self.__ydata = ydata  # Needed to calculate R^2
        self.__params = ParameterDict(params, popt, pcov)

    def __str__(self):
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

    #
    # READ-ONLY Properties
    # What follows are all the read-only properties of this object.
    # Their definitions are mostly trivial, but necessary to make sure that
    # FitResults can't be changed.
    #

    @property
    def infodict(self):
        return self.__infodict

    @property
    def status_message(self):
        return self.__status_message

    @property
    def iterations(self):
        return self.__iterations

    @property
    def params(self):
        return self.__params

    @property
    def r_squared(self):
        """
        Getter for the r_squared property.
        :return: Regression coefficient.
        """
        ss_err = (self.infodict['fvec'] ** 2).sum()
        ss_tot = ((self.__ydata - self.__ydata.mean()) ** 2).sum()
        return 1 - (ss_err / ss_tot)


class BaseFit(object):
    __metaclass__  = abc.ABCMeta
    __jac = None  # private attribute for the jacobian
    __fit_results = None  # private attribute for the fit_results

    def __init__(self, model, *args, **kwargs):
        """
        :param model: sympy expression.
        :param x: xdata to fit to.  NxM
        :param y: ydata             Nx1
        """
        super(BaseFit, self).__init__(*args, **kwargs)
        self.model = model
        # Get all parameters and variables from the model.
        self.vars, self.params = seperate_symbols(self.model)
        # Compile a scipy function
        self.scipy_func = sympy_to_scipy(self.model, self.vars, self.params)


    def eval_jacobian(self, p, func, x, y):
        """
        Evaluate the jacobian of the model.
        :p: vector of parameter values
        :func: scipy-type function
        :x: array of x values for the evaluation of the Jacobian.
        :y: ydata. Not typically used for determination of the Jacobian but provided by the calling scipy by default.
        :return: vector of values of the derivatives with respect to each parameter.
        """
        residues = []
        for jac in self.jacobian:
            # raise Exception(inspect.getargspec(jac))
            residue = jac(x, p)
            # raise Exception('')
            # If only params in f, we must multiply with an array to preserve the shape of x
            try:
                len(residue)
            except TypeError:  # not itterable
                if len(x):
                    residue *= np.ones(len(x))
                else:
                    residue = np.array([residue])
            finally:
                # res = np.atleast_2d(res)
                residues.append(residue)
                # raise Exception(residue, x)

        return np.array(residues).T

    def get_bounds(self):
        """
        :return: List of tuples of all bounds on parameters.
        """
        return [(np.nextafter(p.value, 0), p.value) if p.fixed else (p.min, p.max) for p in self.params]

    @property
    def jacobian(self):
        """
        Get the scipy functions for the Jacobian. This returns functions only, not values.
        :return: array of derivative functions in all parameters, not values.
        """
        if not self.__jac:
            self.__jac = []
            for param in self.params:
                # Differentiate to every param
                f = sympy.diff(self.model, param)
                # Make them into pythonic functions
                self.__jac.append(sympy_to_scipy(f, self.vars, self.params))
        return self.__jac

    @property
    def fit_results(self):
        """
        FitResults are a read-only property, because we don't want people messing with their
        FitResults, do we?
        :return: FitResult object if available, else None
        """
        return self.__fit_results if self.__fit_results else None

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def get_initial_guesses(self):
        return np.array([param.value for param in self.params])

    @abc.abstractmethod
    def error(self, p, func, x, y):
        """
        Error function to be minimalised. Depending on the algorithm, this can
        return a scalar or a vector.
        :param p: guess params
        :param func: scipy_func to fit to
        :param x: xdata
        :param y: ydata
        :return: scalar of vector.
        """
        return

class LeastSquares(BaseFit):
    def __init__(self, model, x, y, weights=None, sigma=None, *args, **kwargs):
        """
        Least squares fitting. In the notation used for x and y below,
        N_1 - N_i indicate the dimension of the array inserted, and M
        the number of variables. Either the first or the last dim must
        be of size M.
        Vector-valued functions are not currently supported.

        :param model: sympy expression.
        :param x: xdata to fit to.  N_1 x ... x N_i x M
        :param y: ydata             N_1 x ... x N_i
        """
        super(LeastSquares, self).__init__(model, *args, **kwargs)
        # flatten x and y to all but the final dimension.
        # Also checks if the number of variables matches the dim of x
        self.xdata, self.ydata = self._flatten(x, y)

        # set self.sigma with the relevant data.
        if sigma is not None and weights is not None:
            raise Exception('Please provide only weights or sigma, not both.')
        elif weights is not None:
            # We use sigma to calculate the error function.
            self.sigma = 1 / np.sqrt(weights)
        elif sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = 1

        # Check that the dimensions of sigma and y are the same.
        try:
            self.sigma.shape
        except AttributeError:
            pass
        else:
            if y.shape != self.sigma.shape:
                raise Exception('y and sigma must have the same shape.')
            else:
                # self.sigma is an array if this else is reached, so we flatten it.
                self.sigma = self.sigma.reshape(-1)  # flatten




    def _flatten(self, x, y):
        """
        Flattens x up to the dimension of size len(self.vars) and y completely.
        :param x: array of shape N1 x ... x Ni x len(self.vars)
        :param y: array of shape N1 x ... x Ni
        :return: x as (N1 x ... x Ni) x len(self.vars)
                 y as (N1 x ... x Ni)
        """
        if len(self.vars) not in x.shape and not len(x.shape) == 1:
            raise Exception('number of vars does not match the shape of the input x data.')
        elif len(x.shape) == 1:  # this means x is already 1D.
            if x.shape != y.shape:
                raise Exception(
                    'x and y must have the same shape. x has shape {}, whereas y has shape {}.'.format(x.shape,
                                                                                                       y.shape))
            else:  # this data is already flattened.
                return x, y
        else:
            # If the last x dim is as deep as the no of vars, the remaining dimensions should match those of y.
            # Furthermore, the shapes are properly alligned.
            if x.shape[-1] == len(self.vars) and x.shape[:-1] == y.shape:
                x1 = x.T.reshape(len(self.vars), -1)  # flatten all but the dim containing the vars.
                y1 = y.T.reshape(-1)  # flatten
                return x1, y1
            # This is also acceptable, but we will have to transpose the arrays before flattening
            # for the result to make sense.
            elif x.shape[0] == len(self.vars) and x.shape[1:] == y.shape:
                # raise Exception(x.shape, y.shape)
                x1 = x.reshape(len(self.vars), -1)  # flatten all but the dim containing the vars.
                y1 = y.reshape(-1)  # flatten
                return x1, y1
            else:
                raise Exception(
                    'For multidimensional data, the first or the last dimension of xdata is expected to represent the different variables, and the shape of the remaining dimensions should match that of ydata.'
                )

    def execute(self, *args, **kwargs):
        """
        Run fitting and initiate a fit report with the result.
        :return: FitResults object
        """
        popt, cov_x, infodic, mesg, ier = leastsqbound(
            self.error,
            self.get_initial_guesses(),
            args=(self.scipy_func, self.xdata, self.ydata),
            bounds=self.get_bounds(),
            Dfun=self.eval_jacobian,
            full_output=True,
            *args,
            **kwargs
        )

        s_sq = (infodic['fvec'] ** 2).sum() / (len(self.ydata) - len(popt))
        # raise Exception(infodic['fvec'], self.ydata.shape, self.xdata.shape)
        pcov = cov_x * s_sq if cov_x is not None else None
        self.__fit_results = FitResults(
            params=self.params,
            popt=popt,
            pcov=pcov,
            infodic=infodic,
            mesg=mesg,
            ier=ier,
            ydata=self.ydata,  # Needed to calculate R^2
        )
        return self.__fit_results

    def error(self, p, func, x, y):
        """
        :param p: param vector
        :param func: pythonic function
        :param x: xdata
        :param y: ydata
        :return: difference between the data and the fit for the given params.
        This function and eval_jacobian should have been staticmethods, but that
        way eval_jacobian does not work.
        """
        ans = func(x, p)
        # self.sigma here should be replaced by sigma as an arg!
        # More sleep is needed before I can think about this...
        return (ans - y)/self.sigma

    def get_initial_guesses(self):
        """
        Constructs a list of initial guesses from the Parameter objects.
        If no initial value is given, 1.0 is used.
        :return: list of initial guesses for self.params.
        """
        return super(LeastSquares, self).get_initial_guesses()

class Fit(LeastSquares):
    """
    Wrapper for LeastSquares to give it a more appealing name.
    """
    pass


class Minimize(BaseFit):
    def __init__(self, model, xdata=None, ydata=None, constraints=None, *args, **kwargs):
        """
        :model: Model to minimize
        :constraints: constraints the minimization is subject to
        :xdata:
        :ydata: data the minimization is subject to.
        """
        super(Minimize, self).__init__(model)
        self.xdata = xdata if xdata is not None else np.array([])
        self.ydata = ydata if ydata is not None else np.array([])
        self.constraints = constraints if constraints else []

    def error(self, p, func, x, y):
        if x != np.array([]) and y != np.array([]):
            return func(x, p) - y
        else:
            return func(x, p)

    def get_initial_guesses(self):
        return super(Minimize, self).get_initial_guesses()

    def execute(self, method='SLSQP', *args, **kwargs):
        ans = minimize(
            self.error,
            self.get_initial_guesses(),
            args=(self.scipy_func, self.xdata, self.ydata),
            method=method,
            # method='L-BFGS-B',
            bounds=self.get_bounds(),
            constraints = self.get_constraints(),
            jac=self.eval_jacobian,
            options={'disp': True},
        )

        # Build infodic
        infodic = {
            'fvec': ans.fun,
            'nfev': ans.nfev,
        }
        # s_sq = (infodic['fvec'] ** 2).sum() / (len(self.ydata) - len(popt))
        # pcov = cov_x * s_sq if cov_x is not None else None
        self.__fit_results = FitResults(
            params=self.params,
            popt=ans.x,
            pcov=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit,
            ydata=self.ydata,  # Needed to calculate R^2
        )
        return self.__fit_results

    def get_constraints(self):
        """
            Turns self.constraints into a scipy compatible format.
            :return: dict of scipy compatile statements.
            """
        from sympy import Eq, Gt, Ge, Ne, Lt, Le

        cons = []
        types = {
            Eq: 'eq', Gt: 'ineq', Ge: 'ineq', Ne: 'ineq', Lt: 'ineq', Le: 'ineq'
        }

        def make_jac(constraint_lhs, p, x):
            """
            :param constraint_lhs: equation of a constraint. The lhs is assumed to be an eq, rhs a number.
            :param p: current value of the parameters to be evaluated.
            :return: numerical jacobian.
            """
            sym_jac = []
            for param in self.params:
                sym_jac.append(sympy.diff(constraint_lhs, param))
            ans = np.array(
                [sympy_to_scipy(jac, self.vars, self.params)(x, p) for jac in
                 sym_jac]
            )
            return ans

        for key, constraint in enumerate(self.constraints):
            # jac = make_jac(c, p)
            cons.append({
                'type': types[constraint.__class__],
                # Assume the lhs is the equation.
                'fun': lambda p, x, c: sympy_to_scipy(c.lhs, self.vars, self.params)(x, p),
                # Assume the lhs is the equation.
                'jac' : lambda p, x, c: make_jac(c.lhs, p, x),
                'args': (self.xdata, constraint)
            })
        cons = tuple(cons)
        return cons



class Maximize(Minimize):
    def error(self, p, func, x, y):
        """ Change the sign in order to maximize. """
        return - super(Maximize, self).error(p, func, x, y)

    def eval_jacobian(self, p, func, x, y):
        """ Change the sign in order to maximize. """
        return - super(Maximize, self).eval_jacobian(p, func, x, y)


class Likelihood(Maximize):
    """
    Fit using a Likelihood approach.
    """
    def __init__(self, model, xdata, *args, **kwargs):
        """
        :param model: sympy expression.
        :param x: xdata to fit to.  Nx1
        """
        super(Likelihood, self).__init__(model, xdata=xdata, *args, **kwargs)

    def execute(self, method='SLSQP', *args, **kwargs):
        # super(Likelihood, self).execute(*args, **kwargs)
        ans = minimize(
            self.error,
            self.get_initial_guesses(),
            args=(self.scipy_func, self.xdata, self.ydata),
            method=method,
            bounds=self.get_bounds(),
            constraints = self.get_constraints(),
            # jac=self.eval_jacobian, # If I find a meaning to jac I'll let you know.
            options={'disp': True},
        )

        # Build infodic
        infodic = {
            'fvec': ans.fun,
            'nfev': ans.nfev,
        }

        self.__fit_results = FitResults(
            params=self.params,
            popt=ans.x,
            pcov=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit,
            ydata=self.ydata,  # Needed to calculate R^2
        )
        return self.__fit_results

    def get_initial_guesses(self):
        return super(Likelihood, self).get_initial_guesses()

    def error(self, p, func, x, y=None):
        """
        Error function to be maximised(!) in the case of likelihood fitting.
        :param p: guess params
        :param func: scipy_func to fit to
        :param x: xdata
        :param y: Not important for likelihood.
        :return: scalar of vector.
        """
        ans = - np.nansum(np.log(func(x, p)))
        # ans = - np.product(func(x, p)) # Why doesn't this work?
        return ans