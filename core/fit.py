from traits.api import *
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from core.argument import Parameter, Variable
import sympy
import numpy as np
import abc
from leastsqbound import leastsqbound

class FitResults(HasStrictTraits):
    params = List(Parameter)
    popt = Array
    pcov = Array
    infodic = Dict
    mesg = Str
    ier = Int

    # def __init__(self, *args, **kwargs):
    #     """
    #     :param params:
    #     :param popt: Optimal fit parameters.
    #     :param pcov: Errors in said parameters.
    #     """
    #     super(FitResults, self).__init__(*args, **kwargs)

    def __str__(self):
        res = 'Parameter Value        Standard Deviation\n'
        for key, p in enumerate(self.params):
            res += '{:10}{:e} {:e}\n'.format(p.name, self.popt[key], np.sqrt(self.pcov[key, key]), width=20)

        res += 'Fitting status message: {}\n'.format(self.mesg)
        res += 'Number of iterations:   {}\n'.format(self.infodic['nfev'])
        return res


class BaseFit(ABCHasStrictTraits):
    model = Instance(Expr)
    xdata = Array
    ydata = Array
    params = List(Parameter)
    vars = List(Variable)
    scipy_func = Callable
    fit_results = Instance(FitResults)
    jacobian = Property(depends_on='model')

    def __init__(self, model, x, y, *args, **kwargs):
        """
        :param model: sympy expression.
        :param x: xdata to fit to.  NxM
        :param y: ydata             Nx1
        """
        super(BaseFit, self).__init__(*args, **kwargs)
        self.model = model
        self.xdata = x
        self.ydata = y
        # Get all parameters and variables from the model.
        self.params, self.vars = self.seperate_symbols()
        # Check if the number of variables matches the dim of x
        # raise Exception(x.shape, self.vars)
        if len(self.vars) not in x.shape and not len(x.shape) == 1:
            raise Exception('number of vars does not match the shape of the input x data.')
        # Compile a scipy function
        self.scipy_func = self.sympy_to_scipy(model)

    def seperate_symbols(self):
        params = []
        vars = []
        for symbol in self.model.free_symbols:
            if isinstance(symbol, Parameter):
                params.append(symbol)
            elif isinstance(symbol, Variable) or isinstance(symbol, Symbol):
                vars.append(symbol)
            else:
                raise TypeError('model contains an unknown symbol type, {}'.format(type(symbol)))
        return params, vars

    def sympy_to_scipy(self, func):
        """
        Turn the wonderfully elegant sympy expression into an ugly scypy
        function.
        """
        func_str = str(func)
        if len(self.vars) == 1:
            func_str = func_str.replace(str(self.vars[0]), 'x')
        else:
            for key, var in enumerate(self.vars):
                func_str = func_str.replace(str(var), 'x[{}]'.format(key))

        try:
            param_str = str(self.params[0])
        except IndexError:
            param_str = ''
        for param in self.params[1:]:
            param_str += ', {}'.format(param)

        # Replace mathematical functions by their numpy equavalent.
        for key, value in {'log': 'np.log', 'exp': 'np.exp'}.iteritems():
            func_str = func_str.replace(key, value)

        import imp

        code = """
def f(x, {0}):
    return {1}
""".format(param_str, func_str)

        # Compile to fully working python function
        module = imp.new_module('scipy_function')
        exec code in globals(), module.__dict__ # globals() is needed to have numpy defined
        return module.f

    def get_initial_guesses(self):
        """
        Constructs a list of initial guesses from the Parameter objects.
        If no initial value is given, 1.0 is used.
        :return: list of initial guesses for self.params.
        """
        return np.array([param.value for param in self.params])

    def get_jacobian(self, p, func, x, y):
        """
        Create the jacobian of the model. This can then be used by
        :return:
        """
        funcs = []
        for jac in self.jacobian:
            res = jac(x, *p)
            # If only params in f, we must multiply with an array to preserve the shape of x
            try:
                len(res)
            except TypeError: # not itterable
                res *= np.ones(len(x))
            finally:
                funcs.append(res)
        ans = np.array(funcs).T
        return ans

    def get_bounds(self):
        """
        :return: List of tuples of all bounds on parameters.
        """

        return [(np.nextafter(p.value, 0), p.value) if p.fixed else (p.min, p.max) for p in self.params]

    @cached_property
    def _get_jacobian(self):
        jac = []
        for param in self.params:
            # Differentiate to every param
            f = sympy.diff(self.model, param)
            # Make them into pythonic functions
            jac.append(self.sympy_to_scipy(f))
        return jac

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        return

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


class Fit(BaseFit):
    def execute(self, *args, **kwargs):
        """
        Run fitting and initiate a fit report with the result.
        :return: FitResults object
        """
        from scipy.optimize import leastsq

        popt, cov_x, infodic, mesg, ier = leastsqbound(
            self.error,
            self.get_initial_guesses(),
            args=(self.scipy_func, self.xdata, self.ydata),
            bounds=self.get_bounds(),
            Dfun=self.get_jacobian,
            full_output=True,
            *args,
            **kwargs
        )

        s_sq = (infodic['fvec']**2).sum()/(len(self.ydata)-len(popt))
        pcov =  cov_x*s_sq
        self.fit_results = FitResults(
            params=self.params,
            popt=popt, pcov=pcov, infodic=infodic, mesg=mesg, ier=ier
        )
        return self.fit_results

    def error(self, p, func, x, y):
        """
        :param p: param vector
        :param func: pythonic function
        :param x: xdata
        :param y: ydata
        :return: difference between the data and the fit for the given params.
        This function and get_jacobian should have been staticmethods, but that
        way get_jacobian does not work.
        """
        # Must unpack p for func
        return func(x, *p) - y


class Minimize(BaseFit):
    def execute(self, *args, **kwargs):
        """
        Run fitting and initiate a fit report with the result.
        :return: FitResults object
        """
        from scipy.optimize import minimize

        # s_sq = (infodic['fvec']**2).sum()/(len(self.ydata)-len(popt))
        # pcov =  cov_x*s_sq
        # self.fit_results = FitResults(
        #     params=self.params,
        #     popt=popt, pcov=pcov, infodic=infodic, mesg=mesg, ier=ier
        # )
        # return self.fit_results
        ans = minimize(
            self.error,
            # [1.0],
            self.get_initial_guesses(),
            args=(self.scipy_func, self.xdata, self.ydata),
            # method='L-BFGS-B',
            # bounds=self.get_bounds(),
            # jac=self.get_jacobian,
        )
        print ans
        return ans

    def error(self, p, func, x, y):
        ans = ((self.scipy_func(self.xdata, *p) - y)**2).sum()
        print p
        return ans