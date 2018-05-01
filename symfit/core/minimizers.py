import abc
from collections import namedtuple

from scipy.optimize import minimize
import sympy
import numpy as np

from .support import key2str, keywordonly
from .leastsqbound import leastsqbound
from .fit_results import FitResults

DummyModel = namedtuple('DummyModel', 'params')

class BaseMinimizer(object):
    """
    ABC for all Minimizers.
    """
    def __init__(self, objective, parameters):
        """
        :param objective: Objective function to be used.
        :param parameters: List of :class:`~symfit.core.argument.Parameter` instances
        """
        self.objective = objective
        self.params = parameters

    @abc.abstractmethod
    def execute(self, **options):
        """
        The execute method should implement the actual minimization procedure,
        and should return a :class:`~symfit.core.fit_results.FitResults` instance.
        :param options: options to be used by the minimization procedure.
        :return:  an instance of :class:`~symfit.core.fit_results.FitResults`.
        """
        pass

    @property
    def initial_guesses(self):
        return [p.value for p in self.params]

class BoundedMinimizer(BaseMinimizer):
    """
    ABC for Minimizers that support bounds.
    """
    @property
    def bounds(self):
        return [(p.min, p.max) for p in self.params]

class ConstrainedMinimizer(BaseMinimizer):
    """
    ABC for Minimizers that support constraints
    """
    @keywordonly(constraints=None)
    def __init__(self, *args, **kwargs):
        constraints = kwargs.pop('constraints')
        super(ConstrainedMinimizer, self).__init__(*args, **kwargs)
        self.constraints = constraints

class GradientMinimizer(BaseMinimizer):
    """
    ABC for Minizers that support the use of a jacobian
    """

    @keywordonly(jacobian=None)
    def __init__(self, *args, **kwargs):
        jacobian = kwargs.pop('jacobian')
        super(GradientMinimizer, self).__init__(*args, **kwargs)
        self.jacobian = jacobian


class ScipyMinimize(object):
    """
    Mix-in class that handles the execute calls to scipy.optimize.minimize.
    """
    def __init__(self, *args, **kwargs):
        self.constraints = []
        self.jacobian = None
        self.wrapped_jacobian = None
        super(ScipyMinimize, self).__init__(*args, **kwargs)
        self.wrapped_objective = self.wrap_func(self.objective)

    def wrap_func(self, func):
        """
        Given an objective function `func`, make sure it is always called via
        keyword arguments with the relevant parameter names.

        :param func: Function to be wrapped to keyword only calls.
        :return: wrapped function
        """
        # parameters = {param.name: value for param, value in zip(self.params, values)}
        if func is None:
            return None
        def wrapped_func(values):
            parameters = key2str(dict(zip(self.params, values)))
            return func(**parameters)
        return wrapped_func

    @keywordonly(tol=1e-9)
    def execute(self, bounds=None, jacobian=None, **minimize_options):
        ans = minimize(
            self.wrapped_objective,
            self.initial_guesses,
            bounds=bounds,
            constraints=self.constraints,
            jac=self.wrapped_jacobian,
            **minimize_options
        )
        # Build infodic
        infodic = {
            'nfev': ans.nfev,
        }

        fit_results = dict(
            model=DummyModel(params=self.params),
            popt=ans.x,
            covariance_matrix=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit,
            objective_value=ans.fun,
        )

        if 'hess_inv' in ans:
            try:
                fit_results['hessian_inv'] = ans.hess_inv.todense()
            except AttributeError:
                fit_results['hessian_inv'] = ans.hess_inv
        return FitResults(**fit_results)

    @staticmethod
    def scipy_constraints(constraints, data):
        """
        Returns all constraints in a scipy compatible format.

        :return: dict of scipy compatible statements.
        """
        cons = []
        types = {  # scipy only distinguishes two types of constraint.
            sympy.Eq: 'eq', sympy.Ge: 'ineq',
        }

        for key, constraint in enumerate(constraints):
            # jac = make_jac(c, p)
            cons.append({
                'type': types[constraint.constraint_type],
                # Assume the lhs is the equation.
                'fun': lambda p, x, c: c(*(list(x.values()) + list(p)))[0],
                # Assume the lhs is the equation.
                'jac': lambda p, x, c: [component(*(list(x.values()) + list(p)))
                                        for component in
                                        c.numerical_jacobian[0]],
                'args': (data, constraint)
            })
        cons = tuple(cons)
        return cons

class ScipyGradientMinimize(ScipyMinimize, GradientMinimizer):
    def __init__(self, *args, **kwargs):
        super(ScipyGradientMinimize, self).__init__(*args, **kwargs)
        self.wrapped_jacobian = self.wrap_func(self.jacobian)

    def execute(self, **minimize_options):
        return super(ScipyGradientMinimize, self).execute(jacobian=self.wrapped_jacobian, **minimize_options)


class BFGS(ScipyGradientMinimize):
    def execute(self, **minimize_options):
        return super(BFGS, self).execute(method='BFGS', **minimize_options)


class SLSQP(ScipyGradientMinimize, ConstrainedMinimizer, BoundedMinimizer):
    def execute(self, **minimize_options):
        return super(SLSQP, self).execute(
            method='SLSQP',
            bounds=self.bounds,
            **minimize_options
        )


class LBFGSB(ScipyGradientMinimize, BoundedMinimizer):
    def execute(self, **minimize_options):
        return super(LBFGSB, self).execute(
            method='L-BFGS-B',
            bounds=self.bounds,
            **minimize_options
        )


class NelderMead(ScipyMinimize, BaseMinimizer):
    def execute(self, **minimize_options):
        return super(NelderMead, self).execute(method='Nelder-Mead', **minimize_options)


class MINPACK(ScipyMinimize, GradientMinimizer, BoundedMinimizer):
    """
    Wrapper to scipy's implementation of MINPACK, since it is the industry
    standard.
    """
    def __init__(self, *args, **kwargs):
        self.jacobian = None
        super(MINPACK, self).__init__(*args, **kwargs)

    def execute(self, **minpack_options):
        """
        :param minpack_options: Any named arguments to be passed to leastsqbound
        """
        popt, pcov, infodic, mesg, ier = leastsqbound(
            self.wrapped_objective,
            # Dfun=self.jacobian,
            x0=self.initial_guesses,
            bounds=self.bounds,
            full_output=True,
            **minpack_options
        )

        fit_results = dict(
            model=DummyModel(params=self.params),
            popt=popt,
            covariance_matrix=None,
            infodic=infodic,
            mesg=mesg,
            ier=ier,
            chi_squared=np.sum(infodic['fvec']**2),
        )

        return FitResults(**fit_results)