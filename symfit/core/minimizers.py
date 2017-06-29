import abc

from scipy.optimize import minimize
import sympy
import numpy as np

from .support import key2str, keywordonly
from .leastsqbound import leastsqbound

class BaseMinimizer:
    def __init__(self, objective, parameters, absolute_sigma=True):
        self.objective = objective
        self.params = parameters
        self.absolute_sigma = absolute_sigma

    @abc.abstractmethod
    def execute(self, **options):
        pass

    @property
    def initial_guesses(self):
        return [p.value for p in self.params]

class BoundedMinimizer(BaseMinimizer):
    @property
    def bounds(self):
        return [(p.min, p.max) for p in self.params]

class ConstrainedMinimizer(BaseMinimizer):
    def __init__(self, *args, constraints=None, **kwargs):
        super(ConstrainedMinimizer, self).__init__(*args, **kwargs)
        self.constraints = constraints

class GradientMinimizer(BaseMinimizer):
    def __init__(self, *args, jacobian=None, **kwargs):
        super(GradientMinimizer, self).__init__(*args, **kwargs)
        self.jacobian = jacobian


class ScipyMinimize(object):
    def __init__(self, *args, **kwargs):
        self.constraints = []
        self.jacobian = None
        super(ScipyMinimize, self).__init__(*args, **kwargs)
        self.wrapped_objective = self.wrap_func(self.objective)

    def wrap_func(self, func):
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
            jac=jacobian,
            **minimize_options
        )
        # Build infodic
        infodic = {
            'nfev': ans.nfev,
        }

        fit_results = dict(
            popt=ans.x,
            pcov=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit,
            value=ans.fun,
        )

        return fit_results

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


class MINPACK(GradientMinimizer, BoundedMinimizer):
    """
    Wrapper to scipy's implementation of MINPACK. Since it is the industry
    standard
    """
    def __init__(self, *args, **kwargs):
        self.jacobian = None
        super(MINPACK, self).__init__(*args, **kwargs)
        self.wrapped_objective = ScipyMinimize.wrap_func(self, self.objective)

    # def wrap_func(self, func):
    #     # parameters = {param.name: value for param, value in zip(self.params, values)}
    #     if func is None:
    #         return None
    #     def wrapped_func(values):
    #         # raise Exception(values)
    #         parameters = key2str(dict(zip(self.params, values)))
    #         return np.repeat(np.sqrt(func(**parameters)), len(values) + 1)
    #     return wrapped_func

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

        # if self.absolute_sigma:
        #     s_sq = 1
        # else:
        #     # Rescale the covariance matrix with the residual variance
        #     ss_res = np.sum(infodic['fvec']**2)
        #     for data in self.dependent_data.values():
        #         if data is not None:
        #             degrees_of_freedom = np.product(data.shape) - len(popt)
        #             break
        #
        #     s_sq = ss_res / degrees_of_freedom
        #
        # pcov = cov_x * s_sq if cov_x is not None else None

        fit_results = dict(
            popt=popt,
            # pcov=pcov,
            infodic=infodic,
            mesg=mesg,
            ier=ier,
        )
        # self._fit_results.gof_qualifiers['r_squared'] = \
        #     r_squared(self.model, self._fit_results, self.data)
        return fit_results