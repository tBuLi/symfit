import abc
from collections import namedtuple
from functools import partial, wraps

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
        self.parameters = parameters
        self._fixed_params = [p for p in parameters if p.fixed]
        self.objective = partial(objective, **{p.name: p.value for p in self._fixed_params})
        self.params = [p for p in parameters if not p.fixed]

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
        self.constraints = [
            partial(constraint, **{p.name: p.value for p in self._fixed_params})
            for constraint in constraints
        ]

class GradientMinimizer(BaseMinimizer):
    """
    ABC for Minizers that support the use of a jacobian
    """
    @keywordonly(jacobian=None)
    def __init__(self, *args, **kwargs):
        jacobian = kwargs.pop('jacobian')
        super(GradientMinimizer, self).__init__(*args, **kwargs)

        if jacobian is not None:
            jac_with_fixed_params = partial(jacobian, **{p.name: p.value for p in self._fixed_params})
            self.wrapped_jacobian = self.resize_jac(jac_with_fixed_params)
        else:
            self.jacobian = None
            self.wrapped_jacobian = None

    def resize_jac(self, func):
        """
        Removes values with identical indices to fixed parameters from the
        output of func. func has to return the jacobian of a scalar function.
        :param func: Jacobian function to be wrapped. Is assumed to be the
            jacobian of a scalar function.
        :return: Jacobian corresponding to non-fixed parameters only.
        """
        if func is None:
            return None
        @wraps(func)
        def resized(*args, **kwargs):
            out = func(*args, **kwargs)
            # Make one dimensional, corresponding to a scalar function.
            out = np.atleast_1d(np.squeeze(out))
            mask = [p not in self._fixed_params for p in self.parameters]
            return out[mask]
        return resized

class ScipyMinimize(object):
    """
    Mix-in class that handles the execute calls to :func:`scipy.optimize.minimize`.
    """
    def __init__(self, *args, **kwargs):
        self.constraints = []
        self.jacobian = None
        self.wrapped_jacobian = None
        super(ScipyMinimize, self).__init__(*args, **kwargs)
        self.wrapped_objective = self.list2kwargs(self.objective)

    def list2kwargs(self, func):
        """
        Given an objective function `func`, make sure it is always called via
        keyword arguments with the relevant parameter names.

        :param func: Function to be wrapped to keyword only calls.
        :return: wrapped function
        """
        if func is None:
            return None
        # Because scipy calls the objective with a list of parameters as
        # guesses, we use 'values' instead of '*values'.
        @wraps(func)
        def wrapped_func(values):
            parameters = key2str(dict(zip(self.params, values)))
            return np.array(func(**parameters))
        return wrapped_func

    @keywordonly(tol=1e-9)
    def execute(self, bounds=None, jacobian=None, constraints=None, **minimize_options):
        ans = minimize(
            self.wrapped_objective,
            self.initial_guesses,
            bounds=bounds,
            constraints=constraints,
            jac=jacobian,
            **minimize_options
        )
        # Build infodic
        infodic = {
            'nfev': ans.nfev,
        }

        best_vals = []
        found = iter(ans.x)
        for param in self.parameters:
            if param.fixed:
                best_vals.append(param.value)
            else:
                best_vals.append(next(found))

        fit_results = dict(
            model=DummyModel(params=self.parameters),
            popt=best_vals,
            covariance_matrix=None,
            infodic=infodic,
            mesg=ans.message,
            ier=ans.nit if hasattr(ans, 'nit') else float('nan'),
            objective_value=ans.fun,
        )

        if 'hess_inv' in ans:
            try:
                fit_results['hessian_inv'] = ans.hess_inv.todense()
            except AttributeError:
                fit_results['hessian_inv'] = ans.hess_inv
        return FitResults(**fit_results)

class ScipyGradientMinimize(ScipyMinimize, GradientMinimizer):
    """
    Base class for :func:`scipy.optimize.minimize`'s gradient-minimizers.
    """
    def __init__(self, *args, **kwargs):
        super(ScipyGradientMinimize, self).__init__(*args, **kwargs)
        self.wrapped_jacobian = self.list2kwargs(self.wrapped_jacobian)

    def execute(self, **minimize_options):
        return super(ScipyGradientMinimize, self).execute(jacobian=self.wrapped_jacobian, **minimize_options)

class ScipyConstrainedMinimize(ScipyMinimize, ConstrainedMinimizer):
    """
    Base class for :func:`scipy.optimize.minimize`'s constrained-minimizers.
    """
    def __init__(self, *args, **kwargs):
        super(ScipyConstrainedMinimize, self).__init__(*args, **kwargs)
        self.wrapped_constraints = self.scipy_constraints(self.constraints)

    def execute(self, **minimize_options):
        return super(ScipyConstrainedMinimize, self).execute(constraints=self.wrapped_constraints, **minimize_options)

    def scipy_constraints(self, constraints):
        """
        Returns all constraints in a scipy compatible format.

        :return: dict of scipy compatible statements.
        """
        cons = []
        types = {  # scipy only distinguishes two types of constraint.
            sympy.Eq: 'eq', sympy.Ge: 'ineq',
        }

        for key, partialed_constraint in enumerate(constraints):
            constraint_type = partialed_constraint.func.constraint_type
            cons.append({
                'type': types[constraint_type],
                # Takes an nd.array of params and a partialed_constraint, and
                # evaluates the constraint with these parameters.
                # Wrap `c` so it is always called via keywords.
                'fun': lambda p, c: self.list2kwargs(c)(list(p))[0],
                'args': [partialed_constraint]
            })

        cons = tuple(cons)
        return cons

class BFGS(ScipyGradientMinimize):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s BFGS algorithm.
    """
    def execute(self, **minimize_options):
        return super(BFGS, self).execute(method='BFGS', **minimize_options)


class SLSQP(ScipyConstrainedMinimize, GradientMinimizer, BoundedMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s SLSQP algorithm.
    """
    def __init__(self, *args, **kwargs):
        super(SLSQP, self).__init__(*args, **kwargs)
        # We have to break DRY because you cannot inherit from both
        # ScipyConstrainedMinimize and ScipyGradientMinimize. So SLQSP is a
        # special case. This is the same code as in ScipyGradientMinimize.
        self.wrapped_jacobian = self.list2kwargs(self.wrapped_jacobian)

    def execute(self, **minimize_options):
        return super(SLSQP, self).execute(
            method='SLSQP',
            bounds=self.bounds,
            jacobian=self.wrapped_jacobian,
            **minimize_options
        )

    def scipy_constraints(self, constraints):
        """
        Returns all constraints in a scipy compatible format.

        :return: dict of scipy compatible constraints, including jacobian term.
        """
        # Take the normal scipy compatible constraints, and add jacobians.
        scipy_constr = super(SLSQP, self).scipy_constraints(constraints)
        for partialed_constraint, scipy_constraint in zip(constraints, scipy_constr):
            partialed_kwargs = partialed_constraint.keywords
            # In the case of the jacobian, first eval_jacobian of the
            # constraint has to be partialed since that has not been done
            # yet. Then it is made callable by keywords only, and finally
            # the shape of the jacobian is made to match the number of
            # unfixed parameters in the call, len(p).
            scipy_constraint['jac'] = lambda p, c: self.resize_jac(
                    self.list2kwargs(
                        partial(c.func.eval_jacobian, **partialed_kwargs)
                    )
                )(list(p))
        return scipy_constr


class COBYLA(ScipyConstrainedMinimize):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s COBYLA algorithm.
    """
    def execute(self, **minimize_options):
        return super(COBYLA, self).execute(
            method='COBYLA', **minimize_options
        )


class LBFGSB(ScipyGradientMinimize, BoundedMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s LBFGSB algorithm.
    """
    def execute(self, **minimize_options):
        return super(LBFGSB, self).execute(
            method='L-BFGS-B',
            bounds=self.bounds,
            **minimize_options
        )


class NelderMead(ScipyMinimize, BaseMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s NelderMead algorithm.
    """
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