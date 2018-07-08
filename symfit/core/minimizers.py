import abc
import sys
from collections import namedtuple, Counter
from functools import partial

from scipy.optimize import minimize, differential_evolution
import sympy
import numpy as np

from .support import key2str, keywordonly
from .leastsqbound import leastsqbound
from .fit_results import FitResults

if sys.version_info >= (3,0):
    import inspect as inspect_sig
else:
    import funcsigs as inspect_sig

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
        try:
            return self._initial_guesses
        except AttributeError:
            return [p.value for p in self.params]

    @initial_guesses.setter
    def initial_guesses(self, vals):
        self._initial_guesses = vals


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

        if jacobian is not None:
            jac_with_fixed_params = partial(jacobian, **{p.name: p.value for p in self._fixed_params})
            self.wrapped_jacobian = self.resize_jac(jac_with_fixed_params)
        else:
            self.jacobian = None
            self.wrapped_jacobian = None

    def resize_jac(self, func):
        """
        Removes values with identical indices to fixed parameters from the
        output of func.

        :param func: Function to be wrapped
        :return: wrapped function
        """
        if func is None:
            return None
        def wrapped(*args, **kwargs):
            out = func(*args, **kwargs)
            jac = []
            for param, val in zip(self.parameters, out):
                if not param.fixed:
                    jac.append(val)
            return jac
        return wrapped


class GlobalMinimizer(BaseMinimizer):
    """
    A minimizer that looks for a global minimum, instead of a local one.
    """
    def __init__(self, *args, **kwargs):
        super(GlobalMinimizer, self).__init__(*args, **kwargs)


class ChainedMinimizer(BaseMinimizer):
    """
    A minimizer that consists of multiple other minimizers, each executed in
    order.
    This is valuable if you have minimizers that are not good at finding the
    exact minimum such as :class:`~symfit.core.minimizers.NelderMead` or
    :class:`~symfit.core.minimizers.DifferentialEvolution`.
    """
    @keywordonly(minimizers=None)
    def __init__(self, *args, **kwargs):
        '''
        :param minimizers: a :class:`~collections.abc.Sequence` of
            :class:`~symfit.core.minimizers.BaseMinimizer` objects, which need
            to be run in order.
        :param \*args: passed to :func:`symfit.core.minimizers.BaseMinimizer.__init__`.
        :param \*\*kwargs: passed to :func:`symfit.core.minimizers.BaseMinimizer.__init__`.
        '''
        minimizers = kwargs.pop('minimizers')
        super(ChainedMinimizer, self).__init__(*args, **kwargs)
        self.minimizers = minimizers
        self.__signature__ = self._make_signature()

    def execute(self, **minimizer_kwargs):
        """
        Execute the chained-minimization. In order to pass options to the
        seperate minimizers, they can  be passed by using the
        names of the minimizers as keywords. For example::

            fit = Fit(self.model, self.xx, self.yy, self.ydata,
                      minimizer=[DifferentialEvolution, BFGS])
            fit_result = fit.execute(
                DifferentialEvolution={'seed': 0, 'tol': 1e-4, 'maxiter': 10},
                BFGS={'tol': 1e-4}
            )

        In case of multiple identical minimizers an index is added to each
        keyword argument to make them identifiable. For example, if::

            minimizer=[BFGS, DifferentialEvolution, BFGS])

        then the keyword arguments will be 'BFGS', 'DifferentialEvolution',
        and 'BFGS_2'.

        :param minimizer_kwargs: Minimizer options to be passed to the
            minimzers by name
        :return:  an instance of :class:`~symfit.core.fit_results.FitResults`.
        """
        bound_arguments = self.__signature__.bind(**minimizer_kwargs)
        # Include default values in bound_argument object
        for param in self.__signature__.parameters.values():
            if param.name not in bound_arguments.arguments:
                bound_arguments.arguments[param.name] = param.default

        answers = []
        next_guess = self.initial_guesses
        for minimizer, kwargs in zip(self.minimizers, bound_arguments.arguments.values()):
            minimizer.initial_guesses = next_guess
            ans = minimizer.execute(**kwargs)
            next_guess = list(ans.params.values())
            answers.append(ans)
        final = answers[-1]
        # TODO: Compile all previous results in one, instead of just the
        # number of function evaluations. But there's some code down the
        # line that expects scalars.
        final.infodict['nfev'] = sum(ans.infodict['nfev'] for ans in answers)
        return final

    def _make_signature(self):
        """
        Create a signature for `execute` based on the minimizers this
        `ChainedMinimizer` was initiated with. For the format, see the docstring
        of :meth:`ChainedMinimizer.execute`.

        :return: :class:`inspect.Signature` instance.
        """
        # Create KEYWORD_ONLY arguments with the names of the minimizers.
        name = lambda x: x.__class__.__name__
        count = Counter(
            [name(minimizer) for minimizer in self.minimizers]
        ) # Count the number of each minimizer, they don't have to be unique

        # Note that these are inspect_sig.Parameter's, not symfit parameters!
        parameters = []
        for minimizer in reversed(self.minimizers):
            if count[name(minimizer)] == 1:
                # No ambiguity, so use the name directly.
                param_name = name(minimizer)
            else:
                # Ambiguity, so append the number of remaining minimizers
                param_name = '{}_{}'.format(name(minimizer), count[name(minimizer)])
            count[name(minimizer)] -= 1

            parameters.append(
                inspect_sig.Parameter(
                    param_name,
                    kind=inspect_sig.Parameter.KEYWORD_ONLY,
                    default={}
                )
            )
        return inspect_sig.Signature(parameters=reversed(parameters))


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
            return np.array(func(**parameters))
        return wrapped_func

    def _pack_output(self, ans):
        """
        Packs the output of a minimization in a
        :class:`~symfit.core.fit_results.FitResults`.

        :param ans: The output of a minimization as produced by
            :func:`scipy.optimize.minimize`
        :returns: :class:`~symfit.core.fit_results.FitResults`
        """
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
            ier=ans.nit,
            objective_value=ans.fun,
        )

        if 'hess_inv' in ans:
            try:
                fit_results['hessian_inv'] = ans.hess_inv.todense()
            except AttributeError:
                fit_results['hessian_inv'] = ans.hess_inv
        return FitResults(**fit_results)

    @keywordonly(tol=1e-9)
    def execute(self, bounds=None, jacobian=None, **minimize_options):
        """
        Calls the wrapped algorithm.

        :param bounds: The bounds for the parameters. Usually filled by
            :class:`~symfit.core.minimizers.BoundedMinimizer`.
        :param jacobian: The Jacobian. Usually filled by
            :class:`~symfit.core.minimizers.ScipyGradientMinimize`.
        :param \*\*minimize_options: Further keywords to pass to
            :func:`scipy.optimize.minimize`. Note that your `method` will
            usually be filled by a specific subclass.
        """
        ans = minimize(
            self.wrapped_objective,
            self.initial_guesses,
            bounds=bounds,
            constraints=self.constraints,
            jac=jacobian,
            **minimize_options
        )
        return self._pack_output(ans)

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
    """
    A base class for all :mod:`scipy` based minimizers that use a gradient.
    """
    def __init__(self, *args, **kwargs):
        super(ScipyGradientMinimize, self).__init__(*args, **kwargs)
        self.wrapped_jacobian = self.wrap_func(self.wrapped_jacobian)

    def execute(self, **minimize_options):
        return super(ScipyGradientMinimize, self).execute(jacobian=self.wrapped_jacobian, **minimize_options)


class BFGS(ScipyGradientMinimize):
    """
    A wrapper around :func:`scipy.optimize.minimize` using `method='BFGS'`.
    """
    def execute(self, **minimize_options):
        return super(BFGS, self).execute(method='BFGS', **minimize_options)

class DifferentialEvolution(ScipyMinimize, GlobalMinimizer, BoundedMinimizer):
    """
    A wrapper around :func:`scipy.optimize.differential_evolution`.
    """
    @keywordonly(strategy='rand1bin', popsize=40, mutation=(0.423, 1.053),
                 recombination=0.95, polish=False, init='latinhypercube')
    def execute(self, **de_options):
        ans = differential_evolution(self.wrap_func(self.objective),
                                     self.bounds,
                                     **de_options)
        return self._pack_output(ans)


class SLSQP(ScipyGradientMinimize, ConstrainedMinimizer, BoundedMinimizer):
    """
    A wrapper around :func:`scipy.optimize.minimize` using `method='SLSQP'`.
    """
    def execute(self, **minimize_options):
        return super(SLSQP, self).execute(
            method='SLSQP',
            bounds=self.bounds,
            **minimize_options
        )


class LBFGSB(ScipyGradientMinimize, BoundedMinimizer):
    """
    A wrapper around :func:`scipy.optimize.minimize` using `method='L-BFGS-B'`.
    """
    def execute(self, **minimize_options):
        return super(LBFGSB, self).execute(
            method='L-BFGS-B',
            bounds=self.bounds,
            **minimize_options
        )


class NelderMead(ScipyMinimize, BaseMinimizer):
    """
    A wrapper around :func:`scipy.optimize.minimize` using `method='Nelder-Mead'`.
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
        :param \*\*minpack_options: Any named arguments to be passed to leastsqbound
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
