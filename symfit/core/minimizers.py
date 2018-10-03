import abc
import sys
from collections import namedtuple, Counter

from scipy.optimize import minimize, differential_evolution, basinhopping
import sympy
import numpy as np

from .support import key2str, keywordonly, partial
from .leastsqbound import leastsqbound
from .fit_results import FitResults

if sys.version_info >= (3,0):
    import inspect as inspect_sig
    from functools import wraps
else:
    import funcsigs as inspect_sig
    from functools32 import wraps

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
            method=self.method_name(),
            bounds=bounds,
            constraints=constraints,
            jac=jacobian,
            **minimize_options
        )
        return self._pack_output(ans)

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
            ier=ans.nit if hasattr(ans, 'nit') else float('nan'),
            objective_value=ans.fun,
        )

        if 'hess_inv' in ans:
            try:
                fit_results['hessian_inv'] = ans.hess_inv.todense()
            except AttributeError:
                fit_results['hessian_inv'] = ans.hess_inv
        return FitResults(**fit_results)

    @classmethod
    def method_name(cls):
        """
        Returns the name of the minimize method this object represents. This is
        needed because the name of the object is not always exactly what needs
        to be passed on to scipy as a string.
        :return:
        """
        return cls.__name__

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

class DifferentialEvolution(ScipyMinimize, GlobalMinimizer, BoundedMinimizer):
    """
    A wrapper around :func:`scipy.optimize.differential_evolution`.
    """
    @keywordonly(strategy='rand1bin', popsize=40, mutation=(0.423, 1.053),
                 recombination=0.95, polish=False, init='latinhypercube')
    def execute(self, **de_options):
        ans = differential_evolution(self.list2kwargs(self.objective),
                                     self.bounds,
                                     **de_options)
        return self._pack_output(ans)


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

class LBFGSB(ScipyGradientMinimize, BoundedMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s LBFGSB algorithm.
    """
    def execute(self, **minimize_options):
        return super(LBFGSB, self).execute(
            bounds=self.bounds,
            **minimize_options
        )

    @classmethod
    def method_name(cls):
        return "L-BFGS-B"

class NelderMead(ScipyMinimize, BaseMinimizer):
    """
    Wrapper around :func:`scipy.optimize.minimize`'s NelderMead algorithm.
    """
    @classmethod
    def method_name(cls):
        return 'Nelder-Mead'


class BasinHopping(ScipyMinimize, BaseMinimizer):
    """
    Wrapper around :func:`scipy.optimize.basinhopping`'s basin-hopping algorithm.

    As always, the best way to use this algorithm is through
    :class:`~symfit.core.fit.Fit`, as this will automatically select a local
    minimizer for you depending on whether you provided bounds, constraints, etc.

    However, BasinHopping can also be used directly. Example (with jacobian)::

        import numpy as np
        from symfit.core.minimizers import BFGS, BasinHopping
        from symfit import parameters

        def func2d(x1, x2):
            f = np.cos(14.5 * x1 - 0.3) + (x2 + 0.2) * x2 + (x1 + 0.2) * x1
            return f

        def jac2d(x1, x2):
            df = np.zeros(2)
            df[0] = -14.5 * np.sin(14.5 * x1 - 0.3) + 2. * x1 + 0.2
            df[1] = 2. * x2 + 0.2
            return df

        x0 = [1.0, 1.0]
        np.random.seed(555)
        x1, x2 = parameters('x1, x2', value=x0)
        fit = BasinHopping(func2d, [x1, x2], local_minimizer=BFGS)
        minimizer_kwargs = {'jac': fit.list2kwargs(jac2d)}
        fit_result = fit.execute(niter=200, minimizer_kwargs=minimizer_kwargs)

    See :func:`scipy.optimize.basinhopping` for more options.
    """
    @keywordonly(local_minimizer=BFGS)
    def __init__(self, *args, **kwargs):
        """
        :param local_minimizer: minimizer to be used for local minimization
            steps. Can be any subclass of
            :class:`symfit.core.minimizers.ScipyMinimize`.
        :param args: positional arguments to be passed on to `super`.
        :param kwargs: keyword arguments to be passed on to `super`.
        """
        self.local_minimizer = kwargs.pop('local_minimizer')
        super(BasinHopping, self).__init__(*args, **kwargs)

        type_error_msg = 'Currently only subclasses of ScipyMinimize are ' \
                         'supported, since `scipy.optimize.basinhopping` uses ' \
                         '`scipy.optimize.minimize`.'
        # self.local_minimizer has to be a subclass or instance of ScipyMinimize
        # Since no one function exists to test this, we try/except instead.
        try:
            # Test if subclass. If this line doesn't fail, we are dealing with
            # some class. If it fails, we assume that it is an instance.
            issubclass(self.local_minimizer, ScipyMinimize)
        except TypeError:
            # It is not a class at all, so test if it's an instance instead
            if not isinstance(self.local_minimizer, ScipyMinimize):
                # Only ScipyMinimize subclasses supported
                raise TypeError(type_error_msg)
        else:
            if not issubclass(self.local_minimizer, ScipyMinimize):
                # Only ScipyMinimize subclasses supported
                raise TypeError(type_error_msg)
            self.local_minimizer = self.local_minimizer(self.objective, self.parameters)

    def execute(self, **minimize_options):
        """
        Execute the basin-hopping minimization.
        
        :param minimize_options: options to be passed on to
            :func:`scipy.optimize.basinhopping`.
        :return: :class:`symfit.core.fit_results.FitResults`
        """
        if 'minimizer_kwargs' not in minimize_options:
            minimize_options['minimizer_kwargs'] = {}

        if 'method' not in minimize_options['minimizer_kwargs']:
            # If no minimizer was set by the user upon execute, use local_minimizer
            minimize_options['minimizer_kwargs']['method'] = self.local_minimizer.method_name()
        if 'jac' not in minimize_options['minimizer_kwargs'] and isinstance(self.local_minimizer, GradientMinimizer):
            # Assign the jacobian
            minimize_options['minimizer_kwargs']['jac'] = self.local_minimizer.wrapped_jacobian
        if 'constraints' not in minimize_options['minimizer_kwargs'] and isinstance(self.local_minimizer, ConstrainedMinimizer):
            # Assign constraints
            minimize_options['minimizer_kwargs']['constraints'] = self.local_minimizer.wrapped_constraints
        if 'bounds' not in minimize_options['minimizer_kwargs'] and isinstance(self.local_minimizer, BoundedMinimizer):
            # Assign bounds
            minimize_options['minimizer_kwargs']['bounds'] = self.local_minimizer.bounds

        ans = basinhopping(
            self.wrapped_objective,
            self.initial_guesses,
            **minimize_options
        )
        return self._pack_output(ans)


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
