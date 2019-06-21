Internal API Structure
======================

Here we describe how the code is organized internally. This is only really
relevant for advanced users and developers.

Fitting 101
-----------

Fitting a model to data is, at it's most basic, a parameter optimisation, and
depending on whether you do a least-squares fit or a loglikelihood fit your
objective function changes. This means we can split the process of fitting in
three distinct, isolated parts: the :class:`~symfit.core.models.Model`, the
Objective and the Minimizer. 

In practice, :class:`~symfit.core.fit.Fit` will choose an appropriate objective
and minimizer on the basis of the model and the data, but you can also give it
specific instances and classes; just in case you know better.

For both the minimizers and objectives there are abstract base classes, which
describe the minimal API required. If a minimizer is more specific, e.g. it
supports constraints, then there are corresponding abstract classes for that,
e.g. :class:`~symfit.core.minimizers.ConstrainedMinimizer`.

Models
------

Models house the mathematical definition of the model we want to use to fit.
For the typical usecase in :mod:`symfit` these are fully symbolical, and
therefore a lot of their properties can be inspected automatically.

As a basic quality, all models are callable, i.e. they have implemented
``__call__``. This is used to numerically evaluate the model given the
parameters and independent variables. In order to make sure you get all the
basic functionality, always inherit from :class:`~symfit.core.models.BaseModel`.

Next level up, if they inherit from :class:`~symfit.core.models.GradientModel`
then they will have ``eval_jacobian``, which will numerically evaluate the
jacobian of the model. Lastly, if they inherit from
:class:`~symfit.core.models.HessianModel`, they will also have ``eval_hessian``
to evaluate the hessian of the model. The standard
:class:`~symfit.core.models.Model` is all of the above.

Odd ones out from the current library are
:class:`~symfit.core.models.CallableNumericalModel` and
:class:`~symfit.core.models.ODEModel`. They only inherit from
:class:`~symfit.core.models.BaseModel` and are therefore callable,
but their other behaviors are custom build.

Since :mod:`symfit` ``0.5.0``, the core of the model has been improved
significantly. At the center of these improvements is
:attr:`~symfit.core.models.BaseModel.connectivity_mapping`. This mapping
represent the connectivity matrix of the variables and parameters, and therefore
encodes which variable depends on which. This is used in ``__call__`` to
evaluate the components in order. To help with this, models have
:attr:`~symfit.core.models.BaseModel.ordered_symbols`. This property is the
topologically sorted ``connectivity_mapping``, and dictates the order in which
variables have to be evaluated.

Objectives
----------

Objectives wrap both the Model and the data supplied, and expose only the free
parameters of the model to the outside world.
When called they must return a scalar. This scalar will be *minimized*, so when
you need something maximized, be sure to add a negation in the right place(s).
They can be called by using the parameter names as keyword arguments, or with a
list of parameter values in the same order as
:attr:`~symfit.core.models.BaseModel.free_params` (alphabetical).
The latter is there because this is how ``scipy`` likes it.
Be sure to inherit from the abstract base class(es) so you're sure you define
all the methods that are expected of an objective. Similar to the models, they
come in three types: :class:`~symfit.core.objectives.BaseObjective`,
:class:`~symfit.core.objectives.GradientObjective` and
:class:`~symfit.core.objectives.HessianObjective`. These must implement
``__call__``, ``eval_jacobian`` and ``eval_hessian`` respectively.

When defining a new objective, it is best to inherit from
:class:`~symfit.core.objectives.HessianObjective` and to define all three if
possible. When feeding a model that does not implement ``eval_hessian`` to a
:class:`~symfit.core.objectives.HessianObjective` no puppies die,
:class:`~symfit.core.fit.Fit` is clever enough to prevent this.

Minimizers
----------

Last in the chain are the minimizers. They are provided with a function to
minimize (the objective) and the :class:`~symfit.core.argument.Parameter` s as
a function of which the objective should be minimized. Note that once again
there are different base classes for minimizers that take e.g. bounds or
support gradients. Their :meth:`~symfit.core.minimizers.BaseMinimizer.execute`
method takes the metaparameters for the minimization.
Again, be sure to inherit from the appropriate base class(es) if you're
implementing your own minimizer to make sure all the expected methods are there.
:class:`~symfit.core.fit.Fit` depends on this to make its decisions.
And if you're wrapping Scipy style minimizers, have a look at
:class:`~symfit.core.minimizers.ScipyMinimize`
to avoid a duplication of efforts.

Minimizers must always implement a method ``execute``, which will return an
instance of :class:`~symfit.core.fit_results.FitResults`. Any ``*args`` and
``**kwargs`` given to execute must
be passed to the underlying minimizer.

Fit
---

:class:`~symfit.core.fit.Fit` is responsible for stringing all of the above
together intelligently.
When not coached into the right direction, it will decide which minimizer and
objective to use on the basis of the model and data.