Internal API Structure
======================

Here we describe how the code is organized internally. This is only really
relevant for advanced users and developers.

Fitting 101
-----------

Fitting a model to data is, at it's most basic, a parameter optimisation, and
depending on whether you do a least-squares fit or a loglikelihood fit your
objective function changes. This means we can split the process of fitting in
three distint, isolated parts:: the :class:`~symfit.core.fit.Model`, the
Objective and the Minimizer. 

In practice, :class:`~symfit.core.fit.Fit` will choose an appropriate objective
and minimizer, but you can also give it specific instances and classes; just in
case you know better.

For both the minimizers and objectives there are abstract base classes, which
describe the minimal API required. There are corresponding abstract classes for
e.g. :class:`~symfit.core.minimizers.ConstrainedMinimizer`.

Objectives
----------

Objectives wrap both the Model and the data supplied, and when called must
return a scalar. This scalar will be *minimized*, so when you need something
maximized, be sure to add a negation in the right place(s). They must be
called with the parameter values as keyword arguments. Be sure to inherit from
the abstract base class(es) so you're sure you define all the methods that are
expected.

Minimizers
----------

Minimizers minimize. They are provided with a function to minimize (the
objective) and the :class:`~symfit.core.argument.Parameter` s as a function of which the objective should be
minimized. Note that once again there are different base classes for minimizers
that take e.g. bounds or support gradients. Their
:meth:`~symfit.core.minimizers.BaseMinimizer.execute` method takes the
metaparameters for the minimization. Again, be sure to inherit from the
appropriate base class(es) if you're implementing your own minimizer to make
sure all the expected methods are there. And if you're wrapping Scipy style
minimizers, have a look at :class:`~symfit.core.minimizers.ScipyMinimize` to
avoid a duplication of efforts.




