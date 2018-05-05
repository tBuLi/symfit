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
objective) and the :class:`~symfit.core.argument.Parameter` s as a function of
which the objective should be minimized. Note that once again there are
different base classes for minimizers that take e.g. bounds or support
gradients. Their :meth:`~symfit.core.minimizers.BaseMinimizer.execute` method
takes the metaparameters for the minimization. Again, be sure to inherit from
the appropriate base class(es) if you're implementing your own minimizer to
make sure all the expected methods are there. And if you're wrapping Scipy
style minimizers, have a look at :class:`~symfit.core.minimizers.ScipyMinimize`
to avoid a duplication of efforts.

Example
-------

Let's say we have some data::

    xdata = np.linspace(0, 100, 25)
    a_vec = np.random.normal(15, scale=2, size=xdata.shape)
    b_vec = np.random.normal(100, scale=2, size=xdata.shape)
    ydata = a_vec * xdata + b_vec

And we want to fit it to some model::

    a = Parameter('a', value=0, min=0, max=1000)
    b = Parameter('b', value=0, min=0, max=1000)
    x = Variable('x')
    model = a * x + b

If we want to fit this normally (but with a specified minimizer), we'd write
the following::

    fit = Fit(mode, xdata, ydata, minimizer=BFGS)
    fit_result = fit.execute()

Now instead, we want to call the minimizer directly. We first define a custom
objective function (actually just a chi squared)::

    def f(x, a, b):
        return a * x + b

    def chi_squared(a, b):
        return np.sum((ydata - f(xdata, a, b))**2)

    custom_minimize = BFGS(chi_squared, [a, b])
    custom_minimize.execute()

You'll see that the result of both will be the same!

