How Does ``Fit`` Work?
======================

In this section we describe in more detail how :class:`~symfit.core.fit.Fit`
gets from a (named) model and a bunch of data to a fit. We will use
:class:`~symfit.core.fit.NumericalLeastSquares` as example.

Consider the following example::

    from symfit import parameters, variables, Fit

    a, b = parameters('a, b')
    x, y = variables('x, y')
    model = {y: a * x + b}

    fit = Fit(model, x=x_data, y=y_data, sigma_y=sigma_data)
    fit_result = fit.execute()

The first thing :mod:`symfit` does is build :math:`\chi^2` for your model::

    chi_squared = sum((y - f)**2/sigmas[y]**2 for y, f in model.items())

In this line ``sigmas`` is a dict which contains all variables that where given a
value, or returns 1 otherwise.

This :math:`\chi^2` is then transformed into a python function which can then
be used to do the numerical calculations::

    vars, params = seperate_symbols(chi_squared)
    py_chi_squared = lambdify(vars + params, chi_squared)

We are now almost there. Just two steps left. The first is to wrap all the data
into the ``py_chi_squared`` function using :func:`~functoolsiable.partial` into the
function to be optimized::

    from functools import partial

    error = partial(py_chi_squared, **data_per_var)

where ``data_per_var`` is a dict containing variable names: value pairs.

Now all that is left is to minimize ``error`` as a function of the parameters.
:class:`~symfit.core.fit.NumericalLeastSquares` does this by calling
:func:`~symfit.core.leastsqbound.leastsqbound` and have it find the best fit
parameters::

    best_fit_parameters, covariance_matrix = leastsqbound(
        error,
        self.guesses,
        self.eval_jacobian,
        self.bounds,
    )

That's it! Finally there are some steps to generate a
:class:`~symfit.core.fit.FitResults` object, but these are not important for
our current discussion.

