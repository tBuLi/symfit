Quick Start
===========

If you simply want the most important parts about symfit, you came to the right place.

Single Variable Problem
-----------------------
::

  from symfit.api import Parameter, Variable, exp, Fit
  
  A = Parameter(100, min=0)
  b = Parameter()
  x = Variable()
  model = A * exp(x * b)

  xdata = # your 1D xdata. This is a quick start guide, so I'm assuming you know how to get it.
  ydata = # 1D ydata

  fit = Fit(model, xdata, ydata)
  fit_result = fit.execute()

  # Plot the fit.
  # The model *has* to be called by keyword arguments to prevent any ambiguity
  y = model(x=xdata, **fit_result.params)
  plt.plot(xdata, y)
  plt.show()

symfit.api exposes sympy.api
----------------------------

``symfit.api`` exposes the sympy api as well, so mathematical expressions such as ``exp``, ``sin`` and ``pi`` are importable from ``symfit.api`` as well. For more, read the `sympy docs
<http://docs.sympy.org>`_.

Initial Guess
-------------
For fitting to work as desired you should always give a good initial guess for a parameter. 
The ``Parameter`` object can therefore be initiated with the following keywords:

* ``value`` the initial guess value.
* ``min`` Minimal value for the parameter.
* ``max`` Maximal value for the parameter.
* ``fixed`` Fix the value of the parameter during the fitting to ``value``.

In the example above, we might change our ``Parameter``'s to the following after looking at a plot of the data::

  a = Parameter(value=4, min=3, max=6)

Multivariable Problem
---------------------

Let M be the number of variables in your model, and N the number of data point in xdata.
Symfit assumes xdata to be of shape :math:`N \times M` or even :math:`N_1 \times \dots N_i \times M` dimensional, as long as either the first or last axis of the array is of the same length as the number of variables in your model.
Currently it is assumed that the function is not vector valued, meaning that for every datapoint in xdata, only a single y value is returned.
Vector valued functions are on my ToDo list. 

::

  from symfit.api import Parameter, Variable, Fit
  
  a = Parameter()
  b = Parameter()
  x = Variable()
  y = Variable()
  model = a * x**2 + b * y**2

  xdata = # your NxM data.
  ydata = # ydata

  fit = Fit(model, xdata, ydata)
  fit_result = fit.execute()

  # Plot the fit.
  z = model(x=xdata[:, 0] y=xdata[:, 1], **fit_result.params)
  plt.plot(xdata, z)
  plt.show()