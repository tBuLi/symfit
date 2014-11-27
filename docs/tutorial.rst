Tutorial
========

Simple Example
--------------
The example below shows how easy it is to define a model that we could fit to. ::

  from symfit.api import Parameter, Variable
  
  a = Parameter()
  b = Parameter()
  x = Variable()
  model = a * x + b

Lets fit this model to some generated data. ::

  from symfit.api import Fit
  import numpy as np
  
  xdata = np.linspace(0, 100, 100) # From 0 to 100 in 100 steps
  a_vec = np.random.normal(5.0, scale=2.0, size=(100,))
  b_vec = np.random.normal(105.0, scale=30.0, size=(100,))
  ydata = a_vec * xdata + b_vec # Point scattered around the line 5 * x + 105
  
  fit = Fit(model, xdata, ydata)
  fit_result = fit.execute()

Printing ``fit_result`` will give a full report on the values for every parameter, including the uncertainty, and quality of the fit.

Initial Guess
-------------
For fitting to work as desired you should always give a good initial guess for a parameter. 
The ``Parameter`` object can therefore be initiated with the following keywords:

* ``value`` the initial guess value.
* ``min`` Minimal value for the parameter.
* ``max`` Maximal value for the parameter.
* ``fixed`` Fix the value of the parameter during the fitting to ``value``.

In the example above, we might change our ``Parameter``'s to the folling after looking at a plot of the data: ::

  a = Parameter(value=4, min=3, max=6)

Accessing the Results
---------------------
A call to ``Fit.execute()`` returns a ``FitResults`` instance. 
This object holds all information about the fit. 
The fitting process does not modify the ``Parameter`` objects. 
In this example, ``a.value`` will still be ``4.0`` and not the value we obtain after fitting. To get the value of fit paramaters we can do::

  >>> print(fit_result.params.a)
  >>> 5.283944...
  >>> print(fit_result.params.a_stdev)
  >>> 0.3022389...
  >>> print(fit_result.params.b)
  >>> 100.6052...
  >>> print(fit_result.params.b_stdev)
  >>> 0.3022389...

For more FitResults, see the API docs. (Under construction.)

Evaluating the Model
--------------------
With these parameters, we could now evaluate the model with these parameters so we can make a plot of it.
In order to do this, we simply call the model with these values::

  import matplotlib.pyplot as plt
  
  y = model(x=xdata, a=fit_result.params.a, b=fit_result.params.b)
  plt.plot(xdata, y)
  plt.show()
  
The model **has** to be called by keyword arguments to prevent any ambiguity. So the following does not work::
  y = model(xdata, fit_result.params.a, fit_result.params.b)
  
To make life easier, there is a nice shorthand notation to imidiately use a fit result::
  y = model(x=xdata, **fit_result.params)
  
This unpacks the .params object as a dict. For more info view ParameterDict.
  
