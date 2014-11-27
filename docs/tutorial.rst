Tutorial
========

Simple Example
--------------
The example below shows how easy it is to define a model that we could fit to.::

from symfit.api import Parameter, Variable

a = Parameter()
b = Parameter()
x = Variable()
model = a * x + b

Lets fit this model to some generated data.

``
from symfit.api import Fit
import numpy as np

xdata = np.linspace(0, 100, 100) # From 0 to 100 in 100 steps
a_vec = np.random.normal(5.0, scale=2.0, size=(100,))
b_vec = np.random.normal(105.0, scale=30.0, size=(100,))
ydata = a_vec * xdata + b_vec # Point scattered around the line 5 * x + 105

fit = Fit(model, xdata, ydata)
fit_result = fit.execute()
``
Printing ``fit_result`` will give a full report on the values for every parameter, including the uncertainty, and quality of the fit.

Guess Parameters
----------------
For fitting to work as desired you should always give a good initial guess for a parameter. The ``Parameter`` object can therefore be initiated with the following keywords:
* ``value`` the initial guess value
* ``min`` Minimal value for the parameter.
* ``max`` Maximal value for the parameter.
* ``fixed`` Fix the value of the parameter during the fitting to ``value``.
In the example above, we might change our ``Parameter``'s to the folling after looking at a plot of the data:
``
a = Parameter(value=5, min=4, max=6)
``
Accesing the Results
--------------------
