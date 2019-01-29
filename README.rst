.. image:: https://zenodo.org/badge/24005390.svg
   :target: https://zenodo.org/badge/latestdoi/24005390
.. image:: https://coveralls.io/repos/github/tBuLi/symfit/badge.svg?branch=master
   :target: https://coveralls.io/github/tBuLi/symfit?branch=master


Please cite this DOI if ``symfit`` benefited your publication. Building this has been a lot of work, and as young researchers your citation means a lot to us.
Martin Roelfs & Peter C Kroon, symfit. doi:10.5281/zenodo.1133336

Documentation
=============
http://symfit.readthedocs.org

Project Goals
=============

The goal of this project is simple: to make fitting in Python pythonic.
What does pythonic fitting look like? Well, there's a simple test. If I can
give you pieces of example code and don't have to use any additional words to
explain what it does, it's pythonic.

.. code-block:: python

  from symfit import parameters, variables, Fit, Model
  import numpy as np
   
  xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
  ydata = np.array([2.3, 3.3, 4.1, 5.5, 6.7])
  yerr = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
  
  a, b = parameters('a, b')
  x, y = variables('x, y')
  model = Model({y: a * x + b})
  
  fit = Fit(model, x=xdata, y=ydata, sigma_y=yerr)
  fit_result = fit.execute()

Cool right? So now that we have done a fit, how do we use the results?

.. code-block:: python

  import matplotlib.pyplot as plt
  
  yfit = model(x=xdata, **fit_result.params)[y]
  plt.plot(xdata, yfit)
  plt.show()

.. figure:: http://symfit.readthedocs.org/en/latest/_images/linear_model_fit.png
  :width: 600px
  :alt: Linear Fit

Need I say more? How about I let another code example do the talking?

.. code-block:: python

  from symfit import parameters, Fit, Equality, GreaterThan
  
  x, y = parameters('x, y')
  model = 2 * x * y + 2 * x - x**2 - 2 * y**2
  constraints = [
      Equality(x**3, y),
      GreaterThan(y, 1),
  ]
  
  fit = Fit(- model, constraints=constraints)
  fit_result = fit.execute()

I know what you are thinking. "What if I need to fit to a system of Ordinary Differential Equations?"

.. code-block:: python

  from symfit import variables, Parameter, ODEModel, Fit, D
  import numpy as np
  
  tdata = np.array([10, 26, 44, 70, 120])
  adata = 10e-4 * np.array([44, 34, 27, 20, 14])
          
  a, b, t = variables('a, b, t')
  k = Parameter('k', 0.1)
  
  model_dict = {
      D(a, t): - k * a**2,
      D(b, t): k * a**2,
  }
  
  ode_model = ODEModel(model_dict, initial={t: 0.0, a: 54 * 10e-4, b: 0.0})
  
  fit = Fit(ode_model, t=tdata, a=adata, b=None)
  fit_result = fit.execute()

For more fitting delight, check the docs at http://symfit.readthedocs.org.
