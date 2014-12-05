Introduction
============

Existing fitting modules are not very pythonic in their API and can be difficult for humans to use. This project aims to marry the power of ``scipy.optimize`` with the readability of ``SymPy`` to create a highly readable and easy to use fitting package which works for projects of any scale.

``symfit`` makes it extremely easy to provide guesses for your parameter and to bound them to a certain range::
	a = Parameter(1.0, min=0.0, max=5.0)

To define models to fit to::
	x = Variable()
	A = Parameter()
	sig = Parameter(1.0, min=0.0, max=5.0)
	x0 = Parameter(1.0, min=0.0)
	# Gaussian distrubution
	model = exp(-(x - x0)**2/(2 * sig**2))

And finally, to execute the fit::
	fit = Fit(model, xdata, ydata)
	fit_result = fit.execute()

And to evaluate the model using the best fit parameters::
	y = model(x=xdata, **fit_result.params)

.. figure:: _static/gaussian_intro.png
   :width: 500px
   :alt: Gaussian Data

For the full code to this or other examples, check the example library here: :ref:`example-library`.

Technical Reasons
-----------------
On a more technical note, this symbolic approach turns out to have great technical advantages over using scipy directly. In order to fit, the algorithm needs the Jacobian: a matrix containing the derivatives of your model in it's parameters. Because of the symbolic nature of ``symfit``, this is determined for you on the fly, saving you the trouble of having to determine the derivatives yourself. Furthermore, having this Jacobian allows good estimation of the errors in your parameters, something ``scipy`` does not always succeed in.

