On Standard Deviations
======================

This essay is meant as a reflection on the implementation of Standard Deviations
and/or measurement errors in :mod:`symfit`. Although reading this essay in it's
entirely will only be interesting to a select few, I urge anyone who uses
:mod:`symfit` to read the following summarizing bullet points, as :mod:`symfit`
is **not** backward-compatible with :mod:`scipy`.

* standard deviations are assumed to be measurement errors by default, not
  relative weights. This is the opposite of the :mod:`scipy` definition. Set
  ``absolute_sigma=False`` when calling :class:`~symfit.core.fit.Fit` to get
  the :mod:`scipy` behavior.


Analytical Example
------------------

The implementation of standard deviations should be in agreement with cases to
which the analytical solution is known. :mod:`symfit` was build such that this
is true. Let's follow the example outlined by [taldcroft]_. We'll be sampling
from a normal distribution with :math:`\mu = 0.0` and varying :math:`\sigma`.
It can be shown that given a sample from such a distribution:

.. math:: \mu = 0.0
.. math:: \sigma_{\mu} = \frac{\sigma}{\sqrt{N}}

where N is the size of the sample. We see that the error in the sample mean
scales with the :math:`\sigma` of the distribution.

In order to reproduce this with :mod:`symfit`, we recognize that determining
the avarage of a set of numbers is the same as fitting to a constant. Therefore
we will fit to samples generated from distributions with :math:`\sigma = 1.0`
and :math:`\sigma = 10.0` and check if this matches the analytical values.
Let's set :math:`N = 10000`.
::

    N = 10000
    sigma = 10.0
    np.random.seed(10)
    yn = np.random.normal(size=N, scale=sigma)

    a = Parameter('a')
    y = Variable('y')
    model = {y: a}

    fit = Fit(model, y=yn, sigma_y=sigma)
    fit_result = fit.execute()

    fit_no_sigma = Fit(model, y=yn)
    fit_result_no_sigma = fit_no_sigma.execute()

This gives the following results:

* a = 5.102056e-02 |+-| 1.000000e-01 when ``sigma_y`` is provided. This matches
  the analytical prediction.
* a = 5.102056e-02 |+-| 9.897135e-02 without ``sigma_y`` provided. This is
  incorrect.

If we run the above code example with ``sigma = 1.0``, we get the following
results:

* a = 5.102056e-03 |+-| 9.897135e-03 when ``sigma_y`` is provided. This matches
  the analytical prediction.
* a = 5.102056e-03 |+-| 9.897135e-03 without ``sigma_y`` provided. This is also
  correct, since providing no weights is the same as setting the weights to 1.

To conclude, if :mod:`symfit` is provided with the standard deviations, it will
give the expected result by default. As shown in [taldcroft]_ and
:mod:`symfit`'s tests, :func:`scipy.optimize.curve_fit` has to be provided with
the ``absolute_sigma=True`` setting to do the same.

.. important::
  We see that even if the weight provided to every data point is the same, the
  *scale* of the weight still effects the result. :mod:`scipy` was build such
  that the opposite is true: if all datapoints have the same weight, the error
  in the parameters does not depend on the scale of the weight.

  This difference is due to the fact that :mod:`symfit` is build for areas of
  science where one is dealing with measurement errors. And with measurement
  errors, the size of the errors obviously matters for the certainty of the fit
  parameters, even if the errors are the same for every measurement.

  If you want the :mod:`scipy` behavior, initiate :class:`~symfit.core.fit.Fit`
  with ``absolute_sigma=False``.

Comparison to Mathematica
=========================

In Mathematica, the default setting is also to use relative weights, which we
just argued is not correct when dealing with measurement errors. In [Mathematica]_
this problem is discussed very nicely, and it is shown how to solve this in
Mathematica.

Since :mod:`symfit` is a fitting tool for the practical man, measurement errors
are assumed by default.

.. [taldcroft] http://nbviewer.jupyter.org/urls/gist.github.com/taldcroft/5014170/raw/31e29e235407e4913dc0ec403af7ed524372b612/curve_fit.ipynb
.. [Mathematica] http://reference.wolfram.com/language/howto/FitModelsWithMeasurementErrors.html
.. |+-| unicode:: U+00B1 .. Plusminus sign
