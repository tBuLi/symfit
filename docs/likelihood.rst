On Likelihood Fitting
=====================

The :class:`~symfit.core.objectives.LogLikelihood` objective function should be
used to perform log-likelihood maximization. The
:meth:`~symfit.core.objectives.LogLikelihood.__call__`
and :meth:`~symfit.core.objectives.LogLikelihood.eval_jacobian` definitions have
been changed to facilitate what one would expect from Likelihood fitting:

`__call__` gives the value of log-likelihood at the given values of
:math:`\vec{p}` and :math:`\vec{x}_i`, where :math:`\vec{p}` is a shorthand
notation for all parameter, and :math:`\vec{x}_i` the same shorthand for all
independent variables.

.. math:: \log{L(\vec{p}|\vec{x}_i)} = \sum_{i=1}^{N} \log{f(\vec{p}|\vec{x}_i)}

:meth:`~symfit.core.objectives.LogLikelihood.eval_jacobian` gives the derivative
with respect to every parameter of the log-likelihood:

.. math:: \nabla_{\vec{p}} \log{L(\vec{p}|\vec{x}_i)} = \sum_{i=1}^{N}
   \frac{1}{f(\vec{p}|\vec{x}_i)} \nabla_{\vec{p}} f(\vec{p}|\vec{x}_i)

Where :math:`\nabla_{\vec{p}}` is the derivative with respect to all parameters
:math:`\vec{p}`. The function therefore returns a vector of length :code:`len(p)`
containing the Jacobian evaluated at the given values of :math:`\vec{p}` and
:math:`\vec{x}`.
