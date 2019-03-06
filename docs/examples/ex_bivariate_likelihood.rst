Example: Likelihood fitting a Bivariate Gaussian
================================================

In this example, we shall perform likelihood fitting to a `bivariate normal
distrubtion <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_,
to demonstrate how ``symfit``'s API can easily be used to perform likelihood
fitting on multivariate problems.

.. literalinclude:: ../../examples/bivariate_likelihood.py
    :language: python

This code prints::

    Parameter Value        Standard Deviation
	rho       6.026420e-01 2.013810e-03
	sig_x     1.100898e-01 2.461684e-04
	sig_y     2.303400e-01 5.150556e-04
	x0        5.901317e-01 3.481346e-04
	y0        8.014040e-01 7.283990e-04
	Fitting status message: b'CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH'
	Number of iterations:   35
	Regression Coefficient: nan