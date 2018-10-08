========
Examples
========

Model Examples
--------------

These are examples of the flexibility of :mod:`symfit` Models. This is because
essentially any valid :mod:`sympy` code can be provided as a model. This makes
it very intuitive to define your mathematical models almost as you would on
paper.

.. toctree::
    :maxdepth: 1

    ex_fourier_series
    ex_piecewise
    ex_poly_surface_fit
    ex_ODEModel
    ex_CallableNumericalModel

Interactive Guess Module
------------------------

The :mod:`symfit.contrib.interactive_guess` contrib module was designed to make
the process of finding initial guesses easier, by presenting the user with an
interactive :mod:`matplotlib` window in which they can play around with the
initial values.

.. toctree::
    :maxdepth: 1

    ex_interactive_guesses_ODE
    ex_interactive_guesses_vector_2D