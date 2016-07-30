Style Guide & Best Practices
============================

Style Guide
-----------

Anything Raymond Hettinger says wins the argument until I have time to write a
proper style guide.

Best Practices
--------------

* It is recommended to always use named models. So not::

    model = a * x**2
    fit = Fit(model, xdata, ydata)

  but::

    model = {y: a * x**2}
    fit = Fit(model, x=xdata, y=ydata)

  In this simple example the two are equivalent but for multidimentional data
  using ordered arguments can become ambiguous and it even appears there is a
  difference in interpretation between py2 and py3. To prevent such ambiguity
  and to make sure your code is transportable, always use named models. And the
  result is more readable anyway right?

* When evaluating models use tuple-unpacking::

    model = {y_1: x**2, y_2: x**3}
    sol_1, sol_2 = model(x=xdata)

  and not::

    sol_1 = model(x=xdata)[0]

  or something similar.