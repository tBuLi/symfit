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
  using ordered arguments can become ambiguous and difficult to read. To
  increase readability, it is therefore recommended to always use named models.

* Evaluating a (vector valued) model returns a :func:`~collections.namedtuple`.
  You can access the elements by either tuple unpacking, or by using the
  variable names. Note that if you use tuple unpacking, the results will be
  ordered alphabetically. The following::

    model = Model({y_1: x**2, y_2: x**3})
    sol_1, sol_2 = model(x=xdata)

  is therefore equivalent to::

    model = Model({y_1: x**2, y_2: x**3})
    solutions = model(x=xdata)
    sol_1 = solutions.y_1
    sol_2 = solutions.y_2

  Using numerical indexing (or something similar) is not recommended as it is
  less readable than the options given above::

    sol_1 = model(x=xdata)[0]
