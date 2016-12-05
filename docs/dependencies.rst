Dependencies and Credits
========================

Always pay credit where credit's due. ``symfit`` uses the following projects to
make it's sexy interface possible:

- `leastsqbound-scipy <https://github.com/jjhelmus/leastsqbound-scipy>`_ is
  used to bound parameters to a given domain.
- `seaborn <http://seaborn.pydata.org>`_ was used to make the beautifully
  styled plots in the example code. All you have to do to sexify your
  matplotlib plot's is import seaborn, even if you don't use it's special
  plotting facilities, so I highly recommend it.
- `numpy and scipy <https://docs.scipy.org/doc/>`_ are of course used to do
  efficient data processing.
- `sympy <http://docs.sympy.org/latest/index.html>`_ is used for the
  manipulation of the symbolic expressions that give this project it's
  high readability.


.. the seaborn images in this documentation were made with the settings that
  can be found in the gaussian example::
   import matplotlib.pyplot as plt
   import seaborn as sns
   palette = sns.color_palette()
   sns.regplot(xdata, ydata, label='data', fit_reg=False)
   plt.plot(xdata, model(xdata, **fit_results), label='fit', color=palette[2])
   plt.legend()
   plt.show()
