"""
A minimal example of global fitting in symfit.
Two datasets are first generated from the same function.

.. math::

    f(x) = y_0 + a * e^{- b * x}

Dataset one and two will share the parameters :math:`a` and :math:`b`.
All dataset will share the parameter :math:`y_0`, which measures the background.
"""

import numpy as np
from symfit import *
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette()

x, y_1, y_2 = variables('x, y_1, y_2')
y0, a_1, a_2, b_1, b_2 = parameters('y0, a_1, a_2, b_1, b_2')

# The following vector valued function links all the equations together
# as stated in the intro.
model = Model({
    y_1: y0 + a_1 * exp(- b_1 * x),
    y_2: y0 + a_2 * exp(- b_2 * x),
})

# Generate data from this model
xdata = np.linspace(0, 10)
ydata1, ydata2 = model(x=xdata, a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
# Add some noise to make it appear like real data
ydata1 += np.random.normal(0, 2, size=ydata1.shape)
ydata2 += np.random.normal(0, 2, size=ydata2.shape)
ydata = [ydata1, ydata2]

# Guesses
a_1.value = 100
a_2.value = 50
b_1.value = 1
b_2.value = 1
y0.value = 10


fit = Fit(model, x=xdata, y_1=ydata1, y_2=ydata2)
fit_result = fit.execute()
fit_curves = model(x=xdata, **fit_result.params)
print(fit_result)

for data, curve, color in zip(ydata, fit_curves, palette):
    plt.plot(xdata, curve, color=color, alpha=0.5)
    plt.plot(xdata, data, color=color)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Global Fitting, MWE')
plt.show()