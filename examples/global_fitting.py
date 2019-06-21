"""
A minimal example of global fitting in symfit.
Two datasets are first generated from the same function.

.. math::

    f(x) = a * x^2 + b * x + y_0

All dataset will share the parameter :math:`y_0`, which measures the background,
but :math:`a` and :math:`b` will be unique for each. Additionally, dataset 2
will contain less datapoints than 1 to demonstrate that this will still work.
"""

import numpy as np
from symfit import *
from symfit.core.support import *
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette()

x_1, x_2, y_1, y_2 = variables('x_1, x_2, y_1, y_2')
y0, a_1, a_2, b_1, b_2 = parameters('y0, a_1, a_2, b_1, b_2')

# The following vector valued function links all the equations together
# as stated in the intro.
model = Model({
    y_1: a_1 * x_1**2 + b_1 * x_1 + y0,
    y_2: a_2 * x_2**2 + b_2 * x_2 + y0,
})

# Generate data from this model
xdata1 = np.linspace(0, 10)
xdata2 = xdata1[::2] # Only every other point.

ydata1, ydata2 = model(x_1=xdata1, x_2=xdata2, a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
# Add some noise to make it appear like real data
np.random.seed(1)
ydata1 += np.random.normal(0, 2, size=ydata1.shape)
ydata2 += np.random.normal(0, 2, size=ydata2.shape)

xdata = [xdata1, xdata2]
ydata = [ydata1, ydata2]

# Guesses
a_1.value = 100
a_2.value = 50
b_1.value = 1
b_2.value = 1
y0.value = 10

sigma_y = np.concatenate((np.ones(20), [2., 4., 5, 7, 3]))

fit = Fit(
    model, x_1=xdata[0], x_2=xdata[1], y_1=ydata[0], y_2=ydata[1], sigma_y_2=sigma_y
)
fit_result = fit.execute()
print(fit_result)
fit_curves = model(x_1=xdata[0], x_2=xdata[1], **fit_result.params)

for xd, yd, curve, color in zip(xdata, ydata, fit_curves, palette):
    plt.plot(xd, curve, color=color, alpha=0.5)
    plt.scatter(xd, yd, color=color)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Global Fitting, MWE')
plt.show()