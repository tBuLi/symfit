from __future__ import division, print_function
from symfit import Parameter, Variable, Fit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.color_palette("Blues")
# sns.set_palette(sns.color_palette("Paired"))

palette = sns.color_palette()
sns.set_palette(palette)
# print sns.color_palette("husl", 8)
a = Parameter()
b = Parameter()
x = Variable()
model = a * x + b

xdata = np.linspace(0, 100, 100) # From 0 to 100 in 100 steps
a_vec = np.random.normal(15.0, scale=2.0, size=(100,))
b_vec = np.random.normal(100, scale=2.0, size=(100,))
ydata = a_vec * xdata + b_vec # Point scattered around the line 5 * x + 105

fit = Fit(model, xdata, ydata)
fit_result = fit.execute()
print(fit_result)

y = model(x=xdata, **fit_result.params)
sns.regplot(xdata, ydata, fit_reg=False)
# plt.plot(xdata, y, color=palette[2])
plt.xlim(0, 100)
# plt.ylim(0, 2000)
plt.show()