from __future__ import division, print_function
from symfit import Parameter, Variable, Fit
from symfit.core.fit import TakesData
from symfit.core.minimizers import *
from symfit.core.objectives import *
from symfit.distributions import Exp
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.color_palette("Blues")
# sns.set_palette(sns.color_palette("Paired"))

# palette = sns.color_palette()
# sns.set_palette(palette)
# print sns.color_palette("husl", 8)
a = Parameter(value=0, min=0.0, max=1000)
b = Parameter(value=0, min=0.0, max=1000)
x = Variable()
model = a * x + b

xdata = np.linspace(0, 100, 25) # From 0 to 100 in 100 steps
a_vec = np.random.normal(15.0, scale=2.0, size=xdata.shape)
b_vec = np.random.normal(100, scale=2.0, size=xdata.shape)
ydata = a_vec * xdata + b_vec # Point scattered around the line 5 * x + 105
print(SLSQP.__mro__)
fit = Fit(model, xdata, ydata, minimizer=MINPACK)
fit_result = fit.execute()
print(fit_result)

y = model(x=xdata, **fit_result.params)
# # sns.regplot(xdata, ydata, fit_reg=False)
# plt.plot(xdata, y)
# plt.scatter(xdata, ydata, c='r')
# # plt.xlim(0, 100)
# # plt.ylim(0, 2000)
# plt.show()