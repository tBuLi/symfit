from symfit.api import Parameter, Variable, Fit, exp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# sns.color_palette("Blues")
# sns.set_palette(sns.color_palette("Paired"))
palette = sns.color_palette()
sns.set_palette(palette)
# print sns.color_palette("husl", 8)

x = Variable()
A = Parameter()
sig = Parameter(name='sig', value=1.4, min=1.0, max=2.0)
x0 = Parameter(name='x0', value=15.0, min=0.0)
# Gaussian distrubution
model = A*exp(-((x - x0)**2/(2 * sig**2)))

# Sample 10000 points from a N(15.0, 1.5) distrubution
sample = np.random.normal(loc=15.0, scale=1.5, size=(10000,))
ydata, bin_edges = np.histogram(sample, 100)
xdata = (bin_edges[1:] + bin_edges[:-1])/2

fit = Fit(model, xdata, ydata)
fit_result = fit.execute()
print fit_result
print model

y = model(x=xdata, **fit_result.params)
sns.regplot(xdata, ydata, fit_reg=False)
plt.plot(xdata, y, color=palette[2])
plt.ylim(0, 400)
plt.show()