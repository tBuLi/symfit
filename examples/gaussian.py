import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from symfit import Parameter, Variable, Fit, GradientModel
from symfit.distributions import Gaussian

palette = sns.color_palette()


x = Variable('x')
y = Variable('y')
A = Parameter('A')
sig = Parameter(name='sig', value=1.4, min=1.0, max=2.0)
x0 = Parameter(name='x0', value=15.0, min=0.0)

# Gaussian distribution
model = GradientModel({y: A*Gaussian(x, x0, sig)})

# Sample 10000 points from a N(15.0, 1.5) distrubution
np.random.seed(seed=123456789)
sample = np.random.normal(loc=15.0, scale=1.5, size=(10000,))
ydata, bin_edges = np.histogram(sample, 100)
xdata = (bin_edges[1:] + bin_edges[:-1])/2

fit = Fit(model, xdata, ydata)
fit_result = fit.execute()

y, = model(x=xdata, **fit_result.params)
sns.regplot(xdata, ydata, fit_reg=False)
plt.plot(xdata, y, color=palette[2])
plt.ylim(0, 400)
plt.show()
