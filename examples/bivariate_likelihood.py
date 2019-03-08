import numpy as np
from symfit import Variable, Parameter, Fit
from symfit.core.objectives import LogLikelihood
from symfit.distributions import BivariateGaussian

x = Variable('x')
y = Variable('y')
x0 = Parameter('x0', value=0.6, min=0.5, max=0.7)
sig_x = Parameter('sig_x', value=0.1, max=1.0)
y0 = Parameter('y0', value=0.7, min=0.6, max=0.9)
sig_y = Parameter('sig_y', value=0.05, max=1.0)
rho = Parameter('rho', value=0.001, min=-1, max=1)

pdf = BivariateGaussian(x=x, mu_x=x0, sig_x=sig_x, y=y, mu_y=y0,
                       sig_y=sig_y, rho=rho)

# Draw 100000 samples from a bivariate distribution
mean = [0.59, 0.8]
corr = 0.6
cov = np.array([[0.11 ** 2, 0.11 * 0.23 * corr],
                [0.11 * 0.23 * corr, 0.23 ** 2]])
np.random.seed(42)
xdata, ydata = np.random.multivariate_normal(mean, cov, 100000).T

fit = Fit(pdf, x=xdata, y=ydata, objective=LogLikelihood)
fit_result = fit.execute()
print(fit_result)