import numpy as np
from symfit import variables, parameters, Fit, exp, Model
from symfit.core.objectives import LogLikelihood

# Draw samples from a bivariate distribution
np.random.seed(42)
data1 = np.random.exponential(5.5, 1000)
data2 = np.random.exponential(6, 2000)

# Define the model for an exponential distribution (numpy style)
a, b = parameters('a, b')
x1, y1, x2, y2 = variables('x1, y1, x2, y2')
model = Model({
    y1: (1 / a) * exp(-x1 / a),
    y2: (1 / b) * exp(-x2 / b)
})
print(model)

fit = Fit(model, x1=data1, x2=data2, objective=LogLikelihood)
fit_result = fit.execute()
print(fit_result)

# Instead, we could also fit with only one parameter to see which works best
model = Model({
    y1: (1 / a) * exp(-x1 / a),
    y2: (1 / a) * exp(-x2 / a)
})

fit = Fit(model, x1=data1, x2=data2, objective=LogLikelihood)
fit_result = fit.execute()
print(fit_result)