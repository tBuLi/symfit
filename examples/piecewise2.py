from symfit import parameters, variables, Fit, Piecewise, exp, Eq, Model
import numpy as np
import matplotlib.pyplot as plt

x, y = variables('x, y')
a, b, x0 = parameters('a, b, x0')

# Make a piecewise model
y1 = x**2 - a * x
y2 = a * x + b
model = Model({y: Piecewise((y1, x <= x0), (y2, x > x0))})

# As a constraint, we demand equality between the two models at the point x0
# to do this, we substitute x -> x0 and demand equality using `Eq`
constraints = [
    Eq(y1.subs({x: x0}), y2.subs({x: x0}))
]
# Generate example data
xdata = np.linspace(-4, 4., 50)
ydata = model(x=xdata, a=0.0, b=1.0, x0=1.0).y
np.random.seed(2)
ydata = np.random.normal(ydata, 0.5)  # add noise

# Help the fit by bounding the switchpoint between the models
x0.min = 0.8
x0.max = 1.2

fit = Fit(model, x=xdata, y=ydata, constraints=constraints)
fit_result = fit.execute()
print(fit_result)

plt.scatter(xdata, ydata)
plt.plot(xdata, model(x=xdata, **fit_result.params).y)
plt.show()