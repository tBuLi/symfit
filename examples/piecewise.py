from symfit import parameters, variables, Fit, Piecewise, exp, Eq, Model
import numpy as np
import matplotlib.pyplot as plt

t, y = variables('t, y')
a, b, d, k, t0 = parameters('a, b, d, k, t0')

# Make a piecewise model
y1 = a * t + b
y2 = d * exp(- k * t)
model = Model({y: Piecewise((y1, t <= t0), (y2, t > t0))})

# As a constraint, we demand equality between the two models at the point t0
# to do this, we substitute t -> t0 and demand equality using `Eq`
constraints = [Eq(y1.diff(t).subs({t: t0}), y2.diff(t).subs({t: t0}))]

# # Generate example data
tdata = np.linspace(0, 4., 200)
ydata = model(t=tdata, a=63, b=300, d=2205, k=3, t0=0.65).y
ydata = np.random.normal(ydata, 0.05 * ydata)  # add 5% noise

# Help the fit by bounding the switchpoint between the models and giving initial
# guesses
t0.min = 0.5
t0.max = 0.8
b.value = 320

fit = Fit(model, t=tdata, y=ydata, constraints=constraints)
fit_result = fit.execute()
print(fit_result)

plt.scatter(tdata, ydata)
plt.plot(tdata, model(t=tdata, **fit_result.params).y)
plt.show()