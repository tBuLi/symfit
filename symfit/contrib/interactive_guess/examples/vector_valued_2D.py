# -*- coding: utf-8 -*-
from symfit import Variable, Parameter, Fit, Model
from symfit.contrib.interactive_guess import InteractiveGuess
import numpy as np


x = Variable('x')
y1 = Variable('y1')
y2 = Variable('y2')
k = Parameter('k', 900)
x0 = Parameter('x0', 1.5)

model = {
    y1: k * (x-x0)**2,
    y2: x - x0
}
model = Model(model)

# Generate example data
x_data = np.linspace(0, 2.5, 50)
data = model(x=x_data, k=1000, x0=1)
y1_data = data.y1
y2_data = data.y2

guess = InteractiveGuess(model, x=x_data, y1=y1_data, y2=y2_data, n_points=250)
guess.execute()
print(guess)

fit = Fit(model, x=x_data, y1=y1_data, y2=y2_data)
fit_result = fit.execute()
print(fit_result)
