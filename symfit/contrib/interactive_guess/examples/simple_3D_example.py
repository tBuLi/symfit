# -*- coding: utf-8 -*-

from symfit import variables, Parameter, exp, Fit, Model
from symfit.distributions import Gaussian
from symfit.contrib.interactive_guess import InteractiveGuess
import numpy as np


x, y, z = variables('x, y, z')
mu_x = Parameter('mu_x', 10)
mu_y = Parameter('mu_y', 10)
sig_x = Parameter('sig_x', 1)
sig_y = Parameter('sig_y', 1)



model = Model({z: Gaussian(x, mu_x, sig_x) * Gaussian(y, mu_y, sig_y)})
x_data = np.linspace(0, 25, 50)
y_data = np.linspace(0, 25, 50)
x_data, y_data = np.meshgrid(x_data, y_data)
x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = model(x=x_data, y=y_data, mu_x=5, sig_x=0.3, mu_y=10, sig_y=1).z

guess = InteractiveGuess(model, x=x_data, y=y_data, z=z_data)
guess.execute()
print(guess)

fit = Fit(model, x=x_data, y=y_data, z=z_data)
fit_result = fit.execute()
print(fit_result)
