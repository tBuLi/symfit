# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:28:58 2015

@author: peterkroon
"""
from symfit import Variable, Parameter, exp, Fit, Model
from symfit.contrib.interactive_guess import InteractiveGuess
import numpy as np


def distr(x, k, x0):
    kbT = 4.11
    return exp(-k*(x-x0)**2/kbT)


x = Variable('x')
y = Variable('y')
k = Parameter('k', 900)
x0 = Parameter('x0', 1.5)

model = Model({y: distr(x, k, x0)})
x_data = np.linspace(0, 2.5, 50)
y_data = model(x=x_data, k=1000, x0=1).y

guess = InteractiveGuess(model, x=x_data, y=y_data, n_points=150)
guess.execute()
print(guess)

fit = Fit(model, x=x_data, y=y_data)
fit_result = fit.execute(maxiter=1000)
print(fit_result)
