# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:28:58 2015

@author: peterkroon
"""
from symfit import Variable, Parameter, exp
from symfit.contrib.interactive_fit import interactive_fit
import numpy as np


def distr(x, k, x0):
    kbT = 4.11
    return exp(-k*(x-x0)**2/kbT)


x = Variable()
y = Variable()
k = Parameter(900)
x0 = Parameter(1.5)

model = {y: distr(x, k, x0)}
x_data = np.linspace(0, 2.5, 50)
y_data = model[y](x=x_data, k=1000, x0=1)
fit = interactive_fit.InteractiveFit2D(model, x=x_data, y=y_data)
fit.visual_guess(250)
print("Guessed values: ")
for p in fit.model.params:
    print("{}: {}".format(p.name, p.value))
fit_result = fit.execute(maxfev=1000)
print(fit_result)
