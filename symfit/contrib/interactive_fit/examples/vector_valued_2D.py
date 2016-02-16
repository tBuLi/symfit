# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:04:10 2016

@author: peterkroon
"""
from symfit import Variable, Parameter
from symfit.contrib.interactive_fit import InteractiveFit2D
import numpy as np


x = Variable()
y1 = Variable()
y2 = Variable()
k = Parameter(900)
x0 = Parameter(1.5)

model = {y1: k * (x-x0)**2,
         y2: x - x0}
x_data = np.linspace(0, 2.5, 50)
y1_data = model[y1](x=x_data, k=1000, x0=1)
y2_data = model[y2](x=x_data, k=1000, x0=1)
fit = InteractiveFit2D(model, x=x_data, y1=y1_data, y2=y2_data, n_points=250)
fit.visual_guess()
print("Guessed values: ")
for p in fit.model.params:
    print("{}: {}".format(p.name, p.value))
fit_result = fit.execute(maxfev=50)
print(fit_result)