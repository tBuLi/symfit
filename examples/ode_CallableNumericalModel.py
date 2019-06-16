from symfit import variables, Parameter, Fit, D, ODEModel, CallableNumericalModel
import numpy as np
import matplotlib.pyplot as plt

def fun(ode_model):
    def fun_(x, k, k2):
        return 2.0 * ode_model(x=x, k=k).y + k2
    return fun_

x_data = np.linspace(0.0, 10.0, 1000)
k_expected = 0.6
k1_expected = 10.0
y_data = 2 * np.exp(k_expected * x_data) + k1_expected

y, x, z = variables('y, x, z')
k = Parameter('k', 0.0)
k2 = Parameter('k2', 0.0)

ode_model = ODEModel({D(y, x): k * y}, initial={x: 0.0, y: 1.0})
model = CallableNumericalModel({z: fun(ode_model)},independent_vars=[x], params=[k,k2])

fit = Fit(model,x=x_data, z=y_data)
fit_result = fit.execute()

print(fit_result)
