from symfit import variables, parameters, Fit, D, ODEModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example of the easy of use of the symfit ODE integration syntax.
a, b, c, d, t = variables('a, b, c, d, t')
k, p, l, m = parameters('k, p, l, m')

a0 = 10
b = a0 - d + a # [B] is not independent.

model_dict = {
    D(d, t): l * c * b - m * d,
    D(c, t): k * a * b - p * c - l * c * b + m * d,
    D(a, t): - k * a * b + p * c,
}

model = ODEModel(model_dict, initial={t: 0.0, a: a0, c: 0.0, d: 0.0})

# Generate some data
tdata = np.linspace(0, 3, 1000)
# Eval the normal way.
AA, AAB, BAAB = model(t=tdata, k=0.1, l=0.2, m=.3, p=0.3)

plt.plot(tdata, AA, color='red', label='[AA]')
plt.plot(tdata, AAB, color='blue', label='[AAB]')
plt.plot(tdata, BAAB, color='green', label='[BAAB]')
plt.plot(tdata, b(d=BAAB, a=AA), color='pink', label='[B]')
# plt.plot(tdata, AA + AAB + BAAB, color='black', label='total')
plt.legend()
plt.show()