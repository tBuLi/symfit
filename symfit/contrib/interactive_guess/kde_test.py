# -*- coding: utf-8 -*-
from symfit import Variable, Parameter, Model
from scipy.stats import gaussian_kde
#from symfit.contrib.interactive_guess import InteractiveGuess2D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


x = Variable()
y = Variable()
z = Variable()
k = Parameter(900)
x0 = Parameter(1.5)

model = {
    z: x - x0 + x * y * k
}
model = Model(model)

# Generate example data
x_data = np.linspace(0, 2.5, 50)
y_data = np.linspace(3, 5, 50)
xx, yy = np.meshgrid(x_data, y_data)
x_data = xx.flatten()
y_data = yy.flatten()

data = model(x=x_data, y=y_data, k=1000, x0=1)
z_data = data.z

print(x_data)
print(y_data)
print(z_data)
x_bins = np.linspace(np.min(x_data), np.max(x_data), 50)
x_bin_assignment = np.digitize(x_data, x_bins) - 1




z_model = model(x_data, y_data, k=900, x0=1).z

# XY projection
data = np.column_stack((x_data, y_data)).T
print(data)
print(data.shape)
kde = gaussian_kde(data)

xx, yy = np.meshgrid(np.linspace(np.min(x_data), np.max(x_data), 50), np.linspace(np.min(y_data), np.max(y_data), 50))
x_grid = xx.flatten()
y_grid = yy.flatten()



z_grid = kde.evaluate(np.column_stack((x_grid, y_grid)).T)
print(z_grid)
print(z_grid.shape)
#plt.contourf(xx, yy, z_grid.reshape(50, 50), cmap='Reds')
#plt.scatter(x_data, y_data)
#
#
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

# XZ projection
data = np.column_stack((x_data, z_data)).T
print(data)
print(data.shape)
kde = gaussian_kde(data)

xx, zz = np.meshgrid(np.linspace(np.min(x_data), np.max(x_data), 50), np.linspace(np.min(z_data), np.max(z_data), 50))
x_grid = xx.flatten()
z_grid = zz.flatten()

y_grid = kde.evaluate(np.column_stack((x_grid, z_grid)).T)
print(y_grid)
print(y_grid.shape)
plt.contourf(xx, zz, y_grid.reshape(50, 50), cmap='Blues')


x_plot_data = []
x_plot_error = []
z_plot_data = []
z_plot_error = []
for idx in range(50):
    idx_mask = x_bin_assignment == idx
    if not np.any(idx_mask):
        continue
    xs = x_data[idx_mask]
    x_plot_data.append(np.mean(xs))
    x_error = np.percentile(xs, [5, 95])
    x_plot_error.append(x_error)
    zs = z_data[idx_mask]
    z_plot_data.append(np.mean(zs))
    z_error = np.percentile(zs, [5, 95])
    z_plot_error.append(z_error)
x_plot_data = np.array(x_plot_data)
x_plot_error = np.abs(np.array(x_plot_error) - x_plot_data[:, np.newaxis] ).T
z_plot_data = np.array(z_plot_data)
z_plot_error = np.abs(np.array(z_plot_error) - z_plot_data[:, np.newaxis] ).T

plt.errorbar(x_plot_data, z_plot_data, xerr=x_plot_error, yerr=z_plot_error, c='r')
plt.xlabel('x')
plt.ylabel('z')
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(x_data, y_data, z_data)
#plt.show()

#
#guess = InteractiveGuess2D(model, x=x_data, y1=y1_data, y2=y2_data, n_points=250)
#guess.execute()
#print(guess)
#
#fit = Fit(model, x=x_data, y1=y1_data, y2=y2_data)
#fit_result = fit.execute()
#print(fit_result)
