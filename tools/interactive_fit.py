# -*- coding: utf-8 -*-

import numpy as np
from symfit.api import Fit, lambdify
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from itertools import combinations
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import functools


class InteractiveFit2D(Fit):
    def _update_plot(self, val):
        for p in self.params:
            self.params[self.params.index(p)].value = self.sliders[p].val
        self.my_plot.set_ydata(self.vec_func(self.xpoints, *list(p.value for p in self.params)))

    def visual_guess(self, n_points=50):
        x_min = np.min(self.xdata)
        x_max = np.max(self.xdata)
        y_min = 0
        y_max = 1

        if len(self.vars) != 1:
            raise IndexError("Only supports 2D problems!")

        ax = plt.subplot(111)
        plt.subplots_adjust(bottom=0.1 + 0.05*len(self.params))
        plt.title("Fit")

        self.xpoints = np.linspace(x_min, x_max, n_points)

        plt.ylim(y_min, y_max)  # Y-Domain for self.model
        plt.xlim(x_min, x_max)  # X-domain for self.model

        self.vec_func = lambdify(self.vars+self.params, self.model, "numpy")
        vals = self.vec_func(self.xpoints, *list(p.value for p in self.params))
        self.my_plot, = ax.plot(self.xpoints, vals, color='red')
        ax.scatter(self.xdata, self.ydata, c='blue')
        plt.axis([x_min, x_max, y_min, y_max])

        i = 0.05
        self.sliders = {}
        for p in self.params:
            if not p.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            ax = plt.axes([0.162, i, 0.68, 0.03], axisbg=axbg)  # start-x, start-y, width, height

            if not hasattr(p, "value") or p.value is None:
                val = 1
            else:
                val = p.value
            if not hasattr(p, "min") or p.min is None:
                minimum = 0
            else:
                minimum = p.min
            if not hasattr(p, "max") or p.max is None:
                maximum = 2 * val
            else:
                maximum = p.max

            slid = plt.Slider(ax, p.name, minimum, maximum, valinit=val)
            self.sliders[p] = slid
            slid.on_changed(self._update_plot)
            i += 0.05

        plt.show()


class ProjectionPlot:
    def __init__(self, axes, xydata, z_interpolator, xymesh, z_function, title):
        self.ax = axes

        self.xydata = xydata
        self.z_interpolator = z_interpolator

        self.xymesh = xymesh
        self.z_function = z_function

        self.title = title

    def scatter(self, variables, **kwargs):
        z_points = self.z_interpolator(**variables)
        self.ax.scatter(*self.xydata, zs=z_points, **kwargs)

    def plot(self, variables_parameters, **kwargs):
        z_values = self.z_function(**variables_parameters)
        self.ax.plot_surface(self.xymesh[0], self.xymesh[1], z_values, **kwargs)

    def clear(self):
        self.ax.cla()
        self.ax.set_title(self.title)
        self.ax.grid(False)

    def update_plot(self, variables, parameters):
        self.clear()
        self.scatter(variables, color='blue', antialiased=False)
        d = variables.copy()
        d.update(parameters)
        self.plot(d, color='red', rstride=1, cstride=1, shade=True, antialiased=False)


class InteractiveFit3D(Fit):
    # TODO: multiprocess redrawing

    def _update_plot(self, variable):
        if variable in self.vars:
            var_idx = self.vars.index(variable)
        else:
            var_idx = None  # Updates all plots

        values_params = {p.name: p.value for p in self.params}

        for xs in self.plots:
            if var_idx not in xs:
                values_vars = {v.name: self.variable_values[v] for v in self.vars}
                for x in xs:
                    del values_vars[self.vars[x].name]  # Some variables are pre-applied to the function in plot. So remove them
                self.plots[xs].update_plot(values_vars, values_params)
        self.fig.canvas.draw()

    def _slider_changed(self, val, variable):
        if variable in self.params:
            param = variable
            param.value = val
        elif variable in self.vars:
            self.variable_values[variable] = val
        self._update_plot(variable)

    def _draw_sliders(self):
        i = 0.05
        x_mins = np.min(self.xdata, axis=1)
        x_maxs = np.max(self.xdata, axis=1)
        for p in self.params:
            if not p.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            if not hasattr(p, "value") or p.value is None:
                val = 1
            else:
                val = p.value
            if not hasattr(p, "min") or p.min is None:
                minimum = 0
            else:
                minimum = p.min
            if not hasattr(p, "max") or p.max is None:
                maximum = 2 * val
            else:
                maximum = p.max
            ax = plt.axes([0.162, i, 0.68, 0.03], axisbg=axbg)  # start-x, start-y, width, height
            slider = self._construct_slider(ax, p, minimum, maximum, val)
            self.sliders[p] = slider
            i += 0.05
        if len(self.vars) > 2:
            for v, xmin, xmax in zip(self.vars, x_mins, x_maxs):
                ax = plt.axes([0.162, i, 0.68, 0.03], axisbg='green')  # start-x, start-y, width, height
                slider = self._construct_slider(ax, v, xmin, xmax, (xmax+xmin)/2)
                self.sliders[v] = slider
                i += 0.05

    def _construct_slider(self, axes, var, min_val, max_val, init_val):
        slid = plt.Slider(axes, var.name, min_val, max_val, valinit=init_val)
        f = functools.partial(self._slider_changed, variable=var)
        slid.on_changed(f)
        return slid

    def visual_guess(self, n_points=50, interpolation='linear'):
        x_mins = np.min(self.xdata, axis=1)
        x_maxs = np.max(self.xdata, axis=1)
        self.nvars = len(self.vars)
        self.sliders = {}
        self.variable_values = dict(zip(self.vars, [0]*self.nvars))

        if interpolation.lower() == 'linear':
            interpolator = LinearNDInterpolator
        elif interpolation.lower() == 'nearest':
            interpolator = NearestNDInterpolator
        else:
            raise KeyError('Unknown interpolation')

        if isinstance(n_points, int):
            n_points = [n_points] * len(x_mins)

        try:
            if len(x_mins) != len(x_maxs) or len(n_points) != len(x_mins) or\
               len(x_mins) != len(self.vars) or len(self.xdata) != len(x_mins):
                raise IndexError("Size mismatch in variables somewhere")
        except TypeError:
            raise IndexError("Size mismatch in variables somewhere")

        if self.nvars == 1:
            raise IndexError("Does not support 2D problems! Try using InteractiveFit2D.")

        self.fig = plt.figure()
        self.fig.set_frameon(False)

        # Make room for the sliders:
        bot = 0.1 + 0.05*len(self.params)
        if self.nvars > 2:
            bot += 0.05*self.nvars
        self.fig.subplots_adjust(bottom=bot)

        xpoints = []
        for x_min, x_max, n_point in zip(x_mins, x_maxs, n_points):
            xpoints.append(np.linspace(x_min, x_max, n_point))
        self.projections = list(combinations(range(self.nvars), 2))

        f = interpolator(self.xdata.T, self.ydata)
        def master_interpolator(**kwargs):  # Converts keyword arguments to positional arguments, and applies it to the interpolator
            args = [0]*len(kwargs)
            for i, v in enumerate(self.vars):
                args[i] = kwargs[v.name]
            return f(*args)

        ncols = np.ceil(len(self.projections)**0.5)
        nrows = ncols

        self.plots = {}
        plotnr = 1
        for x1, x2 in self.projections:
            ax = self.fig.add_subplot(ncols, nrows, plotnr, projection="3d")
            ax.set_axis_off()  # Remove back panes.

            xydata = np.array([self.xdata[x1], self.xdata[x2]])
            mesh = np.meshgrid(xpoints[x1], xpoints[x2])

            z_function = functools.partial(self.model,
                                           **{self.vars[x1].name: mesh[0],
                                              self.vars[x2].name: mesh[1]})
            y_interpolator = functools.partial(master_interpolator,
                                               **{self.vars[x].name: self.xdata[x] for x in (x1, x2)})

            title = "{}{}".format(*[self.vars[x].name for x in (x1, x2)])

            p = ProjectionPlot(ax, xydata, y_interpolator, mesh, z_function, title)
            self.plots[(x1, x2)] = p
            plotnr += 1

        self._draw_sliders()
        self._update_plot(self.params[0])  # Update all plots.
        plt.show()


if __name__ == "__main__":
    def f(xy, a, b):
        x, y, a = xy
        return np.cos(a*x) * b*np.sin(y)

    from symfit.api import Parameter, Variable, sin, cos

    def f_sym(x, y, a, b):
        return cos(a*x) * b*sin(y)

    xdata = np.linspace(-np.pi, np.pi, 7)
    ydata = np.linspace(-np.pi, np.pi, 7)
    adata = np.linspace(0, 2.5, 7)

    xx, yy, aa = np.meshgrid(xdata, ydata, adata)

    xydata = np.array((xx.flatten(), yy.flatten(), aa.flatten()))
    #xydata = np.array((xx.flatten(), yy.flatten()))
    zs = np.array([f(xy, 1.5,  2) for xy in xydata.T])
    zdata = zs.reshape(xx.shape)
#    print(xydata.T.shape, zdata.flatten().shape)
#    interp = NearestNDInterpolator(xydata.T, zdata.flatten())
#    print(interp(0.5, 1, 0.2))
#    interp2 = LinearNDInterpolator(xydata.T, zdata.flatten())
#    print(interp2(0.5, 1, 0.2))

    x = Variable()
    y = Variable()
    a = Variable()
    b = Parameter()

    model = cos(a*x) * b*sin(y)
    model = f_sym(x, y, a, b)
    fit = InteractiveFit3D(model, xydata.T, zdata.flatten())

    fit.visual_guess(15, 'nearest')
    print("Guessed values: ")
    for p in fit.params:
        print("{}: {}".format(p.name, p.value))
    fit_result = fit.execute(maxfev=1000)

    print(fit_result)

    X, Y, A = np.meshgrid(np.linspace(-np.pi, np.pi, 30), np.linspace(-np.pi, np.pi, 30), np.linspace(0, 2.5, 30))

    Z = model(x=X, y=Y, a=A, **fit_result.params)
