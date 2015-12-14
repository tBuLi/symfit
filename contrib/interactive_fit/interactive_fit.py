# -*- coding: utf-8 -*-

import numpy as np
from symfit.api import Fit  # Should be ...api import fit. Or something. Relative imports.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import itertools
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import functools

#from pkg_resources import parse_version

#SUPPORTED_VERSION = '0.3.0'
#SYMFIT_VERSION = symfit.version  # something something.
#
#if parse_version(SUPPORTED_VERSION) != parse_version(SYMFIT_VERSION):
#    raise EnvironmentError("Your symfit version is not supported. YMMV.")


def marginalize(model, guess, remaining_variables, dxs):
#    print(model.independent_vars)
#    print(guess)
#    print([index for index, var in enumerate(model.independent_vars) if var not in remaining_variables])
    reduced_indices = tuple(index for index, var in enumerate(model.independent_vars)
                            if var not in remaining_variables)
    total = np.sum(guess.T, axis=reduced_indices)
#    norm = np.prod([guess.shape[idx] for idx in reduced_indices])
#    norm = 1
#    print(max(total))
    #raise Exception(norm)
    total *= np.prod([dxs[idx] for idx in reduced_indices])
    return total.T



class InteractiveFit2D(Fit):
    """A class that provides a visual_guess method which provides
    an graphical, interactive way of guessing initial fitting parameters."""
    def _update_plot(self, _):
        """Callbak to redraw the plot to reflect the new parameter values."""
        # Since all sliders call this same callback without saying who they are
        # I need to update the values for all parameters. This can be
        # circumvented by creating a seperate callback function for each
        # parameter.
        for p in self.model.params:
            p.value = self._sliders[p].val
        # Also see the comment below about using keyword arguments (line 57)
        vals = self.model(self.xpoints, **{p.name: p.value for p in self.model.params})
        vals = getattr(vals, self.model.dependent_vars[0].name)
        self._my_plot.set_ydata(vals)

    def visual_guess(self, n_points=50):
        """Create a matplotlib window with sliders for all parameters
        in this model, so that you may graphically guess initial fitting
        parameters. n_points is the number of points drawn for the plot.
        Data points are plotted as blue points, the proposed model as
        a red line.
        Slider extremes are taken from the parameters where possible. If
        these are not provided, the minimum is 0; and the maximum is value*2.
        If no initial value is provided, it defaults to 1.
        
        Parameters
        ----------
        n_points : int
            The number of points used for drawing the fitted function.
        
        Returns
        -------
        None
        """
        
        projections = list(itertools.product(self.model.independent_vars, self.model.dependent_vars))
        x_mins = {v: np.min(data) for v, data in self.independent_data.items()}
        x_maxs = {v: np.max(data) for v, data in self.independent_data.items()}
        self.x_points = {v: np.linspace(x_mins[v], x_maxs[v], n_points) for v in self.independent_data}
        # It would be nice if I could get these from model someway...
        # Take from dependent data.
        y_mins = {v: np.min(data) for v, data in self.dependent_data.items()}
        y_maxs = {v: np.max(data) for v, data in self.dependent_data.items()}

        self.fig = plt.figure()

        # Make room for the sliders:
        bot = 0.1 + 0.05*len(self.model.params)
        self.fig.subplots_adjust(bottom=bot)

        # If these are not ints, matplotlib will crash and burn with an utterly
        # vague error.
        ncols = int(np.ceil(len(projections)**0.5))
        nrows = int(np.ceil(len(projections)/ncols))

        self.plots = {}
        for plotnr, proj in enumerate(projections, 1):
            x, y = proj
            ax = self.fig.add_subplot(ncols, nrows, plotnr, label='{}{}'.format(x.name, y.name))
            self.plots[proj] = ax
            ax.set_ylim(y_mins[y.name], y_maxs[y.name])
            ax.set_xlim(x_mins[x.name], x_maxs[x.name])
        
        i = 0.05
        self._sliders = {}
        for p in self.model.params:
            if not p.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            # start-x, start-y, width, height
            ax = self.fig.add_axes((0.162, i, 0.68, 0.03), axis_bgcolor=axbg, label=p.name)
            val = p.value
            if p.min is None:
                minimum = 0
            else:
                minimum = p.min
            if p.max is None:
                maximum = 2 * val
            else:
                maximum = p.max

            slid = plt.Slider(ax, p.name, minimum, maximum, valinit=val)
            self._sliders[p] = slid
            slid.on_changed(self._update_plot)
            i += 0.05
        plt.show()
        return 
        self.xpoints = np.linspace(x_min, x_max, n_points)

        plt.ylim(y_min, y_max)  # Y-Domain for self.model
        plt.xlim(x_min, x_max)  # X-domain for self.model

        # It might be better to use keyword arguments (p.name: p.value), but
        # I don't know (any more) if that's supported by self.model.
        vals = self.model(self.xpoints, **{p.name: p.value for p in self.model.params})
        vals = getattr(vals, self.model.dependent_vars[0].name)

        self._my_plot, = ax.plot(self.xpoints, vals, color='red')
        ax.scatter(self.independent_data[self.model.independent_vars[0].name],
                   self.dependent_data[self.model.dependent_vars[0].name],
                   c='blue')
        plt.axis([x_min, x_max, y_min, y_max])

        i = 0.05
        self._sliders = {}
        for p in self.model.params:
            if not p.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            # start-x, start-y, width, height
            ax = plt.axes([0.162, i, 0.68, 0.03], axisbg=axbg)

            val = p.value
            if p.min is None:
                minimum = 0
            else:
                minimum = p.min
            if p.max is None:
                maximum = 2 * val
            else:
                maximum = p.max

            slid = plt.Slider(ax, p.name, minimum, maximum, valinit=val)
            self._sliders[p] = slid
            slid.on_changed(self._update_plot)
            i += 0.05

        plt.show()


class ProjectionPlot:
    """Helper class for InteractiveFit3D. It holds information for a plot of
    one projection in 3 dimensions.
    Data is plotted as a scatterplot in blue, the model as a red surface."""
    def __init__(self, axes, xydata, z_interpolator, xymesh, z_function, title):
        """axes: The axes upon which to plot
        xydata: The original data
        z_interpolator: The function to call to project xydata on 3 dimensions
        given values for the other free variables
        xymesh: an ordered grid of points for which to calculate z_function
        at given parameters.
        z_function: called with xymesh, values for the other free variables
        and parameters to create a surface plot.
        title: The title for this plot"""
        self.ax = axes

        self.xydata = xydata
        self.z_interpolator = z_interpolator

        self.xymesh = xymesh
        self.z_function = z_function

        self.title = title

    def scatter(self, variables, **kwargs):
        """Given other free variables, draw a scatterplot with the
        original data. The original data is projected on the relevant axes."""
        z_points = self.z_interpolator(**variables)
        self.ax.scatter(*self.xydata, zs=z_points, **kwargs)

    def plot(self, variables_parameters, **kwargs):
        """Given the other free variables and parameters, plot the model as a 
        surface"""
        z_values = self.z_function(**variables_parameters)
        self.ax.plot_surface(self.xymesh[0], self.xymesh[1], z_values, **kwargs)

    def clear(self):
        """Clears the current plot"""
        self.ax.cla()
        self.ax.set_title(self.title)
        self.ax.grid(False)

    def update_plot(self, variables, parameters):
        """Give values for the variables and parameters, update the plot"""
        self.clear()
        self.scatter(variables, color='blue', antialiased=False)
        d = variables.copy()
        d.update(parameters)
        self.plot(d, color='red', rstride=1, cstride=1, shade=True, antialiased=False)


class InteractiveFit3D(Fit):
    """A class which plots N-dimensional models by projecting them
    on all combinations of 3 dimensions in order to allow users to
    make a graphical guess of initial fitting values."""
    # TODO: multiprocess redrawing

    def _update_plot(self, variable):
        """Update the subplots that changed, given a new value for variable."""
        if variable in self.model.independent_vars:
            var_idx = self.model.independent_vars.index(variable)
        else:
            var_idx = None  # Updates all plots

        values_params = {p.name: p.value for p in self.model.params}

        for xs in self.plots:
            if var_idx not in xs:
                values_vars = {v.name: self.variable_values[v] for v in self.model.independent_vars}
                for x in xs:
                    del values_vars[self.model.independent_vars[x].name]  # Some variables are pre-applied to the function in plot. So remove them
                self.plots[xs].update_plot(values_vars, values_params)
        self.fig.canvas.draw()

    def _slider_changed(self, val, variable):
        """Update the stored parameter and variable values,
        and update plots."""
        if variable in self.model.params:
            param = variable
            param.value = val
        elif variable in self.model.independent_vars:
            self.variable_values[variable] = val
        self._update_plot(variable)

    def _draw_sliders(self):
        """Constructs and draws sliders for all parameters and variables."""
        i = 0.05
        x_mins = np.min(self.xdata, axis=1)
        x_maxs = np.max(self.xdata, axis=1)
        for p in self.model.params:
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
            self._sliders[p] = slider
            i += 0.05
        if len(self.model.independent_vars) > 2:
            for v, xmin, xmax in zip(self.model.independent_vars, x_mins, x_maxs):
                ax = plt.axes([0.162, i, 0.68, 0.03], axisbg='green')  # start-x, start-y, width, height
                slider = self._construct_slider(ax, v, xmin, xmax, (xmax+xmin)/2)
                self._sliders[v] = slider
                i += 0.05

    def _construct_slider(self, axes, var, min_val, max_val, init_val):
        """Constructs a slider and registers a variable specific call-back
        modelled on _slider_changed, such that it gets called with the
        variable that changed."""
        slid = plt.Slider(axes, var.name, min_val, max_val, valinit=init_val)
        f = functools.partial(self._slider_changed, variable=var)
        slid.on_changed(f)
        return slid

    def visual_guess(self, n_points=50, interpolation='linear'):
        """Create a matplotlib window with sliders for all parameters
        in this model, so that you may graphically guess initial fitting
        parameters. n_points is the number of points drawn for the plot
        per dimension. A tuple of integers with the same length as the number
        of dimensions is also allowed.
        interpolation is the interpolation order used interpolate/project
        the original data on the relevant dimensions. Allowed values are
        'linear' (default) and 'nearest'.
        Data points are plotted as blue points, the proposed model as
        a red surface.
        Slider extremes are taken from the parameters where possible. If
        these are not provided, the minimum is 0; and the maximum is value*2.
        If no initial value is provided, it defaults to 1."""
        print(self.dependent_data)
        print(self.independent_data)
        print(self.model.dependent_vars)
        print(self.model.independent_vars)
        x_mins = {}
        x_maxs = {}
        for name, data in self.independent_data.items():
            x_mins[name] = np.min(data)
            x_maxs[name] = np.max(data)
        #x_mins = np.min(self.xdata, axis=1)
        #x_maxs = np.max(self.xdata, axis=1)
        self.nvars = len(self.model.independent_vars)
        
        self._sliders = {}
        self.variable_values = dict(zip(self.model.independent_vars, [0]*self.nvars))

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
               len(x_mins) != len(self.model.independent_vars) or len(self.xdata) != len(x_mins):
                raise IndexError("Size mismatch in variables somewhere")
        except TypeError:  # Can't do len somewhere.
            raise IndexError("Size mismatch in variables somewhere")

        if self.nvars == 1:
            raise IndexError("Does not support 2D problems! Try using InteractiveFit2D.")

        self.fig = plt.figure()
        self.fig.set_frameon(False)

        # Make room for the sliders:
        bot = 0.1 + 0.05*len(self.model.params)
        if self.nvars > 2:
            bot += 0.05*self.nvars
        self.fig.subplots_adjust(bottom=bot)

        xpoints = {}
        for x_name, n_point in zip(x_mins.keys(), n_points):
            xpoints[x_name].append(np.linspace(x_mins[x_name], x_maxs[x_name], n_point))
        #self.projections = list(combinations(range(self.nvars), 2))
        self.projections = list(combinations(self.model.dependent_vars, 2))

        f = interpolator(self.xdata.T, self.ydata)
        def master_interpolator(**kwargs):
            """Converts keyword arguments to positional arguments,
            and applies it to the interpolator"""
            args = [0]*len(kwargs)
            for i, v in enumerate(self.model.independent_vars):
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
            
            # Pre-apply all the data that will not change for this projection
            z_function = functools.partial(self.model,
                                           **{self.model.independent_vars[x1].name: mesh[0],
                                              self.model.independent_vars[x2].name: mesh[1]})
            y_interpolator = functools.partial(master_interpolator,
                                               **{self.model.independent_vars[x].name: self.xdata[x] for x in (x1, x2)})

            title = "{}{}".format(*[self.model.independent_vars[x].name for x in (x1, x2)])

            p = ProjectionPlot(ax, xydata, y_interpolator, mesh, z_function, title)
            self.plots[(x1, x2)] = p
            plotnr += 1

        self._draw_sliders()
        self._update_plot(self.model.params[0])  # Update all plots.
        plt.show()


if __name__ == "__main__":
    from symfit import Parameter, Variable, exp, pi, variables, sqrt, parameters
    import scipy.stats
    
#    x = Variable()
#    y = Variable()
#    k = Parameter(900)
#    x0 = Parameter(1.5)
#
#    def distr(x, k, x0):
#        kbT = 2.479  # kJ/mol
#        return exp(-k*(x-x0)**2/kbT)

    mean = [1, 2]
    cov = [[0.1, 0], [0, 5.0]]
    
    np.random.seed(0)
    xydata = np.random.multivariate_normal(mean, cov, 100)
    kernel = scipy.stats.gaussian_kde(xydata.T, bw_method='silverman')
#    print(xydata)
#    print(kernel.covariance)
#    print(kernel.factor)
#    print(kernel.covariance/kernel.factor)
#    print(np.linalg.det(kernel.covariance/kernel.factor))
    mins = np.min(xydata, axis=0)
    maxs = np.max(xydata, axis=0)
    x_data = np.linspace(mins[0], maxs[0], 100)
    y_data = np.linspace(mins[1], maxs[1], 100)
    xx, yy = np.meshgrid(x_data, y_data)
#    xx = xx.flatten()
#    yy = yy.flatten()
    xy_grid = np.column_stack((xx.ravel(), yy.ravel()))
#    print(xy_grid)
    grid_prob = kernel.pdf(xy_grid.T)
    grid_prob = grid_prob.reshape(xx.shape)
#    fig = plt.figure()
    
    dx = np.diff(x_data)[0]
    dy = np.diff(y_data)[0]
#    print(dx)

#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(xx, yy, grid_prob,  s=10)
##    ax.scatter(xydata[:, 0], xydata[:, 1], color='r')
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#    xx = xx.flatten()
#    yy = yy.flatten()
    
    x, y, p = variables('x, y, p')
    mu_x, mu_y, sig_x, sig_y, rho, amp = parameters('mu_x, mu_y, sig_x, sig_y, rho, amp')
    rho.value = 0.1
    z = (x - mu_x)**2/sig_x**2 + (y - mu_y)**2/sig_y**2 - 2 * rho * (x - mu_x) * (y - mu_y)/(sig_x * sig_y)
    model = {p: amp * exp(- z / (2 * (1 - rho**2))) / (2 * pi * sig_x * sig_y * sqrt(1 - rho**2))}
    z_hat = model[p](x=xydata[:, 0], y=xydata[:, 1], mu_x=mean[0], mu_y=mean[1], rho=0, sig_x=cov[0][0], sig_y=cov[1][1], amp=1)
    guess = model[p](x=xx, y=yy, mu_x=mean[0], mu_y=mean[1], rho=0, sig_x=cov[0][0], sig_y=cov[1][1], amp=1)
    p_hat = kernel.pdf(xydata.T) * z_hat
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(xx, yy, z_data)
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#    print(p_hat)
#    print(grid_prob)
    fit = InteractiveFit2D(model, x=xx, y=yy, p=guess)
##    print(guess.shape)
##    print(np.max(guess))
##    print(guess.dtype)
#    marginal_guess = marginalize(fit.model, guess, (x,), [dx, dy])
#    marginal_prob = marginalize(fit.model, p_hat, tuple(), [dx, dy])
#    print(marginal_prob)
#    marginal_prob = marginalize(fit.model, p_hat, [x], [dx, dy])
#    plt.plot(x_data, marginal_guess)
#    plt.plot(x_data, marginal_prob)
##    plt.ylim(0, 2)
#    plt.show()
    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.scatter(xx, yy, marginal)
#    ax.set_xlabel('X Label')
#    ax.set_ylabel('Y Label')
#    ax.set_zlabel('Z Label')
#    
#    fit = InteractiveFit2D(model, x=xx, y=yy, p=z_data)
    fit.visual_guess(200)
    print("Guessed values: ")
    for p in fit.model.params:
        print("{}: {}".format(p.name, p.value))
    fit_result = fit.execute(maxfev=1000)
    print(fit_result)
#    model = {y: distr(x, k, x0)}
#    x_data = np.linspace(0, 2.5, 50)
#    y_data = model[y](x=x_data, k=1000, x0=1)
#    fit = InteractiveFit2D(model, x=x_data, y=y_data)
#    fit.visual_guess(250)
#    print("Guessed values: ")
#    for p in fit.model.params:
#        print("{}: {}".format(p.name, p.value))
#    fit_result = fit.execute(maxfev=1000)
#    print(fit_result)
