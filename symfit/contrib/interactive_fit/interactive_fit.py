# -*- coding: utf-8 -*-

import numpy as np
from symfit import Fit  # Should be ...api import fit. Or something. Relative imports.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import itertools


#from pkg_resources import parse_version

#SUPPORTED_VERSION = '0.3.0'
#SYMFIT_VERSION = symfit.version  # something something.
#
#if parse_version(SUPPORTED_VERSION) != parse_version(SYMFIT_VERSION):
#    raise EnvironmentError("Your symfit version is not supported. YMMV.")


class InteractiveFit2D(Fit):
    """A class that provides a visual_guess method which provides
    an graphical, interactive way of guessing initial fitting parameters."""

    def __init__(self, *args, **kwargs):
        if 'n_points' in kwargs:
            n_points = kwargs.pop('n_points')
        else:
            n_points = 100
        super().__init__(*args, **kwargs)

        if len(self.independent_data) > 1:
            raise IndexError("Only 2D problems are supported.")

        self._projections = list(itertools.product(self.model.independent_vars,
                                                   self.model.dependent_vars))

        x_mins = {v: np.min(data) for v, data in self.independent_data.items()}
        x_maxs = {v: np.max(data) for v, data in self.independent_data.items()}

        self._x_points = {v: np.linspace(x_mins[v], x_maxs[v], n_points)
                          for v in self.independent_data}

        y_mins = {v: np.min(data) for v, data in self.dependent_data.items()}
        y_maxs = {v: np.max(data) for v, data in self.dependent_data.items()}

        self.fig = plt.figure()

        # Make room for the sliders:
        bot = 0.1 + 0.05*len(self.model.params)
        self.fig.subplots_adjust(bottom=bot)

        # If these are not ints, matplotlib will crash and burn with an utterly
        # vague error.
        nrows = int(np.ceil(len(self._projections)**0.5))
        ncols = int(np.ceil(len(self._projections)/nrows))

        self._plots = {}
        for plotnr, proj in enumerate(self._projections, 1):
            x, y = proj
            ax = self.fig.add_subplot(ncols, nrows, plotnr,
                                      label='{} {}'.format(x.name, y.name))
            ax.set_title(ax.get_label())
            ax.set_ylim(y_mins[y.name], y_maxs[y.name])
            ax.set_xlim(x_mins[x.name], x_maxs[x.name])
            # TODO reduce dimensionality.
            ax.scatter(self.independent_data[x.name],
                       self.dependent_data[y.name])

            vals = self._get_data(x, y)
            plot, = ax.plot(*vals, c='red')
            self._plots[proj] = plot

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

    def _update_plot(self, _):
        """Callbak to redraw the plot to reflect the new parameter values."""
        # Since all sliders call this same callback without saying who they are
        # I need to update the values for all parameters. This can be
        # circumvented by creating a seperate callback function for each
        # parameter.
        for p in self.model.params:
            p.value = self._sliders[p].val
        for indep_var, dep_var in self._projections:
            plot = self._plots[(indep_var, dep_var)]
            # TODO: reduce dimensionality of self._x_points and vals for this projection
            vals = self._get_data(indep_var, dep_var)
            plot.set_data(*vals)

    def _get_data(self, independent_var, dependent_var):
        """
        Convenience method for evaluating the model, giving the projection
        dependent_var, independent_var

        Parameters
        ----------
        dependent_var : Variable
            The dependent variable to calculate the data for
        independent_var : Variable
            The independent variable to calculate the data for

        Returns
        -------
        x_points, y_points
        """
        x_points = self._x_points[independent_var.name]
        arguments = {independent_var.name: x_points}
        arguments.update({p.name: p.value for p in self.model.params})
        return x_points, self.model[dependent_var](**arguments)

    def visual_guess(self, n_points=100):
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
        plt.show()
        return


if __name__ == "__main__":
    from symfit import Parameter, Variable, exp, parameters

    def distr(x, k, x0):
        kbT = 4.11
        return exp(-k*(x-x0)**2/kbT)
#
#    x = Variable()
#    y = Variable()
#    k = Parameter(900)
#    x0 = Parameter(1.5)
#
#    model = {y: distr(x, k, x0)}
#    x_data = np.linspace(0, 2.5, 50)
#    y_data = model[y](x=x_data, k=1000, x0=1)
#    fit = InteractiveFit2D(model, x=x_data, y=y_data, n_points=250)
#    fit.visual_guess()
#    print("Guessed values: ")
#    for p in fit.model.params:
#        print("{}: {}".format(p.name, p.value))
#    fit_result = fit.execute(maxfev=1000)
#    print(fit_result)

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
