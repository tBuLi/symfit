# SPDX-FileCopyrightText: 2014-2020 Martin Roelfs
#
# SPDX-License-Identifier: MIT

# -*- coding: utf-8 -*-

import matplotlib as mpl
mpl.use('Agg')

from ... import ODEModel, Derivative, latex
from ...core.fit import TakesData
from ...core.support import keywordonly, key2str, deprecated

import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.ioff()


class InteractiveGuess(TakesData):
    """A class that provides an graphical, interactive way of guessing initial
    fitting parameters."""

    @keywordonly(n_points=50, log_contour=True, percentile=(5, 95))
    def __init__(self, *args, **kwargs):
        """Create a matplotlib window with sliders for all parameters
        in this model, so that you may graphically guess initial fitting
        parameters. n_points is the number of points drawn for the plot.
        Data points are plotted as a blue contour plot, the proposed model as
        a red line. The errorbars on the proposed model represent the
        percentile of data within the thresholds.

        Slider extremes are taken from the parameters where possible. If
        these are not provided, the minimum is 0; and the maximum is value*2.
        If no initial value is provided, it defaults to 1.

        This will modify the values of the parameters present in model.

        :param n_points: The number of points used for drawing the
            fitted function. Defaults to 50.
        :type n_points: int

        :param log_contour: Whether to plot the contour plot of the log of the
            density, rather than the density itself. If True, any density less
            than 1e-7 will be considered 0. Defaults to True.
        :type log_contour: bool

        :param percentile: Controls the errorbars on the proposed model, such
            that the lower errorbar will cover percentile[0]% of the data, and
            the upper will cover percentile[1]%. Defaults to [5, 95], with
            corresponds to a 90% percentile. Should be a list of 2 numbers.
        :type percentile: list
        """
        self.log_contour = kwargs.pop('log_contour')
        n_points = kwargs.pop('n_points')
        self.percentile = kwargs.pop('percentile')
        super(InteractiveGuess, self).__init__(*args, **kwargs)
        if len(self.independent_data) > 1:
            self._dimension_strategy = StrategynD(self)
        else:
            self._dimension_strategy = Strategy2D(self)
        # TODO: Some of the code here is specific to the n-D case and should
        # be moved.
        self._projections = list(itertools.product(self.model.independent_vars,
                                                   self.model.dependent_vars))
        x_mins = {v: np.min(data) for v, data in self.independent_data.items()}
        x_maxs = {v: np.max(data) for v, data in self.independent_data.items()}

        # Stretch the plot 10-20% in the X direction, since that is visually
        # more appealing. We can't evaluate the model for x < x_initial, so
        # don't.
        for x in self.model.independent_vars:
            plotrange_x = x_maxs[x] - x_mins[x]
            if not hasattr(self.model, 'initial'):
                x_mins[x] -= 0.1 * plotrange_x
            x_maxs[x] += 0.1 * plotrange_x
        # Generate the points at which to evaluate the model with the proposed
        # parameters for plotting
        self._x_points = {v: np.linspace(x_mins[v], x_maxs[v], n_points)
                          for v in self.independent_data}
        meshgrid = np.meshgrid(*(self._x_points[v]
                                 for v in self.independent_data))
        self._x_grid = {v: meshgrid[idx].flatten()
                        for idx, v in enumerate(self.independent_data)}

        # Stretch the plot 20% in the Y direction, since that is visually more
        # appealing
        y_mins = {v: np.min(data) for v, data in self.dependent_data.items()}
        y_maxs = {v: np.max(data) for v, data in self.dependent_data.items()}
        for y in self.dependent_data:
            plotrange_y = y_maxs[y] - y_mins[y]
            y_mins[y] -= 0.1 * plotrange_y
            y_maxs[y] += 0.1 * plotrange_y

        self._y_points = {v: np.linspace(y_mins[v], y_maxs[v], n_points)
                          for v in self.dependent_data}

        self._set_up_figure(x_mins, x_maxs, y_mins, y_maxs)
        self._set_up_sliders()
        self._update_plot(None)

    @keywordonly(show=True, block=True)
    def execute(self, **kwargs):
        """
        Execute the interactive guessing procedure.

        :param show: Whether or not to show the figure. Useful for testing.
        :type show: bool
        :param block: Blocking call to matplotlib
        :type show: bool

        Any additional keyword arguments are passed to
        matplotlib.pyplot.show().
        """
        show = kwargs.pop('show')
        if show:
            # self.fig.show()  # Apparently this does something else,
            # see https://github.com/matplotlib/matplotlib/issues/6138
            plt.show(**kwargs)

    def _set_up_figure(self, x_mins, x_maxs, y_mins, y_maxs):
        """
        Prepare the matplotlib figure: make all the subplots; adjust their
        x and y range; plot the data; and plot an putative function.
        """
        self.fig = plt.figure()

        # Make room for the sliders:
        bot = 0.1 + 0.05*len(self.model.params)
        self.fig.subplots_adjust(bottom=bot)

        # If these are not ints, matplotlib will crash and burn with an utterly
        # vague error.
        nrows = int(np.ceil(len(self._projections)**0.5))
        ncols = int(np.ceil(len(self._projections)/nrows))

        # Make all the subplots: set the x and y limits, scatter the data, and
        # plot the putative function.
        self._plots = {}

        for plotnr, proj in enumerate(self._projections, 1):
            x, y = proj
            if Derivative(y, x) in self.model:
                title_format = '$\\frac{{\\partial {dependant}}}{{\\partial {independant}}} = {expression}$'
            else:
                title_format = '${dependant}({independant}) = {expression}$'
            plotlabel = title_format.format(
                dependant=latex(y, mode='plain'),
                independant=latex(x, mode='plain'),
                expression=latex(self.model[y], mode='plain'))
            ax = self.fig.add_subplot(ncols, nrows, plotnr,
                                      label=plotlabel)
            ax.set_title(ax.get_label())
            ax.set_ylim(y_mins[y], y_maxs[y])
            ax.set_xlim(x_mins[x], x_maxs[x])
            ax.set_xlabel('${}$'.format(x))
            ax.set_ylabel('${}$'.format(y))
            self._plot_data(proj, ax)
            plot = self._plot_model(proj, ax)
            self._plots[proj] = plot

    def _set_up_sliders(self):
        """
        Creates an slider for every parameter.
        """
        i = 0.05
        self._sliders = {}
        for param in self.model.params:
            if not param.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            # start-x, start-y, width, height
            ax = self.fig.add_axes((0.162, i, 0.68, 0.03),
                                   facecolor=axbg, label=param)
            val = param.value
            if not hasattr(param, 'min') or param.min is None:
                minimum = 0
            else:
                minimum = param.min
            if not hasattr(param, 'max') or param.max is None:
                maximum = 2 * val
            else:
                maximum = param.max

            slid = plt.Slider(ax, param, minimum, maximum,
                              valinit=val, valfmt='% 5.4g')
            self._sliders[param] = slid
            slid.on_changed(self._update_plot)
            i += 0.05

    def _plot_data(self, proj, ax):
        """Defers plotting the data to self._dimension_strategy"""
        return self._dimension_strategy.plot_data(proj, ax)

    def _plot_model(self, proj, ax):
        """Defers plotting the proposed model to self._dimension_strategy"""
        return self._dimension_strategy.plot_model(proj, ax)

    def _update_specific_plot(self, indep_var, dep_var):
        """Defers updating the proposed model to self._dimension_strategy"""
        return self._dimension_strategy.update_plot(indep_var, dep_var)

    def _update_plot(self, _):
        """Callback to redraw the plot to reflect the new parameter values."""
        # Since all sliders call this same callback without saying who they are
        # I need to update the values for all parameters. This can be
        # circumvented by creating a seperate callback function for each
        # parameter.
        for param in self.model.params:
            param.value = self._sliders[param].val
        for indep_var, dep_var in self._projections:
            self._update_specific_plot(indep_var, dep_var)

    def _eval_model(self):
        """
        Convenience method for evaluating the model with the current parameters

        :return: named tuple with results
        """
        arguments = self._x_grid.copy()
        arguments.update({param: param.value for param in self.model.params})
        return self.model(**key2str(arguments))

    def __str__(self):
        """
        Represent the guesses in a human readable way.

        :return: string with the guessed values.
        """
        msg = 'Guessed values:\n'
        for param in self.model.params:
            msg += '{}: {}\n'.format(param.name, param.value)
        return msg


class Strategy2D:
    """
    A strategy that describes how to plot a model that depends on a single independent variable,
    and how to update that plot.
    """
    def __init__(self, interactive_guess):
        self.ig = interactive_guess

    def plot_data(self, proj, ax):
        """
        Creates and plots a scatter plot of the original data.
        """
        x, y = proj
        ax.scatter(self.ig.independent_data[x],
                   self.ig.dependent_data[y], c='b')

    def plot_model(self, proj, ax):
        """
        Plots the model proposed for the projection proj on ax.
        """
        x, y = proj
        y_vals = getattr(self.ig._eval_model(), y.name)
        x_vals = self.ig._x_points[x]
        plot, = ax.plot(x_vals, y_vals, c='red')
        return plot

    def update_plot(self, indep_var, dep_var):
        """
        Updates the plot of the proposed model.
        """
        evaluated_model = self.ig._eval_model()
        plot = self.ig._plots[(indep_var, dep_var)]
        y_vals = getattr(evaluated_model, dep_var.name)
        x_vals = self.ig._x_points[indep_var]
        plot.set_data(x_vals, y_vals)


class StrategynD:
    """
    A strategy that describes how to plot a model that depends on a multiple independent variables,
    and how to update that plot.
    """
    def __init__(self, interactive_guess):
        self.ig = interactive_guess

    def plot_data(self, proj, ax):
        """
        Creates and plots the contourplot of the original data. This is done
        by evaluating the density of projected datapoints on a grid.
        """
        x, y = proj
        x_data = self.ig.independent_data[x]
        y_data = self.ig.dependent_data[y]
        projected_data = np.column_stack((x_data, y_data)).T
        kde = gaussian_kde(projected_data)

        xx, yy = np.meshgrid(self.ig._x_points[x], self.ig._y_points[y])
        x_grid = xx.flatten()
        y_grid = yy.flatten()

        contour_grid = kde.pdf(np.column_stack((x_grid, y_grid)).T)
        # This is an fugly kludge, but it seems nescessary to make low density
        # areas show up.
        if self.ig.log_contour:
            contour_grid = np.log(contour_grid)
            vmin = -7
        else:
            vmin = None
        ax.contourf(xx, yy, contour_grid.reshape(xx.shape),
                    50, vmin=vmin, cmap='Blues')

    def plot_model(self, proj, ax):
        """
        Plots the model proposed for the projection proj on ax.
        """
        x, y = proj
        evaluated_model = self.ig._eval_model()
        y_vals = getattr(evaluated_model, y.name)
        x_vals = self.ig._x_grid[x]
        plot = ax.errorbar(x_vals, y_vals, xerr=0, yerr=0, c='red')
        return plot

    def update_plot(self, indep_var, dep_var):
        """
        Updates the plot of the proposed model.
        """
        evaluated_model = self.ig._eval_model()
        y_vals = getattr(evaluated_model, dep_var.name)
        x_vals = self.ig._x_grid[indep_var]

        x_plot_data = []
        y_plot_data = []
        y_plot_error = []
        # TODO: Numpy magic
        # We need the error interval for every plotted point, so find all
        # the points plotted at x=x_i, and do some statistics on those.
        # Since all the points are on a grid made by meshgrid, the error
        # in x will alwys be 0.
        for x_val in self.ig._x_points[indep_var]:
            # We get away with this instead of digitize because x_vals is
            # on a grid made with meshgrid
            idx_mask = x_vals == x_val
            xs = x_vals[idx_mask]
            x_plot_data.append(xs[0])
            ys = y_vals[idx_mask]
            y_plot_data.append(np.mean(ys))
            y_error = np.percentile(ys, self.ig.percentile)
            y_plot_error.append(y_error)

        x_plot_data = np.array(x_plot_data)
        y_plot_data = np.array(y_plot_data)
        y_plot_error = np.array(y_plot_error)

        xs = np.column_stack((x_plot_data, x_plot_data))
        yerr = y_plot_error + y_plot_data[:, np.newaxis]
        y_segments = np.dstack((xs, yerr))
        plot_line, caps, error_lines = self.ig._plots[(indep_var, dep_var)]
        plot_line.set_data(x_plot_data, y_plot_data)
        error_lines[1].set_segments(y_segments)


class InteractiveGuess2D(InteractiveGuess):
    @deprecated(InteractiveGuess)
    def __init__(self, *args, **kwargs):
        # Deprecated as of 01/06/2017
        super(InteractiveGuess2D, self).__init__(*args, **kwargs)
