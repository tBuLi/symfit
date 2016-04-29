# -*- coding: utf-8 -*-

import numpy as np
from ...core.fit import TakesData
from ...core.support import keywordonly, key2str
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import itertools
import sympy.printing.latex

#from pkg_resources import parse_version

#SUPPORTED_VERSION = '0.3.0'
#SYMFIT_VERSION = symfit.version  # something something.
#
#if parse_version(SUPPORTED_VERSION) != parse_version(SYMFIT_VERSION):
#    raise EnvironmentError("Your symfit version is not supported. YMMV.")


class InteractiveGuess2D(TakesData):
    """A class that provides a visual_guess method which provides
    an graphical, interactive way of guessing initial fitting parameters."""

    @keywordonly(n_points=100)
    def __init__(self, *args, **kwargs):
        """Create a matplotlib window with sliders for all parameters
        in this model, so that you may graphically guess initial fitting
        parameters. n_points is the number of points drawn for the plot.
        Data points are plotted as blue points, the proposed model as
        a red line.
        Slider extremes are taken from the parameters where possible. If
        these are not provided, the minimum is 0; and the maximum is value*2.
        If no initial value is provided, it defaults to 1.

        This will modify the values of the parameters present in model.

        Parameters
        ----------
        n_points : int
            The number of points used for drawing the fitted function.
        """
        n_points = kwargs.pop('n_points')
        super(InteractiveGuess2D, self).__init__(*args, **kwargs)

        if len(self.independent_data) != 1:
            raise IndexError("Only 2D problems are supported.")

        self._projections = list(itertools.product(self.model.independent_vars,
                                                   self.model.dependent_vars))

        x_mins = {v: np.min(data) for v, data in self.independent_data.items()}
        x_maxs = {v: np.max(data) for v, data in self.independent_data.items()}
        for x in self.independent_data:
            plotrange_x = x_maxs[x] - x_mins[x]
            x_mins[x] -= 0.1 * plotrange_x
            x_maxs[x] += 0.1 * plotrange_x
        self._x_points = {v: np.linspace(x_mins[v], x_maxs[v], n_points)
                          for v in self.independent_data}

        y_mins = {v: np.min(data) for v, data in self.dependent_data.items()}
        y_maxs = {v: np.max(data) for v, data in self.dependent_data.items()}        
        for y in self.dependent_data:
            plotrange_y = y_maxs[y] - y_mins[y]
            y_mins[y] -= 0.1 * plotrange_y
            y_maxs[y] += 0.1 * plotrange_y
        
        self._set_up_figure(x_mins, x_maxs, y_mins, y_maxs)
        self._set_up_sliders()
        self.fig.show()

    def _set_up_figure(self, x_mins, x_maxs, y_mins, y_maxs):
        """
        Prepare the matplotlib figure: make all the subplots; adjust their
        x and y range; scatterplot the data; and plot an putative function.
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
            plotlabel = '${}({}) = {}$'.format(
                sympy.printing.latex(y, mode='plain'),
                x.name,
                sympy.printing.latex(self.model[y], mode='plain'))
            ax = self.fig.add_subplot(ncols, nrows, plotnr,
                                      label=plotlabel)
            ax.set_title(ax.get_label())
            ax.set_ylim(y_mins[y.name], y_maxs[y.name])
            ax.set_xlim(x_mins[x.name], x_maxs[x.name])
            # TODO reduce dimensionality.
            ax.scatter(self.independent_data[x.name],
                       self.dependent_data[y.name])

            vals = self._get_data(x, y)
            plot, = ax.plot(*vals, c='red')
            self._plots[proj] = plot

    def _set_up_sliders(self):
        """
        Creates an slider for every parameter.
        """
        i = 0.05
        self._sliders = {}
        for p in self.model.params:
            if not p.fixed:
                axbg = 'lightgoldenrodyellow'
            else:
                axbg = 'red'
            # start-x, start-y, width, height
            ax = self.fig.add_axes((0.162, i, 0.68, 0.03),
                                   axis_bgcolor=axbg, label=p.name)
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
        arguments = {independent_var: x_points}
        arguments.update({p: p.value for p in self.model.params})
        return x_points, self.model[dependent_var](**key2str(arguments))
