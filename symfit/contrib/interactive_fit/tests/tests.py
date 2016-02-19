# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:25:07 2016

@author: peterkroon
"""

from symfit.contrib import interactive_fit
from symfit import Variable, Parameter, exp
from symfit.distributions import Gaussian
import numpy as np
import unittest
import matplotlib.colors


def distr(x, k, x0):
    kbT = 4.11
    return exp(-k*(x-x0)**2/kbT)


class Gaussian2DInteractiveFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

        x = Variable()
        y = Variable()
        k = Parameter(900)
        x0 = Parameter(1.5)

        # You can NOT do this in one go. Blame Sympy. Not my fault.
        cls.k = k
        cls.x0 = x0

        model = {y: distr(x, k, x0)}
        x_data = np.linspace(0, 2.5, 50)
        y_data = model[y](x=x_data, k=1000, x0=1)
        cls.fit = interactive_fit.InteractiveFit2D(model, x=x_data, y=y_data)

    def test_number_of_sliders(self):
        self.assertEqual(len(self.fit._sliders), 2)

    def test_slider_labels(self):
        for parameter in self.fit.model.params:
            self.assertEqual(self.fit._sliders[parameter].ax.get_label(),
                             parameter.name)

    def test_slider_callback_parameter_values(self):
        new_val = np.random.random()
        self.fit._sliders[self.fit.model.params[0]].set_val(new_val)
        other = self.fit.model.params[1].value
        try:
            self.assertEqual(self.fit.model.params[0].value, new_val)
            self.assertEqual(self.fit.model.params[1].value, other)
        finally:
            self.fit._sliders[self.fit.model.params[0]].reset()

    def test_slider_callback_data(self):
        x_points = self.fit._x_points['x']
        hi = np.max(x_points)
        lo = np.min(x_points)
        new_x = (hi - lo) * np.random.random() + lo
        new_k = 2000 * np.random.random()
        self.fit._sliders[self.k].set_val(new_k)
        self.fit._sliders[self.x0].set_val(new_x)
        try:
            kbT = 4.11
            true_data = np.exp(-new_k*(x_points-new_x)**2/kbT)
            actual_data = self.fit._plots[self.fit._projections[0]].get_ydata()
            self.assertTrue(np.allclose(true_data, actual_data))
        finally:
            self.fit._sliders[self.k].reset()
            self.fit._sliders[self.x0].reset()

    def test_get_data(self):
        y = self.fit.model.dependent_vars[0]
        x = self.fit.model.independent_vars[0]
        x_points = x_points = self.fit._x_points['x']
        k = self.k.value
        x0 = self.x0.value
        kbT = 4.11
        true_y = np.exp(-k*(x_points-x0)**2/kbT)
        actual_x, actual_y = self.fit._get_data(x, y)
        self.assertTrue(np.allclose(x_points, actual_x) and
                        np.allclose(true_y, actual_y))

    def test_number_of_projections(self):
        self.assertEqual(len(self.fit._projections), 1)

    def test_number_of_plots(self):
        self.assertEqual(len(self.fit._plots), 1)

    def test_plot_titles(self):
        for proj in self.fit._projections:
            plot = self.fit._plots[proj]
            self.assertEqual(plot.axes.get_title(),
                             "{} {}".format(proj[0].name, proj[1].name))

    def test_plot_colors(self):
        for plot in self.fit._plots.values():
            color = matplotlib.colors.ColorConverter().to_rgb(plot.get_color())
            self.assertEqual(color, (1, 0, 0))


class VectorValuedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        cls.fit = interactive_fit.InteractiveFit2D(model, x=x_data, y1=y1_data, y2=y2_data)

    def test_number_of_projections(self):
        self.assertEqual(len(self.fit._projections), 2)

    def test_number_of_plots(self):
        self.assertEqual(len(self.fit._plots), 2)

    def test_plot_titles(self):
        for proj in self.fit._projections:
            plot = self.fit._plots[proj]
            self.assertEqual(plot.axes.get_title(),
                             "{} {}".format(proj[0].name, proj[1].name))


@unittest.skip("3D problems are not yet supported")
class Gaussian3DInteractiveFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mean = (0.6,0.4) # x, y mean 0.6, 0.4
        cov = [[0.2**2,0],[0,0.1**2]]
        data = np.random.multivariate_normal(mean, cov, 1000000)

        # Insert them as y,x here as np fucks up cartesian conventions.
        ydata, xedges, yedges = np.histogram2d(data[:,0], data[:,1], bins=100, range=[[0.0, 1.0], [0.0, 1.0]])
        xcentres = (xedges[:-1] + xedges[1:]) / 2
        ycentres = (yedges[:-1] + yedges[1:]) / 2

        # Make a valid grid to match ydata
        xx, yy = np.meshgrid(xcentres, ycentres, sparse=False)
        xdata = np.dstack((xx, yy)).T # T because np fucks up conventions.

        x0 = Parameter(0.6)
        sig_x = Parameter(0.2, min=0.0)
        x = Variable()
        y0 = Parameter(0.4)
        sig_y = Parameter(0.1, min=0.0)
        A = Parameter()
        y = Variable()
        g = A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)
        cls.g = g
        cls.xdata = xdata
        cls.ydata = ydata
        cls.fit = interactive_fit.InteractiveFit2D(g, xdata, ydata)

    def test_number_of_projections(self):
        self.assertEqual(len(self.fit._projections), 2)

    def test_number_of_plots(self):
        self.assertEqual(len(self.fit._plots), 2)

    def test_plot_titles(self):
        for proj in self.fit._projections:
            plot = self.fit._plots[proj]
            self.assertEqual(plot.axes.get_title(),
                             "{} {}".format(proj[0].name, proj[1].name))


if __name__ == '__main__':
    unittest.main()