# -*- coding: utf-8 -*-
from symfit.contrib import interactive_guess
from symfit import Variable, Parameter, exp, latex
from symfit.distributions import Gaussian
import numpy as np
import unittest
import matplotlib.colors
import matplotlib.pyplot as plt

plt.ioff()
def distr(x, k, x0):
    kbT = 4.11
    return exp(-k*(x-x0)**2/kbT)


# Because sympy has issues with large-ish numpy arrays and broadcasting
def np_distr(x, k, x0):
    kbT = 4.11
    return np.exp(-k*(x-x0)**2/kbT)


class Gaussian2DInteractiveGuessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

        x = Variable('x')
        y = Variable('y')
        k = Parameter('k', 900)
        x0 = Parameter('x0', 1.5)

        cls.k = k
        cls.x0 = x0

        model = {y: distr(x, k, x0)}
        x_data = np.linspace(0, 2.5, 50)
        y_data = model[y](x=x_data, k=1000, x0=1)
        cls.guess = interactive_guess.InteractiveGuess(model, x=x_data, y=y_data)

    def test_number_of_sliders(self):
        self.assertEqual(len(self.guess._sliders), 2)

    def test_slider_labels(self):
        for parameter in self.guess.model.params:
            self.assertEqual(self.guess._sliders[parameter].ax.get_label(),
                             parameter.name)

    def test_slider_callback_parameter_values(self):
        new_val = np.random.random()
        other = self.guess.model.params[1].value
        self.guess._sliders[self.guess.model.params[0]].set_val(new_val)
        try:
            self.assertEqual(self.guess.model.params[0].value, new_val)
            self.assertEqual(self.guess.model.params[1].value, other)
        finally:
            self.guess._sliders[self.guess.model.params[0]].reset()

    def test_slider_callback_data(self):
        x = self.guess.model.independent_vars[0]
        x_points = self.guess._x_points[x]
        hi = np.max(x_points)
        lo = np.min(x_points)
        new_x = (hi - lo) * np.random.random() + lo
        new_k = 2000 * np.random.random()
        self.guess._sliders[self.k].set_val(new_k)
        self.guess._sliders[self.x0].set_val(new_x)
        try:
            kbT = 4.11
            true_data = np_distr(x_points, new_k, new_x)
            actual_data = self.guess._plots[self.guess._projections[0]].get_ydata()
            self.assertTrue(np.allclose(true_data, actual_data))
        finally:
            self.guess._sliders[self.k].reset()
            self.guess._sliders[self.x0].reset()

    def test_get_data(self):
        y = self.guess.model.dependent_vars[0]
        x = self.guess.model.independent_vars[0]
        x_points = self.guess._x_points[x]
        k = self.k.value
        x0 = self.x0.value
        kbT = 4.11
        true_y = np_distr(x_points, k, x0)
        data = self.guess._eval_model()
        actual_y = data.y
        actual_x = self.guess._x_points[x]
        self.assertTrue(np.allclose(x_points, actual_x) and
                        np.allclose(true_y, actual_y))

    def test_number_of_projections(self):
        self.assertEqual(len(self.guess._projections), 1)

    def test_number_of_plots(self):
        self.assertEqual(len(self.guess._plots), 1)

    def test_plot_titles(self):
        for proj in self.guess._projections:
            x, y = proj
            plot = self.guess._plots[proj]
            plotlabel = '${}({}) = {}$'.format(
                latex(y, mode='plain'),
                latex(x.name, mode='plain'),
                latex(self.guess.model[y], mode='plain'))
            self.assertEqual(plot.axes.get_title(), plotlabel)

    def test_plot_colors(self):
        for plot in self.guess._plots.values():
            color = matplotlib.colors.ColorConverter().to_rgb(plot.get_color())
            self.assertEqual(color, (1, 0, 0))


class VectorValuedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = Variable('x')
        y1 = Variable('y1')
        y2 = Variable('y2')
        k = Parameter('k', 900)
        x0 = Parameter('x0', 1.5)

        model = {y1: k * (x-x0)**2,
                 y2: x - x0}
        x_data = np.linspace(0, 2.5, 50)
        y1_data = model[y1](x=x_data, k=1000, x0=1)
        y2_data = model[y2](x=x_data, k=1000, x0=1)
        cls.guess = interactive_guess.InteractiveGuess(model, x=x_data, y1=y1_data, y2=y2_data)
#        plt.close(cls.fit.fig)

    def test_number_of_projections(self):
        self.assertEqual(len(self.guess._projections), 2)

    def test_number_of_plots(self):
        self.assertEqual(len(self.guess._plots), 2)

    def test_plot_titles(self):
        for proj in self.guess._projections:
            x, y = proj
            plot = self.guess._plots[proj]
            plotlabel = '${}({}) = {}$'.format(
                latex(y, mode='plain'),
                latex(x.name, mode='plain'),
                latex(self.guess.model[y], mode='plain'))
            self.assertEqual(plot.axes.get_title(), plotlabel)


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
#        xdata = np.dstack((xx, yy)).T # T because np fucks up conventions.

        x0 = Parameter('x0', value=0.6)
        sig_x = Parameter('sig_x', value=0.2, min=0.0)
        x = Variable('x')
        y0 = Parameter('y0', value=0.4)
        sig_y = Parameter('sig_y', value=0.1, min=0.0)
        A = Parameter('A')
        y = Variable('y')
        z = Variable('z')
        g = {z: A * Gaussian(x, x0, sig_x) * Gaussian(y, y0, sig_y)}
        cls.g = g
#        cls.xdata = xdata
#        cls.ydata = ydata
        cls.guess = interactive_guess.InteractiveGuess(g, x=xx.flatten(), y=yy.flatten(), z=ydata.flatten())
        
#        plt.close(cls.fit.fig)

    def test_number_of_projections(self):
        self.assertEqual(len(self.guess._projections), 2)

    def test_number_of_plots(self):
        self.assertEqual(len(self.guess._plots), 2)

    def test_plot_titles(self):
        for proj in self.guess._projections:
            x, y = proj
            plot = self.guess._plots[proj][0]
            plotlabel = '${}({}) = {}$'.format(latex(y, mode='plain'),
                                               latex(x.name, mode='plain'),
                                               latex(self.guess.model[y], mode='plain'))
            self.assertEqual(plot.axes.get_title(), plotlabel)


if __name__ == '__main__':
    unittest.main()
