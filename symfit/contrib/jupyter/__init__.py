from ipywidgets import FloatSlider, interact

from ipywidgets import FloatSlider, interact
import matplotlib.pyplot as plt
import numpy as np

#todo complain about dashes not allowed in parameter names


class Interactive(object):
    """mixin for smitting.fitting Fitting objects"""

    def __init__(self, fit):
        self.fit = fit

    def interactive(self):
        widgets = {name: FloatSlider(v.value, min=v.min, max=v.max, step=(v.max - v.min)/100) for name, v in self.fit.model.parameters.items()}
        self.initialize()

        interact(self.update_func, **widgets)

    def get_x(self):
        x_name = list(self.independent_data.keys())[0]  # in smitting this is a string, not some silly parameter object
        x_d = list(self.independent_data.values())[0]
        x_arr = np.linspace(x_d.min(), x_d.max(), num=100, endpoint=True)
        self._x_name = x_name
        self._x_arr = x_arr
        return x_name, x_d, x_arr

    def initialize(self):
        fig, axes = plt.subplots()
        assert len(self.independent_data) == 1, 'Can only plot data with one independent data component'

        x_name, x_d, x_arr = self.get_x()
        ans = self.model(**{x_name: x_arr}, **{k: v.value for k, v in self.parameters.items()})

        self.lines = {}
        for k in self.dependent_data.keys():
            l, = plt.plot(x_arr, getattr(ans, k), label=k)
            self.lines[k] = l

        for v in self.dependent_data.values():
            plt.scatter(x_d, v)

        plt.legend()

        # return fig, axes

    def update_func(self, **kwargs):
        ans = self.model(**{self._x_name: self._x_arr}, **{k: float(v) for k, v in kwargs.items()})
        for k, v in kwargs.items():
            self.parameters[k].value = float(v)

        for k in self.lines.keys():
            self.lines[k].set_ydata(getattr(ans, k))

    def plot_result(self):  #todo multiple subplots for multiple vars?
        fig, ax = plt.subplots()
        assert len(self.independent_data) == 1, 'Can only plot data with one independent data component'
        assert self.res is not None, 'No FitResult available, execute fitting first'

        x_name, x_d, x_arr = self.get_x()
        ans = self.model(**{x_name: x_arr}, **self.res.params)

        for k in self.dependent_data.keys():
            l, = plt.plot(x_arr, getattr(ans, k), label=k)

        for v in self.dependent_data.values():
            plt.scatter(x_d, v)

        plt.legend()
        return fig, ax