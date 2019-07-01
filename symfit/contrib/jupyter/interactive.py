import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import FloatSlider, interact


class Interactive(object):
    def __init__(self, fit, num=100):
        self.fit = fit
        self.num = num

        self.parameters = {p.name: p for p in self.fit.model.params} # Ahh the pleasure of having a dict with names!

    def interact(self):
        widgets = {k: FloatSlider(v.value, min=v.min, max=v.max, step=(v.max - v.min)/100) for k, v in self.parameters.items()}
        self.initialize()

        interact(self.update_func, **widgets)

    def get_x(self):
        self.x_name = list(self.fit.independent_data.keys())[0].name
        self.x_d = list(self.fit.independent_data.values())[0]
        self.x_arr = np.linspace(self.x_d.min(), self.x_d.max(), num=self.num, endpoint=True)

    def initialize(self):
        fig, axes = plt.subplots()
        assert len(self.fit.independent_data) == 1, 'Can only plot data with one independent data component'

        self.get_x()
        ans = self.fit.model(**{self.x_name: self.x_arr}, **{k: v.value for k, v in self.parameters.items()})

        self.lines = {}
        for k in self.fit.dependent_data.keys():
            l, = plt.plot(self.x_arr, getattr(ans, k.name), label=k.name)
            self.lines[k] = l

        for v in self.fit.dependent_data.values():
            plt.scatter(self.x_d, v)

        plt.legend()

        # return fig, axes

    def update_func(self, **kwargs):
        ans = self.fit.model(**{self.x_name: self.x_arr}, **{k: float(v) for k, v in kwargs.items()})

        for k, v in kwargs.items():
            self.parameters[k].value = float(v)

        for k in self.lines.keys():
            self.lines[k].set_ydata(getattr(ans, k.name))

    def plot_result(self, result):  #todo multiple subplots for multiple vars?
        fig, ax = plt.subplots()
        assert len(self.fit.independent_data) == 1, 'Can only plot data with one independent data component'
        self.get_x()

        ans = self.fit.model(**{self.x_name: self.x_arr}, **result.params)

        for k in self.fit.dependent_data.keys():
            l, = plt.plot(self.x_arr, getattr(ans, k.name), label=k.name)

        for v in self.fit.dependent_data.values():
            plt.scatter(self.x_d, v)

        plt.legend()
        return fig, ax