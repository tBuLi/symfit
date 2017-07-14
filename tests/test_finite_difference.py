#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:34:23 2017

@author: peterkroon
"""
import unittest
import symfit as sf
import numpy as np
import warnings


class FiniteDifferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)

    def test_1_1_model(self):
        '''Tests the case with 1 component and 1 parameter'''
        x, y = sf.variables('x, y')
        a = sf.Parameter(name='a')
        model = sf.Model({y: 3 * a * x**2})
        x_data = np.arange(10)

        exact = model.eval_jacobian(x=x_data, a=3.5)
        approx = model.finite_difference(x=x_data, a=3.5)
        np.testing.assert_allclose(exact, approx)

        exact = model.eval_jacobian(x=3, a=3.5)
        approx = model.finite_difference(x=3, a=3.5)
        np.testing.assert_allclose(exact, approx)

    def test_1_multi_model(self):
        '''Tests the case with 1 component and multiple parameters'''
        x, y = sf.variables('x, y')
        a, b = sf.parameters('a, b')
        model = sf.Model({y: 3 * a * x**2 - sf.exp(b) * x})
        x_data = np.arange(10)

        exact = model.eval_jacobian(x=x_data, a=3.5, b=2)
        approx = model.finite_difference(x=x_data, a=3.5, b=2)
        np.testing.assert_allclose(exact, approx)

        exact = model.eval_jacobian(x=3, a=3.5, b=2)
        approx = model.finite_difference(x=3, a=3.5, b=2)
        np.testing.assert_allclose(exact, approx)

    def test_multi_1_model(self):
        '''Tests the case with multiple components and one parameter'''
        x, y, z = sf.variables('x, y, z')
        a, = sf.parameters('a')
        model = sf.Model({y: 3 * a * x**2,
                          z: sf.exp(a*x)})
        x_data = np.arange(10)

        exact = model.eval_jacobian(x=x_data, a=3.5)
        approx = model.finite_difference(x=x_data, a=3.5)
        np.testing.assert_allclose(exact, approx)

        exact = model.eval_jacobian(x=3, a=3.5)
        approx = model.finite_difference(x=3, a=3.5)
        np.testing.assert_allclose(exact, approx)

    def test_multi_multi_model(self):
        '''Tests the case with multiple components and multiple parameters'''
        x, y, z = sf.variables('x, y, z')
        a, b, c = sf.parameters('a, b, c')
        model = sf.Model({y: 3 * a * x**2 + b * x - c,
                          z: sf.exp(a*x - b) * c})
        x_data = np.arange(10)

        exact = model.eval_jacobian(x=x_data, a=3.5, b=2, c=5)
        approx = model.finite_difference(x=x_data, a=3.5, b=2, c=5)
        np.testing.assert_allclose(exact, approx, rtol=1e-5)

        exact = model.eval_jacobian(x=3, a=3.5, b=2, c=5)
        approx = model.finite_difference(x=3, a=3.5, b=2, c=5)
        np.testing.assert_allclose(exact, approx, rtol=1e-5)

    def test_multi_indep(self):
        '''Tests the case with multiple components, multiple parameters and
            multiple independent variables'''
        w, x, y, z = sf.variables('w, x, y, z')
        a, b, c = sf.parameters('a, b, c')
        model = sf.Model({y: 3 * a * x**2 + b * x * w - c,
                          z: sf.exp(a*x - b) + c*w})
        x_data = np.arange(10)/10
        w_data = np.arange(10)

        exact = model.eval_jacobian(x=x_data, w=w_data, a=3.5, b=2, c=5)
        approx = model.finite_difference(x=x_data, w=w_data, a=3.5, b=2, c=5)
        np.testing.assert_allclose(exact, approx, rtol=1e-5)

        exact = model.eval_jacobian(x=0.3, w=w_data, a=3.5, b=2, c=5)
        approx = model.finite_difference(x=0.3, w=w_data, a=3.5, b=2, c=5)
        np.testing.assert_allclose(exact, approx, rtol=1e-5)

        exact = model.eval_jacobian(x=0.3, w=5, a=3.5, b=2, c=5)
        approx = model.finite_difference(x=0.3, w=5, a=3.5, b=2, c=5)
        np.testing.assert_allclose(exact, approx, rtol=1e-5)
        
    def test_ODE_stdev(self):
        """Make sure that parameters from ODEModels get standard deviations.
        """
        x, v, t = sf.variables('x, v, t')
        k = sf.Parameter(name='k')

        k.min = 0
        k.value = 10
        a = -k * x

        model = sf.ODEModel({
                             sf.D(v, t): a,
                             sf.D(x, t): v,
                             },
                            initial={v: 0, x: 1, t: 0})
        t_data = np.linspace(0, 10, 150)
        noise = np.random.normal(1, 0.05, t_data.shape)
        x_data = model(t=t_data, k=11).x * noise
        v_data = model(t=t_data, k=11).v * noise
        fit = sf.Fit(model, t=t_data, x=x_data, v=v_data)
        result = fit.execute()
        self.assertTrue(result.stdev(k) is not None)
        self.assertTrue(np.isfinite(result.stdev(k)))

if __name__ == '__main__':
    try:
        unittest.main(warnings='ignore')
        # Note that unittest will catch and handle exceptions raised by tests.
        # So this line will *only* deal with exceptions raised by the line
        # above.
    except TypeError:
        # In Py2, unittest.main doesn't take a warnings argument
        warnings.simplefilter('ignore')
        unittest.main()
