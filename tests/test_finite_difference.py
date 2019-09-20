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
        '''
        Tests the case with multiple components, multiple parameters and
        multiple independent variables
        '''
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
        """
        Make sure that parameters from ODEModels get standard deviations.
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

    def test_unequal_data(self):
        """
        Test to make sure finite differences work with data of unequal length.
        """
        x_1, x_2, y_1, y_2 = sf.variables('x_1, x_2, y_1, y_2')
        y0, a_1, a_2, b_1, b_2 = sf.parameters('y0, a_1, a_2, b_1, b_2')

        model = sf.Model({
            y_1: a_1 * x_1**2 + b_1 * x_1 + y0,
            y_2: a_2 * x_2**2 + b_2 * x_2 + y0,
        })

        # Generate data from this model
        xdata1 = np.linspace(0, 10)
        xdata2 = xdata1[::2]  # Only every other point.

        exact = model.eval_jacobian(x_1=xdata1, x_2=xdata2,
                                    a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
        approx = model.finite_difference(x_1=xdata1, x_2=xdata2,
                                         a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111, y0=10.8)
        # First axis is the number of components
        self.assertEqual(len(exact), 2)
        self.assertEqual(len(approx), 2)

        # Second axis is the number of parameters, same for all components
        for exact_comp, approx_comp, xdata in zip(exact, approx, [xdata1, xdata2]):
            self.assertEqual(len(exact_comp), len(model.params))
            self.assertEqual(len(approx_comp), len(model.params))
            for exact_elem, approx_elem in zip(exact_comp, approx_comp):
                self.assertEqual(exact_elem.shape, xdata.shape)
                self.assertEqual(approx_elem.shape, xdata.shape)

        self._assert_equal(exact, approx, rtol=1e-4)

        model = sf.Model({
            y_1: a_1 * x_1**2 + b_1 * x_1,
            y_2: a_2 * x_2**2 + b_2 * x_2,
        })

        exact = model.eval_jacobian(x_1=xdata1, x_2=xdata2,
                                    a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111)
        approx = model.finite_difference(x_1=xdata1, x_2=xdata2,
                                         a_1=101.3, b_1=0.5, a_2=56.3, b_2=1.1111)
        self._assert_equal(exact, approx, rtol=1e-4)

        model = sf.Model({
            y_1: a_1 * x_1**2 + b_1 * x_1,
        })
        exact = model.eval_jacobian(x_1=xdata1, a_1=101.3, b_1=0.5)
        approx = model.finite_difference(x_1=xdata1, a_1=101.3, b_1=0.5)
        self._assert_equal(exact, approx, rtol=1e-4)

    def test_harmonic_oscillator_errors(self):
        """
        Make sure the errors produced by fitting ODE's are the same as when
        fitting an exact solution.
        """
        x, v, t = sf.variables('x, v, t')
        k = sf.Parameter(name='k', value=100)
        m = 1
        a = -k/m * x
        ode_model = sf.ODEModel({sf.D(v, t): a,
                                 sf.D(x, t): v},
                                initial={t: 0, v: 0, x: 1})

        t_data = np.linspace(0, 10, 250)
        np.random.seed(2)
        noise = np.random.normal(1, 0.05, size=t_data.shape)
        x_data = ode_model(t=t_data, k=100).x * noise

        ode_fit = sf.Fit(ode_model, t=t_data, x=x_data, v=None)
        ode_result = ode_fit.execute()

        phi = 0
        A = 1
        model = sf.Model({x: A * sf.cos(sf.sqrt(k/m) * t + phi)})
        fit = sf.Fit(model, t=t_data, x=x_data)
        result = fit.execute()

        self.assertAlmostEqual(result.value(k), ode_result.value(k), places=4)
        self.assertAlmostEqual(result.stdev(k) / ode_result.stdev(k), 1, 2)
        self.assertGreaterEqual(result.stdev(k), ode_result.stdev(k))

    def _assert_equal(self, exact, approx, **kwargs):
        self.assertEqual(len(exact), len(approx))
        for exact_comp, approx_comp in zip(exact, approx):
            np.testing.assert_allclose(exact_comp, approx_comp, **kwargs)


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
