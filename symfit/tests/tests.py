import unittest
import sympy
from sympy import symbols
import numpy as np
from symfit.api import Variable, Parameter, Fit, FitResults
from symfit.functions import Gaussian, Exp
import scipy.stats
from symfit.core.support import sympy_to_scipy, sympy_to_py

class TddInPythonExample(unittest.TestCase):
    def test_gaussian(self):
        x0 = Parameter('x0')
        sig = Parameter('sig')
        x = Variable('x')
        new = sympy.exp(-(x - x0)**2/(2*sig**2))
        self.assertIsInstance(new, sympy.exp)
        g = Gaussian(x, x0, sig)
        print 'here:', g, type(g)
        self.assertTrue(issubclass(g.__class__, sympy.exp))

    def test_callable(self):
        a = Parameter()
        b = Parameter()
        x = Variable()
        y = Variable()
        func = a*x**2 + b*y**2
        result = func(x=2, y=3, a=3, b=9)
        self.assertEqual(result, 3*2**2 + 9*3**2)

        xdata = np.arange(1,10)
        ydata = np.arange(1,10)
        result = func(x=ydata, y=ydata, a=3, b=9)
        self.assertTrue(np.array_equal(result, 3*xdata**2 + 9*ydata**2))

    def test_fitting(self):
        xdata = np.linspace(1,10,10)
        ydata = 3*xdata**2

        a = Parameter('a')
        b = Parameter('b')
        x = Variable('x')
        new = a*x**b

        fit = Fit(new, xdata, ydata)

        func = sympy_to_py(new, [x], [a, b])
        result = func(xdata, 3, 2)
        self.assertTrue(np.array_equal(result, ydata))

        result = fit.scipy_func(fit.xdata, [3, 2])
        self.assertTrue(np.array_equal(result, ydata))

        import inspect
        args, varargs, keywords, defaults = inspect.getargspec(fit.scipy_func)

        # self.assertEqual(args, ['x', 'a', 'b'])
        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)
        print(fit_result)
        self.assertAlmostEqual(fit_result.params.a, 3.0)
        self.assertAlmostEqual(fit_result.params.b, 2.0)

        self.assertIsInstance(fit_result.params.a_stdev, float)
        self.assertIsInstance(fit_result.params.b_stdev, float)

        self.assertIsInstance(fit_result.r_squared, float)

        # Test several false ways to access the data.
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_fdska'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'c'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_stdev'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a_stdev_'])
        self.assertRaises(AttributeError, getattr, *[fit_result.params, 'a__stdev'])

    def test_numpy_functions(self):
        xdata = np.linspace(1,10,10)
        ydata = 45*np.log(xdata*2)

        a = Parameter('a')
        b = Parameter('b', value=2.1, fixed=True)
        x = Variable('x')
        new = a*sympy.log(x*b)


    def test_grid_fitting(self):
        xdata = np.arange(-5, 5, 1)
        ydata = np.arange(-5, 5, 1)
        xx, yy = np.meshgrid(xdata, ydata, sparse=False)
        xdata_coor = np.dstack((xx, yy))

        zdata = (2.5*xx**2 + 3.0*yy**2)

        a = Parameter('a')
        b = Parameter('b')
        x = Variable('x')
        y = Variable('y')
        new = (a*x**2 + b*y**2)

        fit = Fit(new, xdata_coor, zdata)

        # Test the flatten function for consistency.
        xdata_coor_flat, zdata_flat = fit._flatten(xdata_coor, zdata)
        # _flatten transposes such arrays because the variables are in the deepest dimension instead of the first.
        # This is normally not a problem because all we want from the fit is the correct parameters.
        self.assertFalse(np.array_equal(zdata, zdata_flat.reshape((10,10))))
        self.assertTrue(np.array_equal(zdata, zdata_flat.reshape((10,10)).T))
        self.assertFalse(np.array_equal(xdata_coor, xdata_coor_flat.reshape((10,10,2))))
        new_xdata = xdata_coor_flat.reshape((2,10,10)).T
        self.assertTrue(np.array_equal(xdata_coor, new_xdata))


        results = fit.execute()
        self.assertAlmostEqual(results.params.a, 2.5)
        self.assertAlmostEqual(results.params.b, 3.)


    def test_2D_fitting(self):
        xdata = np.random.randint(-10, 11, size=(2, 400))
        zdata = 2.5*xdata[0]**2 + 7.0*xdata[1]**2

        a = Parameter('a')
        b = Parameter('b')
        x = Variable('x')
        y = Variable('y')
        new = a*x**2 + b*y**2

        fit = Fit(new, xdata, zdata)

        result = fit.scipy_func(fit.xdata, [2, 3])

        import inspect
        args, varargs, keywords, defaults = inspect.getargspec(fit.scipy_func)
        self.assertEqual(args, ['x', 'p'])

        fit_result = fit.execute()
        self.assertIsInstance(fit_result, FitResults)

    def test_gaussian_fitting(self):
        xdata = 2*np.random.rand(10000) - 1 # random betwen [-1, 1]
        ydata = scipy.stats.norm.pdf(xdata, loc=0.0, scale=1.0)

        x0 = Parameter()
        sig = Parameter()
        A = Parameter()
        x = Variable()
        g = A * Gaussian(x, x0, sig)

        fit = Fit(g, xdata, ydata)
        fit_result = fit.execute()
        print fit_result
        self.assertAlmostEqual(fit_result.params.A, 0.3989423)
        self.assertAlmostEqual(np.abs(fit_result.params.sig), 1.0)
        self.assertAlmostEqual(fit_result.params.x0, 0.0)


    # def test_minimize(self):
    #     x = Parameter()
    #     y = Parameter()
    #     model = 2*x*y + 2*x - x**2 - 2*y**2
    #     from sympy import Eq
    #     constraints = [
    #         x**3 - y == 0,
    #         y - 1 >= 0,
    #     ]
    #     print(type(x**3 - y))
    #     print x**3 - y == 0.0
    #     self.assertIsInstance(constraints[0], Eq)
    #
    #     fit = Maximize(model, constraints=constraints)
    #     fit_result = fit.execute()
    #     self.assertAlmostEqual(fit_result.x[0], 1.00000009)
    #     self.assertAlmostEqual(fit_result.x[1], 1.)

    def test_parameter_add(self):
        a = Parameter('a')
        b = Parameter('b')
        new = a + b
        self.assertIsInstance(new, sympy.Add)

    def test_symbol_add(self):
        x, y = symbols('x y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

    def test_evaluate_model(self):
        A = Parameter()
        x = Variable()
        new = A * x ** 2

        self.assertEqual(new(x=2, A=2), 8)
        self.assertNotEqual(new(x=2, A=3), 8)

    def test_symbol_object_add(self):
        from sympy.core.symbol import Symbol
        x = Symbol('x')
        y = Symbol('y')
        new = x + y
        self.assertIsInstance(new, sympy.Add)

if __name__ == '__main__':
    unittest.main()
