Project Goals
=============
## Why this Project?
Existing fitting modules are not very pythonic in their API and can be difficult for humans to use. This project aims to marry the power of scipy.optimize with SymPy to create an highly readable and easy to use fitting package which works for projects of any scale.

The example below shows how easy it is to define a model that we could fit to.
```python
from symfit.core.api import Parameter, Variable
import sympy

x0 = Parameter('x0')
sig = Parameter('sig')
x = Variable('x')
gaussian = sympy.exp(-(x - x0)**2/(2*sig**2))/(2*pi*sig)
```

Lets fit this model to some generated data.

```python
from symfit.core.api import Fit

x = # Some numpy array of x values
y = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, x, y)
fit_result = fit.execute()
```
Printing ```fit_result``` will give a full report on the values for every parameter, including the uncertainty, and quality of the fit.

Adding guesses for ```Parameter```'s is simple. Therefore, let's add another step: suppose we are able to estimate bounds for the parameter as well, for example by looking at a plot. We could then do this:

```python
from symfit.core.api import Fit, Parameter, Variable
import sympy

x0 = Parameter('x0', 2.0, min=1.5, max=2.5)
sig = Parameter('sig')
x = Variable('x')
gaussian = sympy.exp(-(x - x0)**2/(2*sig**2))/(2*pi*sig)

x = # Some numpy array of x values
y = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, x, y)
fit_result = fit.execute()
```

The ```Parameter``` options do not stop there. If a parameter is completely fixed during the fitting, we could use ```Parameter('x0', 2.0, fixed=True)``` which is mutually exclusive with the ```min, max``` keywords.

Using this paradigm it is easy to buil multivariable models and fit to them:

```python
from symfit.core.api import Parameter, Variable
from sympy import exp, pi

x0 = Parameter('x0')
y0 = Parameter('x0')
sig_x = Parameter('sig_x')
sig_y = Parameter('sig_y')
x = Variable('x')
y = Variable('y')
gaussian_2d = exp(-((x - x0)**2/(2*sig_x**2) + (y - y0)**2/(2*sig_y**2)))/(2*pi*sig_x*sig_y)
```

Because of the symbolic nature of this program, the Jacobian of the model can always be determined. Although scipy can approximate the Jacobian numerically, it is not always able to appraximate the covariance matrix from this, which we need to calculate the errors in our parameters.

This project will always be able to do as long, assuming your model is differentiable. This means we can do proper error propagation.
Advanced Usage
==============

#### Constrained minimization of multivariate scalar functions
(Not available yet, this is just to show of the symfit syntax for solving the same problem.)

Example taken from http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html

Suppose we want to maximize the following function:

![function](http://docs.scipy.org/doc/scipy/reference/_images/math/775ad8006edfe87928e39f1798d8f53849f7216f.png)

Subject to the following constraits:

![constraints](http://docs.scipy.org/doc/scipy/reference/_images/math/984a489a67fd94bcec325c0d60777d61c12c94f4.png)

In SciPy code the following lines are needed:
```python
def func(x, sign=1.0):
    """ Objective function """
    return sign*(2*x[0]*x[1] + 2*x[0] - x[0]**2 - 2*x[1]**2)
    
def func_deriv(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign*(-2*x[0] + 2*x[1] + 2)
    dfdx1 = sign*(2*x[0] - 4*x[1])
    return np.array([ dfdx0, dfdx1 ])
    
cons = ({'type': 'eq',
         'fun' : lambda x: np.array([x[0]**3 - x[1]]),
         'jac' : lambda x: np.array([3.0*(x[0]**2.0), -1.0])},
        {'type': 'ineq',
         'fun' : lambda x: np.array([x[1] - 1]),
         'jac' : lambda x: np.array([0.0, 1.0])})
         
res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
               constraints=cons, method='SLSQP', options={'disp': True})
```
Takes a couple of readthroughs to make sense, doesn't it? Let's do the same problem in SymFit:

```python
x = Variable('x')
y = Variable('y')
model = 2*x*y + 2*x - x**2 -2*y**2
constraints = [
	x**3 - y == 0,
    y - 1 >= 0,
]

fit = Minimize(model, constraints=constraints)
fit.execute()
```
Done! symfit will determine all derivatives automatically, no need for you to think about it. In order to be consistent with the name in SciPy, ```Minimize``` minimizes with respect to the variables, without taking into acount any data points. To minimize the parameters while constraining the variables, use ```MinimizeParameters``` instead.

```python
fit = MinimizeParameters(model, xdata, ydata, constraints=constraints)
```

Using ```MinimizeParameters``` without ```constraints``` in principle yields the same result as using ```Fit```, which does a least-squares fit. A case could therefore be made for always using ```MinimizeParameters```. However, I cannot comment on whether this is proper usage of the minimalize function.

####Optional Arguments

Knowing that symfit is (currently just) a wrapper to SciPy, you could decide to look in their documentation to specify extra options for the fitting. These extra arguments can be provided to ```execute```, as it will pass on any ```*args, **kwargs``` to leastsq or minimize depending on the context.


How Does it Work?
=================

####```AbstractFunction```'s
Comming soon

####```Argument```'s
Only two kinds input ```Argument``` are defined for a model: ```Variable``` and ```Parameter```.

### Immidiate Goals
- High code readability and a very pythonic feel.
- Efficient Fitting
- Fitting algorithms for any scale using scipy.optimize. From typical least squares fitting to Multivariant fitting with bounds and constraints using the overkill scipy.optimize.minimize.

### Long Term Goals
- Monte-Carlo
- Error Propagation using the uncertainties package

type: any python-type, such as float or int. default = float. 
