Documentation
=============
http://symfit.readthedocs.org

> WARNING: This readme is currently outdated. Use the docs for reference, not this readme.

Project Goals
=============
## Why this Project?
Existing fitting modules are not very pythonic in their API and can be difficult for humans to use. This project aims to marry the power of scipy.optimize with SymPy to create an highly readable and easy to use fitting package which works for projects of any scale.

The example below shows how easy it is to define a model that we could fit to.
```python
from symfit import Parameter, Variable, exp, pi

x0 = Parameter()
sig = Parameter()
x = Variable()
gaussian = exp(-(x - x0)**2/(2 * sig**2)) / (2 * pi * sig)
```

Lets fit this model to some generated data.

```python
from symfit import Fit

xdata = # Some numpy array of x values
ydata = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, xdata, ydata)
fit_result = fit.execute()
```
Printing ```fit_result``` will give a full report on the values for every parameter, including the uncertainty, and quality of the fit.

Adding guesses for ```Parameter```'s is simple: ```Parameter(1.0)``` or ```Parameter{value=1.0)```. Let's add another step: suppose we are able to estimate bounds for the parameter as well, for example by looking at a plot. We could then do this: ```Parameter(2.0, min=1.5, max=2.5)```. Complete example:

```python
from symfit import Fit, Parameter, Variable, exp, pi

x0 = Parameter(2.0, min=1.5, max=2.5)
sig = Parameter()
x = Variable()
gaussian = exp(-(x - x0)**2/(2 * sig**2)) / (2 * pi * sig)

xdata = # Some numpy array of x values
ydata = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, xdata, ydata)
fit_result = fit.execute()
```

The ```Parameter``` options do not stop there. If a parameter is completely fixed during the fitting, we could use ```Parameter(2.0, fixed=True)``` which is mutually exclusive with the ```min, max``` keywords.

Using this paradigm it is easy to build multivariable models and fit to them:

```python
from symfit.api import Parameter, Variable, exp, pi

x0 = Parameter()
y0 = Parameter()
sig_x = Parameter()
sig_y = Parameter()
x = Variable()
y = Variable()
gaussian_2d = exp(-((x - x0)**2/(2*sig_x**2) + (y - y0)**2/(2*sig_y**2)))/(2*pi*sig_x*sig_y)
```

Because of the symbolic nature of this program, the Jacobian of the model can always be determined. Although scipy can approximate the Jacobian numerically, it is not always able to approximate the covariance matrix from this. But this is needed if we want to calculate the errors in our parameters.

This project will always be able to do as long, assuming your model is differentiable. This means we can do proper error propagation.

##Models are Callable
```python 
a = Parameter()
x = Variable()
f = a * x**2
print f(x=3, a=2)
```
They must always be called through keyword arguments to prevent any ambiguity in which parameter or variable you mean.

####Optional Arguments

Knowing that symfit is (currently just) a wrapper to SciPy, you could decide to look in their documentation to specify extra options for the fitting. These extra arguments can be provided to ```execute```, as it will pass on any ```*args, **kwargs``` to leastsq or minimize depending on the context.

FitResults
==========
The FitResults object which is returned by Fit.execute contains all information about the fit. Let's look at this by looking at an example:
```python
from symfit.api import Fit, Parameter, Variable
import sympy

x0 = Parameter(2.0, min=1.5, max=2.5)
sig = Parameter()
x = Variable()
gaussian = sympy.exp(-(x - x0)**2/(2*sig**2))/(2*sympy.pi*sig)

xdata = # Some numpy array of x values
ydata = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, xdata, ydata)
fit_result = fit.execute()

print fit_result.params.x0  # Print the value of x0
print fit_result.params.x0_stdev  # stdev in x0 as obtained from the fit.

try:
    print fit_result.params.x
except AttributeError:  # This will fire
    print 'No such Parameter'
    
print fit_result.r_squared  # Regression coefficient
```
The value/stdev of a parameter can also be obtained in the following way:
```python
fit_result.params.get_value(x0)
fit_result.params.get_stdev(x0)
```


