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
gaussian = sympy.exp(-(x - x0)**2/(2*sig**2))
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
gaussian = sympy.exp(-(x - x0)**2/(2*sig**2))

x = # Some numpy array of x values
y = # Some numpy array of y values, gaussian distribution
fit = Fit(gaussian, x, y)
fit_result = fit.execute()
```

The ```Parameter``` options do not stop there. If a parameter is completely fixed during the fitting, we could use ```Parameter('x0', 2.0, fixed=True)``` which is mutually exclusive with the ```min, max``` keywords.

Using this paradigm it is easy to buil multivariable models and fit to them:

```python
from symfit.core.api import Parameter, Variable
import sympy

x0 = Parameter('x0')
y0 = Parameter('x0')
sig_x = Parameter('sig_x')
sig_y = Parameter('sig_y')
x = Variable('x')
y = Variable('y')
gaussian_2d = # Comming soon
```

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
