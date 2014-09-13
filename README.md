Project Goals
=============
## Why this Project?
Existing fitting modules are not pythonic in their API and can be difficult for humans to use.

### Immidiate Goals
- High code readability and a very pythonic feel.
- Efficient Fitting
- Multivariant fitting with bounds and constraints using the overkill scipy.optimize.minimize.

### Long Term Goals
- Monte-Carlo
- Error Propagation using the uncertainties package


Function Examples
=================
```python
from functions.api import Gaussian

a = Gaussian(mu = 1.0, sig = 1.0)
print a(3) # a eveluated at 3

# Adition of distributions
a = Gaussian(mu = 1.0, sig = 1.0) + Gaussian(mu = 3.0, sig = 1.0)
print a(3)

# Addition or multiplication with a constant:
a = 5.0 + 1.5 * Gaussian(mu = 1.0, sig = 1.0)
```

Fitting Example
===============
```python
from functions.fit.api import Fit
from functions.api import Gaussian
import numpy as np

# Random x, y in certain range
x = np.random.random(...)
y = Gaussian(mu=1.0, sig=1.0)(x)

# Hazard a guess:
model = Gaussian(mu=2.0, sig=0.5)
fit = Fit(model, x, y)
fit.execute()
```

Now let's try something a bit more challenging:
Fitting with bounds we were able to guess by for example plotting the data. In order to do this we need a ```Parameter``` object, the most versatile building block of this package.
```python
# Hazard a guess:
model = Gaussian(mu=Parameter(1.0, min=0.0, max=2.0), sig=0.5)
fit = Fit(model, x, y)
fit.execute()
```
The ```Parameter``` options do not stop there. If a parameter is completely fixed during the fitting, you could use ```Parameter(1.0, fixed=True)``` which is mutually exclusive with the ```min, max``` keywords.

Furthermore, it can be used when a certain parameter appears in multiple places in the model, or they are in some other way related. Take the following example:
```python
# For fitting some parameter might not be independent.
from functions.fitting.api import Parameter
x0 = Paramater(1.0)
model = Gaussian(mu = x0, sig = 1.0) + Gaussian(mu = x0, sig = 2.0)
# The parameter basically produces a pointer in python.
```
Or even more extreme:
```python
model = Gaussian(mu = x0, sig = 1.0) + Gaussian(mu = 2 * x0, sig = 2.0)
```

How Does it Work?
=================

####```AbstractFunction```'s
Things like ```Gaussian``` are all subclasses of ```AbstractFunction```, which transforms under mathematical operations as one would expect. Most ```AbstractFunction``` objects have only one ```Variable``` x and are therefore callable objects as we have seen in the examples. (e.g. ```Gaussian(mu=1.0, sig=1.0)(x=3)```)

Performing mathematical operations on ```AbstractFunction``` subclasses results in a new instance of ```AbstractFunction```, containing as its .func method the result of combining the previous two according to the specified operator.

####```Argument```'s
Only two kinds input ```Argument``` are defined for a model: ```Variable``` and ```Parameter```. When doing something like 
```python
# Addition or multiplication with a constant:
a = 5.0 + 1.5 * Gaussian(mu = 1.0, sig = 1.0)
```
It appears as though we are adding floats to the Gaussian. But addition and multiplication are overloaded in such a way that in reality, this expression returns a new ```AbstractFunction``` object which is equivalent to doing 
```python
Paramater(5.0, fixed=True) + Paramater(1.5, fixed=True) * Gaussian(mu=Paramater(1.0, fixed=True), sig=Paramater(1.0, fixed=True))
```
Multi-Variable
==============
To do multi-variable fitting, we must explicitly declare our variables.
```python
from functions.api import Exp
x = Variable()
y = Variable()
model = 2*x*y + y*Exp(k=1.0, x=x)
fit = Fit(model, x_data, y_data, z_data)
fit.execute()
```
API
===

Variable(Argument)

type: any python-type, such as float or int. default = float. 