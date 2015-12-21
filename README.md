Documentation
=============
http://symfit.readthedocs.org

Project Goals
=============

The goal of this project is simple: to make fitting in Python sexy and pythonic. What does pythonic fitting look like? 
Well, there's a simple test. 
If I can give you pieces of example code and don't have to use any additional words to explain what it does, it's pythonic.

```python
from symfit import parameters, Maximize, Equality, GreaterThan

x, y = parameters('x, y')
model = 2 * x * y + 2 * x - x**2 - 2 * y**2
constraints = [
    Equality(x**3 - y, 0),
    GreaterThan(y - 1, 0),
]

fit = Maximize(model, constraints=constraints)
fit_result = fit.execute()
```

Need I say more? How about I let another code example do the talking?

```python
from symfit import parameters, variables, Fit

a, b = parameters('a, b')
x, y = variables('x, y')
model = {y: a * x + b}

fit = Fit(model, x=xdata, y=ydata, sigma_y=sigma)
fit_result = fit.execute()
```

Cool right? So now that we have done a fit, how do can we use the results?

```python
import matplotlib.pyplot as plt

y = model(x=xdata, **fit_result.params)
plt.plot(xdata, y)
plt.show()
```

<img src="http://symfit.readthedocs.org/en/latest/_images/linear_model_fit.png" alt="Linear Fit" width="200px">

For more, check the docs at http://symfit.readthedocs.org.