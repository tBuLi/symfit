from setuptools import setup
import sys

if sys.version_info >= (3,0):
    install_requires = ['sympy', 'numpy', 'scipy']
else:
    install_requires = ['sympy', 'numpy', 'scipy', 'funcsigs']

long_description = '''
Documentation: http://symfit.readthedocs.org/

This project aims to marry the power of ``scipy.optimize`` with the readability of ``SymPy`` to create a highly readable and easy to use fitting package which works for projects of any scale.

``symfit`` is designed to be very readable::

	x = variables('x')
	A, sig, x0 = parameters('A, sig, x0')

	# Gaussian distribution
	gaussian = A * exp(-(x - x0)**2 / (2 * sig**2))

	fit = Fit(gaussian, xdata, ydata)
	fit_result = fit.execute()

You can also name dependent variables, allowing for sexy assignment of data::

	x, y = variables('x, y')
	model = {y: a * x**2}

	fit = Fit(model, x=xdata, y=ydata, sigma_y=sigma)
	fit.execute()

Constraint maximization has never been this easy::

	x, y = parameters('x, y')
	model = 2*x*y + 2*x - x**2 -2*y**2
	constraints = [
	    Eq(x**3 - y, 0),
	    Ge(y - 1, 0),
	]

	fit = Maximize(model, constraints=constraints)
	fit_result = fit.execute()

And evaluating a model with the best fit parameters is easy since ``symfit`` expressions are callable::

	y = gaussian(x=xdata, **fit_result.params)

.. figure:: http://symfit.readthedocs.org/en/latest/_images/gaussian_intro.png
   :width: 500px
   :alt: Gaussian Data

For many more features such as bounds on ``Parameter``'s, maximum-likelihood fitting, and much more check the docs at http://symfit.readthedocs.org/.

You can find ``symfit`` on github at https://github.com/tBuLi/symfit.
'''

setup(
    name='symfit',
    version='0.3.3',
    description='Symbolic Fitting; fitting as it should be.',
    author='Martin Roelfs',
    author_email='m.roelfs@student.rug.nl',
    packages=['symfit', 'symfit.core', 'symfit.tests'],
    long_description=long_description,
    url='https://github.com/tBuLi/symfit', #'symfit.readthedocs.org',
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],

    # What does your project relate to?
    keywords='fit fitting symbolic',

    install_requires=install_requires,
)