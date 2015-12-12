from setuptools import setup
import sys

if sys.version_info >= (3,0):
    install_requires = ['sympy', 'numpy', 'scipy']
else:
    install_requires = ['sympy', 'numpy', 'scipy', 'funcsigs']

setup(
    name='symfit',
    version='0.3.1',
    description='Symbolic Fitting; fitting as it should be.',
    author='Martin Roelfs',
    author_email='m.roelfs@student.rug.nl',
    packages=['symfit', 'symfit.core', 'symfit.tests'],
    # long_description=long_description,
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