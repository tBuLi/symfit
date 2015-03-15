from distutils.core import setup

setup(
    name='symfit',
    version='0.2.4',
    description='Symbolic Fitting; fitting as it should be.',
    author='Martin Roelfs',
    author_email='m.roelfs@rug.nl',
    packages=['symfit', 'symfit.core'],
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
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],

    # What does your project relate to?
    keywords='fit fitting symbolic',

    # install_requires = ['sympy', 'numpy', 'scipy'],
)