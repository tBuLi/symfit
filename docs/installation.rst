Installation
============
If you are using pip, you can simply run ::

  pip install symfit

from your terminal. If you prefer to use `conda`, run ::

  conda install -c conda-forge symfit

instead. Lastly, if you prefer to install manually you can download
the source from https://github.com/tBuLi/symfit.

symmip module
--------------
To use `symfit`'s :class:`~symfit.symmip.mip.MIP` object for mixed integer programming (MIP) and
mixed integer nonlinear programming (MINLP), you need to have a suitable backend installed.
Because this is an optional feature, no such solver is installed by default.
In order to install the non-commercial SCIPOpt package, install `symfit` by running

  pip install symfit[symmip]

Contrib module
--------------
To also install the dependencies of 3rd party contrib modules such as
interactive guesses, install `symfit` using::

  pip install symfit[contrib]

Dependencies
------------
See `requirements.txt` for a full list.
