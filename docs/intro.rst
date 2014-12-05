Introduction
============

Why This Project
----------------

Existing fitting modules are not very pythonic in their API and can be difficult for humans to use. This project aims to marry the power of ``scipy.optimize`` with the readability of ``SymPy`` to create a highly readable and easy to use fitting package which works for projects of any scale.

This symbolic approach turns out to have great technical advantages over using scipy directly. In order to fit, the algorithm needs the Jacobian: a matrix containing the derivatives of your model in it's parameters. Because of the symbolic nature of ``symfit``, this is determined for you on the fly, saving you the trouble of having to determine the derivatives yourself. Furthermore, having this Jacobian allows good estimation of the errors in your parameters, something ``scipy`` does not always succeed in.

Examples
--------

See more in the example library here: :ref:`example-library`