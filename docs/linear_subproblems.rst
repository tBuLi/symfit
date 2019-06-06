On Linear (Sub)problems & Linear Programming
============================================

``symfit`` models are allowed to have linear (sub)problems. These are defined as

.. math::

	\text{solve} A x &= y \\
	\text{such that} c_l \leq &x \geq c_u

where :math:`c_l` and :math:`c_u` represent the upper and lower bounds on each
component respectively, :math:`A` is a matrix, :math:`x` is a vector (matrix)
and :math:`y` is a vector (matrix).

In code, the above would look as follows::

	x = Parameter('x', min=c_l, max=c_u)
	x = MatrixSymbol(x, M, N)
	A = MatrixSymbol('A', L, M)
	y = MatrixSymbol('y', L, N)

	model = CallableModel({y: A * x})
	fit = Fit(model, A=A_data, y_data)

where ``L``, ``M`` and ``N`` are integers representing the shapes of these
matrices. Based on the presense of bounds, ``Fit`` will decide on a linear
solver to use. If no bounds are present, ``numpy.linalg.lstsq`` is used.
In the presense of bounds, ``scipy.optimize.lsq_linear`` is used instead.

Both of these methods support rectangular matrices :math:`A`, which is why we
use them. In the case of a square matrix the result should be identical to that
of ``numpy.linalg.solve``, but this implementation is agnostic to shapes.

Fit can be forced to use a specific linear solver by providing the keyword
``linear_solver``.

(Non-)Linear Programming
------------------------
`Linear programming
<https://en.wikipedia.org/wiki/Linear_programming>`_ problems are defined as

.. math ::

	\text{Maximize}  &c^T x \\
	\text{solve} 	 &A x \leq y \\
	\text{such that} &x \geq 0

After reading the section above, it should be clear that these are just a subset of the problems solvable with ``symfit``.