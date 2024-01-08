# Inspired by https://www.gurobi.com/documentation/9.5/examples/bilinear_py.html#subsubsection:bilinear.py
#
# This example formulates and solves the following simple bilinear model:
#  maximize    x
#  subject to  x + y + z <= 10
#              x * y <= 2         (bilinear inequality)
#              x * z + y * z = 1  (bilinear equality)
#              x, y, z non-negative (x integral in second version)

from symfit import parameters, Eq
from symfit import MIP

# Create variables
x, y, z = parameters('x, y, z', min=0)

objective = 1.0 * x
constraints = [
    x + y + z <= 10,
    x*y <= 2,
    Eq(x*z + y*z, 1),
]

mip = MIP(- objective, constraints=constraints)
mip_result = mip.execute()

print(mip_result)