# Inspired by https://www.gurobi.com/documentation/9.5/examples/mip1_py.html
#
# Solve the following MIP:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

from symfit import parameters, MIP
from symfit.symmip.backend import GurobiBackend

x, y, z = parameters('x, y, z', binary=True)

objective = x + y + 2 * z
constraints = [
    x + 2 * y + 3 * z <= 4,
    x + y >= 1
]

fit = MIP(- objective, constraints=constraints, backend=GurobiBackend)
fit_result = fit.execute()

print(f"Optimal objective value: {fit_result.objective_value}")
print(
    f"Solution values: "
    f"x={fit_result.value(x)}, "
    f"y={fit_result.value(y)}, "
    f"z={fit_result.value(z)}"
)
print(fit_result)