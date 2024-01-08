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
from symfit.symmip.backend import SCIPOptBackend, GurobiBackend

x, y, z = parameters('x, y, z', binary=True, min=0, max=1)

objective = x + y + 2 * z
constraints = [
    x + 2 * y + 3 * z <= 4,
    x + y >= 1
]

# We know solve this problem with different backends.
for backend in [SCIPOptBackend, GurobiBackend]:
    print(f'Run with {backend=}:')
    fit = MIP(objective, constraints=constraints, backend=backend)
    fit_result = fit.execute()

    print(f"Optimal objective value: {fit_result.objective_value}")
    print(
        f"Solution values: "
        f"x={fit_result[x]}, "
        f"y={fit_result[y]}, "
        f"z={fit_result[z]}"
    )
    print(fit_result, end='\n\n')
