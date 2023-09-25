"""
Inspired by https://www.gurobi.com/documentation/9.5/examples/multiscenario_py.html.

For now this symfit equivalent only solves a single scenario, as we do not (currently) support Gurobi's
multiscenario feature. However, this is still a nice example to demonstrate some of symfit's features.
"""

from symfit import MIP, IndexedBase, Eq, Idx, Parameter, symbols, Sum, pprint
from symfit.symmip.backend import SCIPOptBackend, GurobiBackend
import numpy as np

# Warehouse demand in thousands of units
data_demand = np.array([15, 18, 14, 20])

# Plant capacity in thousands of units
data_capacity = np.array([20, 22, 17, 19, 18])

# Fixed costs for each plant
data_fixed_costs = np.array([12000, 15000, 17000, 13000, 16000])

# Transportation costs per thousand units
data_trans_costs = np.array(
    [[4000, 2000, 3000, 2500, 4500],
     [2500, 2600, 3400, 3000, 4000],
     [1200, 1800, 2600, 4100, 3000],
     [2200, 2600, 3100, 3700, 3200]]
)

# Indices over the plants and warehouses
plant = Idx('plant', range=len(data_capacity))
warehouse = Idx('warehouse', range=len(data_demand))

# Indexed variables. Initial values become coefficients in the objective function.
open = IndexedBase(Parameter('Open', binary=True, value=data_fixed_costs))
transport = IndexedBase(Parameter('Transport', value=data_trans_costs))
fixed_costs = IndexedBase(Parameter('fixed_costs'))
trans_cost = IndexedBase(Parameter('trans_cost'))
capacity = IndexedBase(Parameter('capacity'))
demand = IndexedBase(Parameter('demand'))

objective = Sum(fixed_costs[plant] * open[plant], plant) + Sum(trans_cost[warehouse, plant] * transport[warehouse, plant], warehouse, plant)
constraints = [
    Sum(transport[warehouse, plant], warehouse) <= capacity[plant] * open[plant],
    Eq(Sum(transport[warehouse, plant], plant), demand[warehouse])
]

print('Objective:')
pprint(objective, wrap_line=False)
print('\nSubject to:')
for constraint in constraints:
    pprint(constraint, wrap_line=False)
print('\n\n')

data = {
    fixed_costs[plant]: data_fixed_costs,
    trans_cost[warehouse, plant]: data_trans_costs,
    capacity[plant]: data_capacity,
    demand[warehouse]: data_demand,
}

# We know solve this problem with different backends.
for backend in [SCIPOptBackend, GurobiBackend]:
    print(f'Run with {backend=}:')
    mip = MIP(objective, constraints=constraints, data=data, backend=backend)
    results = mip.execute()
    print(results)
