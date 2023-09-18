from symfit import MIP, IndexedBase, Eq, Idx, Parameter, symbols, Sum
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
open = IndexedBase(Parameter('open', binary=True, value=data_fixed_costs))
transport = IndexedBase(Parameter('transport', value=data_trans_costs))
capacity = IndexedBase(Parameter('capacity'))
demand = IndexedBase(Parameter('demand'))

constraints = [
    Sum(transport[warehouse, plant], warehouse) <= capacity[plant] * open[plant],
    Eq(Sum(transport[warehouse, plant], plant), demand[warehouse])
]

data = {capacity[plant]: data_capacity, demand[warehouse]: data_demand}
mip = MIP(constraints=constraints, data=data)

results = mip.execute()
print(results)
print(results[open])
print(results[transport])