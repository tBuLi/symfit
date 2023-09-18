# Inspired by https://www.gurobi.com/documentation/9.5/examples/sudoku_py.html

import math

import numpy as np

from symfit import Parameter, symbols, IndexedBase, Idx, Sum, Eq
from symfit.symmip.backend import GurobiBackend
from symfit import MIP

with open('sudoku1') as f:
    grid = f.read().split()

for line in grid:
    print(line)

n = len(grid[0])
s = math.isqrt(n)

# Fix variables associated with cells whose values are pre-specified
# lb = np.array([[0 if char == "." else 1 for j, char in enumerate(line)] for i, line in enumerate(grid)])
lb = np.zeros((n, n, n), dtype=int)
for i in range(n):
    for j in range(n):
        if grid[i][j] != '.':
            v = int(grid[i][j]) - 1
            lb[i, j, v] = 1
ub = np.ones_like(lb)

# Prepare the boolean parameters for our sudoku board.
# Because every position on the board can have only one value,
# we make a binary Indexed symbol x[i,j,v], where i is the column,
# j is the row, and v is the value in the (i, j) position.
x = IndexedBase(Parameter('x', binary=True, min=lb, max=ub))
i, j, v = symbols('i, j, v', cls=Idx, range=n)
x_ijv = x[i, j, v]

# Add the sudoku constraints:
#   1. Each cell must take exactly one value: Sum(x[i,j,v], v) == 1
#   2. Each value is used exactly once per row: Sum(x[i,j,v], i) == 1
#   3. Each value is used exactly once per column: Sum(x[i,j,v], j) == 1
#   4. Each value is used exactly once per 3x3 subgrid.
constraints = [
    Eq(Sum(x[i, j, v], v), 1),
    Eq(Sum(x[i, j, v], i), 1),
    Eq(Sum(x[i, j, v], j), 1),
    *[Eq(Sum(x[i, j, v], (i, i_lb, i_lb + s - 1), (j, j_lb, j_lb + s - 1)), 1)
      for i_lb in range(0, n, s)
      for j_lb in range(0, n, s)]
]

fit = MIP(constraints=constraints, backend=GurobiBackend)
result = fit.execute()

print('')
print('Solution:')
print('')
solution = result[x]
for i in range(n):
    sol = ''
    for j in range(n):
        for v in range(n):
            if solution[i, j, v] > 0.5:
                sol += str(v+1)
    print(sol)
