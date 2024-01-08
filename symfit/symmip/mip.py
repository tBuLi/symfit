from dataclasses import dataclass, field
from typing import List, Dict, Protocol
from functools import cached_property, reduce
from collections import defaultdict, namedtuple
import itertools

from sympy import Expr, Rel, Eq, Idx, Indexed, Symbol, lambdify
from sympy.printing.pycode import PythonCodePrinter
import numpy as np

from symfit.symmip.backend import SCIPOptBackend
from symfit.core.support import key2str


VTYPES = ['binary', 'integer', 'real'] # ToDo: Enum?


class MIPBackend(Protocol):
    @property
    def model(self):
        pass

    @property
    def printer(self) -> PythonCodePrinter:
        pass

    def add_var(self, *args, **kwargs):
        pass

    def add_constr(self, *args, **kwargs):
        pass

    @property
    def objective(self):
        pass

    @objective.setter
    def objective(self, obj):
        pass

    def update(self, *args, **kwargs):
        pass

    def optimize(self, *args, **kwargs):
        pass

    @property
    def objective_value(self):
        pass

    @property
    def solution(self) -> dict:
        pass

    def get_value(self, v):
        pass


@dataclass
class MIPResult:
    best_vals: Dict
    objective_value: float

    @property
    def base2indexed(self):
        res = {v.base.label: v for v in self.best_vals if isinstance(v, Indexed)}
        res.update({v.base: v for v in self.best_vals if isinstance(v, Indexed)})
        return res

    def __getitem__(self, v):
        if not isinstance(v, Indexed) and v in self.base2indexed:
            v = self.base2indexed[v]
        return self.best_vals[v]



@dataclass
class MIP:
    objective: Expr = field(default=None)
    constraints: List[Rel] = field(default_factory=list)
    backend: MIPBackend = field(default=SCIPOptBackend)
    data: Dict = field(default_factory=dict)

    indices: List[Idx] = field(init=False)
    variables: List[Symbol] = field(init=False)
    indexed_variables: List[Indexed] = field(init=False)

    def __post_init__(self):
        # Prepare constraints
        self.constraints = [c if isinstance(c, Rel) else Eq(c, 0)
                            for c in self.constraints]
        free_symbols = reduce(set.union,
                              (c.free_symbols for c in self.constraints),
                              self.objective.free_symbols if self.objective else set())

        self.indices = {s for s in free_symbols if isinstance(s, Idx)}
        self.indexed_variables = {s for s in free_symbols
                                  if isinstance(s, Indexed)
                                  if all(isinstance(idx, Idx) for idx in s.indices)}
        labels = {s.base.label for s in self.indexed_variables}
        self.variables = {s for s in free_symbols if not isinstance(s, (Idx, Indexed)) if s not in labels}
        self.backend = self.backend()
        self._make_mip_model()

    @property
    def dependent_vars(self):
        return (self.variables | self.indexed_variables) - self.independent_vars

    @property
    def independent_vars(self):
        return set(self.data)

    @staticmethod
    def param_vtype(p: "Parameter"):
        """ Return the gurobi vtype for the parameter `p`. """
        for vtype in VTYPES:
            if p.assumptions0.get(vtype, False):
                return vtype
        return 'real'

    def _make_mip_model(self):
        # Dictionary mapping symbolic entities to the corresponding MIP variable
        self.mip_vars = defaultdict(dict)

        # If data was provided, then use its value instead of making it a Variable.
        for p, data in self.data.items():
            if isinstance(p, Indexed):
                self.mip_vars[p.base.label] = data
            else:
                self.mip_vars[p] = data

        for p in self.dependent_vars:
            if not isinstance(p, Indexed):
                kwargs = {'name': p.name}
                if p.max is not None:
                    kwargs['ub'] = p.max
                if p.min is not None:
                    kwargs['lb'] = p.min
                kwargs['vtype'] = self.param_vtype(p)
                self.mip_vars[p] = self.backend.add_var(**kwargs)
                continue

            param = p.base.label
            for indices in itertools.product(*(range(idx.lower, idx.upper + 1) for idx in p.indices)):
                kwargs = {'name': f"{param.name}_{'_'.join(str(i) for i in indices)}"}
                if param.max is not None:
                    kwargs['ub'] = param.max[indices]
                if param.min is not None:
                    kwargs['lb'] = param.min[indices]
                if param.value is not None:
                    try:
                        kwargs['obj'] = param.value[indices]
                    except TypeError:
                        pass
                kwargs['vtype'] = self.param_vtype(param)
                self.mip_vars[param][indices if len(indices) > 1 else indices[0]] = self.backend.add_var(**kwargs)

        all_vars = key2str(self.mip_vars)
        # Translate the objective and constraints from sympy to their equivalent for the MIP solver.
        if self.objective:
            obj_func = lambdify(self.mip_vars.keys(), self.objective, printer=self.backend.printer, modules=self.backend.printer.modules)
            obj = obj_func(**all_vars)
            self.backend.objective = obj

        # For constraints the idea is the same as above, but a bit more involved since
        # constraints can have free indices.
        if self.constraints:
            free_indices_for_constraints = defaultdict(list)
            for constraint in self.constraints:
                # Find the free indices in this constraint, and store them in alphabetical order.
                free_indices = tuple(sorted((constraint.free_symbols & self.indices), key=lambda s: s.name))
                free_indices_for_constraints[free_indices].append(constraint)

            # TODO: Group the free indices in such a way that we can iterate over them even more efficiently.
            for free_indices, constraints in free_indices_for_constraints.items():
                # Create the callable for these constraints to translate to MIP solver.
                # Important: the args to these callables are free indices first, then the rest!
                args = free_indices + tuple(self.mip_vars.keys())
                args = tuple(s.name for s in args)  # To string, because sympy is a Dummy.
                constrs_func = lambdify(
                    args, constraints, printer=self.backend.printer, modules=self.backend.printer.modules
                )

                # For all free indices, call the created function to make the MIP equivalent statement.
                for idxs in itertools.product(*[range(i.lower, i.upper + 1) for i in free_indices]):
                    constrs = constrs_func(*idxs, **all_vars)
                    for constr in constrs:
                        self.backend.add_constr(constr)

        self.backend.update()

    def execute(self, *args, **kwargs) -> MIPResult:
        self.backend.optimize(*args, **kwargs)

        # Extract the optimized values corresponding to the dependent variables.
        dependent_vars = sorted(self.dependent_vars, key=lambda x: x.name)
        best_vals = {}
        for v in dependent_vars:
            if isinstance(v, Indexed):
                vtype = self.param_vtype(v.base.label)
                if vtype == 'binary':
                    vals = np.zeros(v.shape, dtype=bool)
                elif vtype == 'real':
                    vals = np.zeros(v.shape, dtype=float)
                elif vtype == 'integer':
                    vals = np.zeros(v.shape, dtype=int)
                for idxs in itertools.product(*[range(i.lower, i.upper + 1) for i in v.indices]):
                    vals[idxs] = self.backend.get_value(self.mip_vars[v.base.label][idxs if len(idxs) > 1 else idxs[0]])

                best_vals[v] = vals
            else:
                best_vals[v] = self.backend.get_value(self.mip_vars[v])

        return MIPResult(objective_value=self.backend.objective_value, best_vals=best_vals)
