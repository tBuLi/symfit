from dataclasses import dataclass, field

import gurobipy as gp

from sympy.printing.pycode import PythonCodePrinter
from symfit.core.argument import Parameter


VTYPES = {'binary': 'B', 'integer': 'I', 'real': 'C',}


class GurobiPrinter(PythonCodePrinter):
    modules = gp

    def _print_Sum(self, expr):
        loops = (
            'for {i} in range({a}, {b}+1)'.format(
                i=self._print(i),
                a=self._print(a),
                b=self._print(b))
            for i, a, b in expr.limits)
        return '(gurobipy.quicksum({function} {loops}))'.format(
            function=self._print(expr.function),
            loops=' '.join(loops))


@dataclass
class GurobiBackend:
    model: gp.Model = field(default_factory=gp.Model)
    printer: PythonCodePrinter = field(default=GurobiPrinter)

    def add_var(self, **kwargs):
        return self.model.addVar(**kwargs)

    def add_constr(self, constrexpr):
        return self.model.addConstr(constrexpr)

    def set_objective(self, obj):
        self.model.setObjective(obj)

    @staticmethod
    def param_vtype(p: Parameter):
        """ Return the gurobi vtype for the parameter `p`. """
        for key, vtype in VTYPES.items():
            if p.assumptions0.get(key, False):
                return vtype

    def update(self):
        self.model.update()

    def optimize(self):
        self.model.optimize()

    @property
    def objective_value(self):
        return self.model.objVal
