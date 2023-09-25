from dataclasses import dataclass, field
import importlib
import warnings

try:
    gp = importlib.import_module('gurobipy')
except ImportError as e:
    warnings.warn('Install gurobipy to use this feature.')
    gp = {}

from sympy.printing.pycode import PythonCodePrinter


VTYPES = {'binary': 'B', 'integer': 'I', 'real': 'C'}


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

    def add_var(self, *, vtype, **kwargs):
        return self.model.addVar(vtype=VTYPES[vtype], **kwargs)

    def add_constr(self, constrexpr):
        return self.model.addConstr(constrexpr)

    @property
    def objective(self):
        return self.model.getObjective()

    @objective.setter
    def objective(self, obj):
        self.model.setObjective(obj)

    def update(self):
        self.model.update()

    def optimize(self):
        self.model.optimize()

    @property
    def objective_value(self):
        return self.model.objVal

    @property
    def solution(self):
        return {v.VarName: v.X for v in self.model.getVars()}

    def get_value(self, v):
        return v.X