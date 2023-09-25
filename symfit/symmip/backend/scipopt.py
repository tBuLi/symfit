from dataclasses import dataclass, field

import pyscipopt as scip

from sympy.printing.pycode import PythonCodePrinter


VTYPES = {'binary': 'B', 'integer': 'I', 'real': 'C'}


class SCIPOptPrinter(PythonCodePrinter):
    modules = scip


@dataclass
class SCIPOptBackend:
    model: scip.Model = field(default_factory=scip.Model)
    printer: PythonCodePrinter = field(default=SCIPOptPrinter)

    def add_var(self, *, vtype, **kwargs):
        return self.model.addVar(vtype=VTYPES[vtype], **kwargs)

    def add_constr(self, constrexpr):
        return self.model.addCons(constrexpr)

    @property
    def objective(self):
        return self.model.getObjective()

    @objective.setter
    def objective(self, obj):
        self.model.setObjective(obj)

    def update(self):
        return

    def optimize(self):
        self.model.optimize()

    @property
    def objective_value(self):
        return self.model.getObjVal()

    @property
    def solution(self):
        return {v.name: self.model.getVal(v) for v in self.model.getVars()}

    def get_value(self, v):
        return self.model.getVal(v)