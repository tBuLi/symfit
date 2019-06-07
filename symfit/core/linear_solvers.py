"""
Linear solvers solve linear problems.

Their API is a hybrid of our objectives and minimizers. This is because
they take in a model and some data, and output not only the tensor valued
parameters, but also a small fit report.
"""


import abc
from collections import OrderedDict
from six import add_metaclass

import numpy as np
from scipy.optimize import lsq_linear
from sympy import MatMul

from symfit.core.models import variabletuple, ModelError
from symfit.core.support import key2str, sympy_to_py, cached_property
from symfit.core.fit_results import FitResults

@add_metaclass(abc.ABCMeta)
class BaseLinearSolver(object):
    """
    ABC for Linear solvers.
    """
    def __init__(self, model, data):
        """
        :param model: `symfit` style model.
        :param data: data for all the variables of the model.
        """
        self.model = model
        self.data = data
        self.Results = variabletuple('Results', self.model.params)

    @cached_property
    def subproblems(self):
        problems = {}
        for y in self.model.dependent_vars:
            for p in self.model.params:
                expr = self.model[y]
                if expr.has(p):
                    if (not isinstance(expr, MatMul) or
                            not (expr.args[0] is p or expr.args[-1] is p)):
                        raise ModelError('Not a linear subproblem')
                    else:
                        expr = expr.xreplace({p: 1}).simplify()
                        expr_func = sympy_to_py(expr, expr.free_symbols)
                        relevant_data = {var: data for var, data in self.data.items()
                                         if var in expr.free_symbols}
                        expr_data = expr_func(**key2str(relevant_data))
                        problems[p] = (expr_data, self.data[y])
        return problems

    @abc.abstractmethod
    def execute(self, *args, **kwargs):
        # TODO: come up with a consistent output format.
        pass

class LstSq(BaseLinearSolver):
    def execute(self, *args, **kwargs):
        tensor_params = {}
        ans = {}
        for x, (A, y) in self.subproblems.items():
            res = np.linalg.lstsq(A, y, *args, **kwargs, rcond=None)
            tensor_params[x] = res[0]
            ans[x.name] = res
        res = FitResults(self.model, popt=[], covariance_matrix=None,
                         minimizer=self, objective=None, message='',
                         tensor_params=tensor_params, **ans)
        return res

class LstSqBounds(BaseLinearSolver):
    def execute(self, *args, **kwargs):
        tensor_params = {}
        ans = {}
        bounds = self.model.bounds
        for (x, (A, y)), (lb, ub) in zip(self.subproblems.items(), bounds):
            M = A.T.dot(A)
            DB = A.T.dot(y)
            x_res = []
            for lb_i, ub_i, DB_i in zip(lb.T, ub.T, DB.T):
                res = lsq_linear(M, DB_i, bounds=(lb_i, ub_i))
                x_res.append(res['x'][:, None])
            tensor_params[x] = np.block([[x_i for x_i in x_res]])
            ans[x.name] = res

        res = FitResults(self.model, popt=[], covariance_matrix=None,
                         minimizer=self, objective=None, message='',
                         tensor_params=tensor_params, **ans)
        return res