import numpy as np

from grecco_sim.util import type_defs


class ScheduleEvaluator(object):
    def __init__(self, y_ks, sim_pars: type_defs.SysParsPVBat):
        self.y_ks = y_ks
        self.pars = sim_pars

    def get_costs(self):
        costs_combined = 0
        for a in self.y_ks:
            yk = self.y_ks[a]
            c_sup = self.pars[a].c_sup
            c_sup = np.array(c_sup) if isinstance(c_sup, list) else np.ones(len(yk)) * c_sup
            costs_a = yk[np.where(yk > 0.)] * c_sup[np.where(yk > 0)]
            costs_a = costs_a.sum()

            costs_combined += costs_a

        return costs_combined

    def get_constr_viol(self, p_lim):

        grid_combined = np.sum(np.array([np.array(self.y_ks[a]) for a in self.y_ks]), axis=0)
        constr_violation = (grid_combined - p_lim)[np.where(grid_combined > p_lim)].sum()

        return constr_violation



