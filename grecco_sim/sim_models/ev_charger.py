import numpy as np
from grecco_sim.sim_models import model_base
from grecco_sim.sim_models import battery
import pandas as pd
import re


class ChargingProcess(object):
    def __init__(self, dt_h: float, params_cp: pd.DataFrame, params_charger: dict):
        self.params_cp = params_cp
        self.params_charger = params_charger
        self.dt_h = dt_h

        self._init_model(params_cp, params_charger)

    def _init_model(self, params_cp, params_charger):

        self.until_departure = params_cp["until_departure"]
        self.capacity = params_charger.capacity
        self.target_soc = params_cp["target_soc"]
        self.soc = params_cp["initial_soc"]

        assert params_charger.p_lim_ac > 0.0 and params_charger.p_lim_dc > 0
        self.p_lim_ac = params_charger.p_lim_ac
        self.p_lim_dc = params_charger.p_lim_dc

        self.converter = battery.ConverterModelEff(params_charger.eff)

    def apply_control(self, control):
        p_ac, p_net = self._set_power(control)
        self._evolve(p_net)
        return p_ac, p_net

    def _set_power_continuous(self, p_ac_set):
        """
        This is the function copied from the battery system with continuous control space
        :param p_ac_set:
        :return:
        """
        # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1.0 - self.soc) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = -self.soc * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power

        p_ac = p_ac_set
        p_net = self.converter.get_dc_power(p_ac_set)

        return p_ac, p_net

    def _set_power(self, p_ac_set):
        # only set power if mode is 1 (charge)
        if p_ac_set > 0:
            p_ac_set = self.p_lim_ac

            # Get power limits imposed from DC side
            max_dc = (1.0 - self.soc) * self.capacity / self.dt_h
            max_dc = min(max_dc, self.p_lim_dc)

            p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
            # This p_ac set is now a valid AC power

            p_ac = p_ac_set
            p_net = self.converter.get_dc_power(p_ac_set)
            return p_ac, p_net
        else:
            return 0.0, 0.0

    def _evolve(self, p_net):
        self.soc = self.soc + p_net / self.capacity * self.dt_h
        self.until_departure -= 1

    def get_remaining_time_h(self):
        return self.until_departure * self.dt_h


class EVCharger(model_base.Model):
    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: dict, ts_in):

        super().__init__(sys_id, horizon, dt_h)

        self.sys_id = sys_id
        self.params_charger = params

        self.soc = np.zeros(self.horizon + 1)
        self.soc.fill(np.NaN)
        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_net = np.zeros(self.horizon)

        self.active_cp = None

        self._init_cps(ts_in, sys_id)

    def _init_cps(self, ts_in, sys_id):
        self.idx_current_cp = 0  # pointer to the current charging process

        # time steps in which a charging process is active
        non_nans = ts_in[
            [
                self.sys_id + "_until_departure",
                self.sys_id + "_initial_soc",
                self.sys_id + "_target_soc",
            ]
        ].dropna()
        # time steps in which a charging process starts, if any
        # TODO schauen ob auch als np array geht
        self.cp_indices = []
        if len(non_nans) > 0:
            self.cp_indices.append(ts_in.index.get_loc(non_nans.index[0]))
            zero_times = non_nans.index[
                (non_nans[sys_id + "_until_departure"] == 0)
                & (non_nans.index != non_nans.index[-1])
            ]
            next_times = zero_times.map(
                lambda t: non_nans.index[non_nans.index.get_loc(t) + 1] if t in zero_times else None
            )
            next_indices = next_times.map(
                lambda t: ts_in.index.get_loc(t) if t is not None else None
            )
            # list of charging processes: index: time_step in which cp starts, columns: until_departure, initial_soc, target_soc
            self.cp_indices.extend(next_indices)
            self.cps = ts_in.iloc[self.cp_indices].copy()
            # self.cps["target_soc"] = 1
            self.cps.columns = self.cps.columns.str.replace(re.escape(sys_id) + "_", "", regex=True)
            # initialice charging process, if there is one at the beginning
            if (not self.cps.empty) & (self.cp_indices[0] == 0):
                self.active_cp = ChargingProcess(self.dt_h, self.cps.iloc[0], self.params_charger)
                self.soc[0] = self.active_cp.soc
        if not hasattr(self, "cps"):
            print("No charging processes for this EV")

    @property
    def system_type(self):
        return "ev"

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        check for availability of EV (active_cp = true)
        if available, apply control, deactive cp if EV is unavailable in the next time step
        if unavailable, check if there is a cp starting in the next time step
        """
        if self.active_cp is not None:
            p_ac, p_net = self.active_cp.apply_control(control)
            soc = self.active_cp.soc

            if self.active_cp.until_departure == 0:
                self.active_cp = None
                self.idx_current_cp += 1
        else:
            p_ac, p_net = 0.0, np.NaN
            soc = np.NaN
            # soc = 0

            if (
                not self.idx_current_cp == len(self.cp_indices)
                and self.cp_indices[self.idx_current_cp] == self.k + 1
            ):
                self.active_cp = ChargingProcess(
                    self.dt_h, self.cps.iloc[self.idx_current_cp], self.params_charger
                )
                soc = self.active_cp.soc

        self.p_ac_set[self.k] = control
        self.p_ac[self.k] = p_ac
        self.p_net[self.k] = p_net

        self.k += 1

        self.soc[self.k] = soc

    def get_state(self):
        state = {"ev_soc": self.soc[self.k], "ev_connected": self.active_cp is not None}
        if self.active_cp is not None:
            state.update(
                {
                    "ev_target_soc": self.active_cp.target_soc,
                    "ev_capacity": self.active_cp.capacity,
                    "ev_remaining_time_h": self.active_cp.get_remaining_time_h(),
                }
            )
        return state

    def get_output(self):
        return {"soc": self.soc, "p_ac": self.p_ac, "p_net": self.p_net}

    def get_grid_power_at(self, k: int):
        return self.p_ac[k]

    def get_grid_power(self):
        return self.p_ac
