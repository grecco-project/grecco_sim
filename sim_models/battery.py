import numpy as np

from grecco_sim.util import type_defs
from grecco_sim.sim_models import model_base


class ConverterModelEff(object):
    """
    The gaol with this model is that the two provided functions are EXACTLY! inverse of each other.

    The equation is always:
    DC = AC - Loss (AC)

    Hence, Loss > 0 always. DC and AC > 0 is equivalent to charging the battery.

    """

    def __init__(self, eff):
        assert 0 < eff <= 1
        self.eff = eff

    def get_ac_power(self, p_dc):
        # No closed form to invert the absolute
        if p_dc < 0:
            return p_dc / (2 - self.eff)
        else:
            return p_dc / self.eff

    def get_dc_power(self, p_ac):
        return p_ac - abs(p_ac) * (1 - self.eff)


class Storage(model_base.Model):
    def __init__(
        self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysParsPVBat, *args
    ):
        """
        Necessary parameters in params dictionary

        +----------------+-------+----------+------+----------------------------------------+
        | Parameter name |  Type | Range    | Unit | Meaning                                |
        +================+=======+==========+======+========================================+
        | soc_init       | float | [0, 1]   |      | Initial storage SoC                    |
        +----------------+-------+----------+------+----------------------------------------+
        | eff            | float | [0, 1]   |      | Efficiency of charging and discharging |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_ac       | float | (0, inf) | kW   | Maximum AC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_dc       | float | (0, inf) | kW   | Maximum DC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | capacity       | float | [0, 1]   | kWh  | Capacity of the storage                |
        +----------------+-------+----------+------+----------------------------------------+


        :param sys_id: unique ID of the system
        :param horizon: horizon of simulation
        :param params: parameters to govern storage behavior in simulation
        """
        super().__init__(sys_id, horizon, dt_h)
        self.soc = np.zeros(self.horizon + 1)
        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_net = np.zeros(self.horizon)

        self._init_model(params)

    def _init_model(self, params):

        self.soc[0] = params.init_soc
        self.capacity = params.capacity

        assert params.p_inv > 0.0 and params.p_lim_dc > 0
        self.p_lim_ac = params.p_inv
        self.p_lim_dc = params.p_lim_dc

        self.converter = ConverterModelEff(params.eff)

    @property
    def system_type(self):
        return "bat"

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        """
        self.k += 1

        self._set_power(control)
        self._evolve()

    def _set_power(self, p_ac_set):

        self.p_ac_set[self.k - 1] = p_ac_set

        # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1.0 - self.soc[self.k - 1]) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = -self.soc[self.k - 1] * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power

        self.p_ac[self.k - 1] = p_ac_set
        self.p_net[self.k - 1] = self.converter.get_dc_power(p_ac_set)

    def _evolve(self):

        self.soc[self.k] = self.soc[self.k - 1] + self.p_net[self.k - 1] / self.capacity * self.dt_h

    def get_state(self):
        return {
            "soc": self.soc[self.k],
        }

    def get_output(self):
        return {"soc": self.soc, "p_ac": self.p_ac, "p_net": self.p_net}

    def get_grid_power_at(self, k: int):
        return self.p_ac[k]

    def get_grid_power(self):
        return self.p_ac


class StorageAging(model_base.Model):
    def __init__(self, sys_id: str, horizon: int, dt_h: float, params: type_defs.SysParsPVBatAging):
        """
        Necessary parameters in params dictionary

        +----------------+-------+----------+------+----------------------------------------+
        | Parameter name |  Type | Range    | Unit | Meaning                                |
        +================+=======+==========+======+========================================+
        | soc_init       | float | [0, 1]   |      | Initial storage SoC                    |
        +----------------+-------+----------+------+----------------------------------------+
        | eff            | float | [0, 1]   |      | Efficiency of charging and discharging |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_ac       | float | (0, inf) | kW   | Maximum AC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | p_lim_dc       | float | (0, inf) | kW   | Maximum DC power of converter          |
        +----------------+-------+----------+------+----------------------------------------+
        | capacity       | float | [0, 1]   | kWh  | Capacity of the storage                |
        +----------------+-------+----------+------+----------------------------------------+
        | soh_init       | float | [0, 1]   |      | Initial state of health                |
        +----------------+-------+----------+------+----------------------------------------+


        :param sys_id: unique ID of the system
        :param horizon: horizon of simulation
        :param params: parameters to govern storage behavior in simulation
        """
        super().__init__(sys_id, horizon, dt_h)

        self.soc = np.zeros(self.horizon + 1)

        self.soh_sep = np.zeros(self.horizon + 1)
        self.soh_tgth = np.zeros(self.horizon + 1)

        self.soh_cal = np.zeros(self.horizon + 1)
        self.soh_cyc = np.zeros(self.horizon + 1)

        self.aging_rated = np.zeros(self.horizon + 1)

        self.p_ac_set = np.zeros(self.horizon)
        self.p_ac = np.zeros(self.horizon)
        self.p_net = np.zeros(self.horizon)

        self._init_model(params)

    def _init_model(self, params):

        self.soc[0] = params.init_soc

        self.capacity_init = params.capacity

        self.soh_sep[0] = params.init_soh
        self.soh_tgth[0] = params.init_soh
        self.soh_cal[0] = params.init_soh + (1 - params.init_soh) / 2
        self.soh_cyc[0] = params.init_soh + (1 - params.init_soh) / 2

        self.capacity = self.capacity_init * self.soh_sep[0]

        self.s_therm = params.s_therm
        self.s_dod = params.s_dod

        self.a_soc = params.a_soc
        self.b_soc = params.b_soc

        self.a_char = params.a_char
        self.b_char = params.b_char

        assert params.p_inv > 0.0 and params.p_lim_dc > 0
        self.p_lim_ac = params.p_inv
        self.p_lim_dc = params.p_lim_dc

        self.converter = ConverterModelEff(params.eff)

    def apply_control(self, control):
        """
        Sign convention is always as a load perspective:
        positive value means charging the battery.

        :param control: intended AC battery power
        :return: no return
        """
        self.k += 1

        self._set_power(control)
        self._evolve()

    def _set_power(self, p_ac_set):

        self.p_ac_set[self.k - 1] = p_ac_set

        # Correct for AC limit on charge power
        p_ac_set = min(self.p_lim_ac, p_ac_set)
        p_ac_set = max(-self.p_lim_ac, p_ac_set)

        # Get power limits imposed from DC side
        max_dc = (1.0 - self.soc[self.k - 1]) * self.capacity / self.dt_h
        max_dc = min(max_dc, self.p_lim_dc)

        min_dc = -self.soc[self.k - 1] * self.capacity / self.dt_h
        min_dc = max(-self.p_lim_dc, min_dc)

        p_ac_set = min(p_ac_set, self.converter.get_ac_power(max_dc))
        p_ac_set = max(p_ac_set, self.converter.get_ac_power(min_dc))
        # This p_ac set is now a valid AC power

        self.p_ac[self.k - 1] = p_ac_set
        self.p_net[self.k - 1] = self.converter.get_dc_power(p_ac_set)

    def aging_rate_cal(self, soh_cal):

        s_soc = self.a_soc * (self.soc[self.k - 1] - 0.5) ** 3 + self.b_soc

        delta_soh_cal = soh_cal - 1.0

        dh_cal_dt = ((self.s_therm * s_soc) ** 2) / (2 * delta_soh_cal)

        return dh_cal_dt

    def aging_rate_cyc(self, soh_cyc):

        s_char = (self.a_char / (self.capacity)) * (abs(self.p_net[self.k - 1])) + self.b_char

        delta_soh_cyc = soh_cyc - 1.0

        dh_cyc_dt = (((self.s_dod * s_char * 1e-2) ** 2) / (4 * self.capacity * 3600)) * (
            (abs(self.p_net[self.k - 1])) / (delta_soh_cyc)
        )

        return dh_cyc_dt

    def _evolve(self):

        self.soc[self.k] = self.soc[self.k - 1] + self.p_net[self.k - 1] / self.capacity * self.dt_h

        self.aging_rated[self.k] = self.aging_rate_cal(
            soh_cal=self.soh_cal[self.k - 1]
        ) + self.aging_rate_cyc(soh_cyc=self.soh_cyc[self.k - 1])

        # k1 = self.aging_rate(soh=self.soh[self.k-1])
        # k2 = self.aging_rate(soh=(self.soh[self.k-1] + ((self.dt_h * 3600) / 2) * k1))
        # k3 = self.aging_rate(soh=(self.soh[self.k-1] + ((self.dt_h * 3600) / 2) * k2))
        # k4 = self.aging_rate(soh=(self.soh[self.k]+ (self.dt_h * 3600)* k3))

        # self.soh[self.k] = self.soh[self.k-1] + ((self.dt_h * 3600) / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.soh_cal[self.k] = (
            self.soh_cal[self.k - 1]
            + self.aging_rate_cal(soh_cal=self.soh_cal[self.k - 1]) * self.dt_h * 3600
        )
        self.soh_cyc[self.k] = (
            self.soh_cyc[self.k - 1]
            + self.aging_rate_cyc(soh_cyc=self.soh_cyc[self.k - 1]) * self.dt_h * 3600
        )
        ## self.soh[self.k] =  self.soh_cal[self.k] + self.soh_cyc[self.k] - 1

        self.soh_sep[self.k] = (
            self.soh_sep[self.k - 1]
            + (
                self.aging_rate_cal(soh_cal=self.soh_cal[self.k - 1])
                + self.aging_rate_cyc(soh_cyc=self.soh_cyc[self.k - 1])
            )
            * self.dt_h
            * 3600
        )

        h_cyc = 1 - ((1 - self.soh_tgth[self.k - 1]) * 0.125)
        h_cal = 1 - ((1 - self.soh_tgth[self.k - 1]) * 0.875)
        self.soh_tgth[self.k] = (
            self.soh_tgth[self.k - 1]
            + (self.aging_rate_cal(soh_cal=h_cal) + self.aging_rate_cyc(soh_cyc=h_cyc))
            * self.dt_h
            * 3600
        )

        self.capacity = self.capacity_init * self.soh_sep[self.k]

    def get_state(self):
        return {
            "soc": self.soc[self.k],
        }

    def get_output(self):
        return {
            "soc": self.soc,
            "soh_sep": self.soh_sep,
            "soh_tgth": self.soh_tgth,
            "soh_cal": self.soh_cal,
            "soh_cyc": self.soh_cyc,
            "aging_rate": self.aging_rated,
            "p_ac": self.p_ac,
            "p_net": self.p_net,
        }
