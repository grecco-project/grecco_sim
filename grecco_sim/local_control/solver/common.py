import abc
from typing import Any, Callable
import numpy as np
from grecco_sim.local_control import mycas
from grecco_sim.local_control.physical import common as phys_commons, ocp_all_flex, ocp_ev
from grecco_sim.local_control.solver import common as solver_commons
from grecco_sim.util import sig_types, type_defs


def dict_to_model_conf(pars_dict: dict[str, type_defs.SysPars]) -> tuple:
    """Helper function to transform the sys pars dict to a model configuration tuple."""
    model_conf = tuple(pars.system for pars in pars_dict.values())
    return tuple(sorted(model_conf))


def get_par_values(
    model_configuration: tuple[str], state: dict, forecast: type_defs.Forecast
) -> dict:
    """Transform state, forecast to parameters for OCP."""
    par_values = {
        "fc_p_unc": forecast.fc_res_load,
    }

    par_values["c_sup"] = forecast.add_fc["c_sup"]
    par_values["c_feed"] = forecast.add_fc["c_feed"]

    # if "ev" in model_configuration:
    #     par_values["ev_soc_init"] = (state["ev_soc"],)

    if "bat" in model_configuration:
        par_values["x_bat_init"] = [state["soc"]]
        par_values["fc_pv_prod"] = -forecast.negative_part

    if "hp" in model_configuration:
        par_values["temp_init"] = state["temp"]
        par_values["temp_outside"] = forecast.add_fc["temp_outside"]
        par_values["solar_heat_gain"] = forecast.add_fc["solar_heat_gain"]

    return par_values


def get_par_values_arbitrary_naming(
    sys_pars: dict[str, type_defs.SysPars], state: dict, forecast: type_defs.Forecast
):
    """Transform state, forecast to parameters for OCP."""
    par_values = {
        "fc_p_unc": forecast.fc_res_load,
    }

    par_values["c_sup"] = forecast.add_fc["c_sup"]
    par_values["c_feed"] = forecast.add_fc["c_feed"]

    par_values["fc_pv_prod"] = -forecast.negative_part

    for flex_id, pars in sys_pars.items():
        match pars.system:
            # case "ev":
            # par_values["ev_soc_init"] = (state["ev_soc"],)

            case "bat":
                par_values[f"{flex_id}_soc_init"] = [state["soc"]]

            case "hp":
                par_values[f"{flex_id}_temp_init"] = state["temp"]
                par_values[f"{flex_id}_temp_outside"] = forecast.add_fc["temp_outside"]
                par_values[f"{flex_id}_solar_heat_gain"] = forecast.add_fc["solar_heat_gain"]

    return par_values


def get_controls(
    sys_id: str,
    horizon: int,
    model_configuration,
    map_type_to_id: dict[str, str],
    solver: mycas.MyNLPSolver,
    has_cp: bool,
) -> dict[str, np.ndarray]:
    """Perform mapping from optimal solution to control dict."""

    ret = {map_type_to_id["load"]: np.zeros(horizon)}

    if "pv" in model_configuration:
        ret[map_type_to_id["pv"]] = np.zeros(horizon)

    if "bat" in model_configuration:
        bat_control = solver.get_custom_function_value("u_bat")
        ret[map_type_to_id["bat"]] = bat_control

    if "hp" in model_configuration:
        hp_control = solver.opt_vector(f"p_el_hp_{sys_id}")
        ret[map_type_to_id["hp"]] = hp_control

    if "ev" in model_configuration:
        if has_cp:
            ev_control = solver.get_custom_function_value("u_ev")
        else:
            ev_control = np.zeros(horizon)
        ret[map_type_to_id["ev"]] = ev_control

    return ret


def get_controls_arbitrary_naming(
    sys_pars: dict[str, type_defs.SysPars],
    horizon: int,
    solver: mycas.MyNLPSolver,
    charging_active: set[str],
) -> dict[str, np.ndarray]:

    ret = {}

    for flex_id, pars in sys_pars.items():
        if pars.system in ["load", "pv"]:
            ret[flex_id] = np.zeros(horizon)
        elif pars.system in ["bat", "hp"]:
            ret[flex_id] = solver.get_custom_function_value(f"h{abs(hash(flex_id))}_control")
        elif pars.system == "ev":
            if flex_id in charging_active:
                ret[flex_id] = solver.get_custom_function_value(f"h{abs(hash(flex_id))}_control")
            else:
                ret[flex_id] = np.zeros(horizon)
        else:
            raise ValueError(f"No known way to get controls for system with type {pars.system}")

    return ret


class SolverWrapper(abc.ABC):
    """Interface for EV and PVBat solution.

    Implement this interface for a solver that implements caching of the OCPs.
    """

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: dict[str, type_defs.SysPars],
        controller_pars: type_defs.OptParameters,
    ):
        self.horizon = horizon
        self.sys_id = sys_id
        self.controller_pars = controller_pars

        # Save the model parameters and make mapping of subsystem type to id
        self.model_configuration = dict_to_model_conf(sys_parameters)

        assert len(set(self.model_configuration)) == len(
            self.model_configuration
        ), "This solver cannot handle systems with subsystems of identical type."

        self.map_type_to_id = {pars.system: sub_id for sub_id, pars in sys_parameters.items()}

    @abc.abstractmethod
    def solve(
        self,
        state: dict[str, Any],
        forecast: type_defs.Forecast,
        signal: sig_types.FirstOrderSignal,
        **kwargs,
    ):
        """Solve problem.

        Main challenge to solve in this function is to assemble needed parameters from input.
        """

    @abc.abstractmethod
    def get_u(self) -> dict[str, np.ndarray]:
        """Return schedule just for controlled flexibility"""

    @abc.abstractmethod
    def get_yg(self) -> np.ndarray:
        """Return schedule for schedule of grid power."""


class LocalSolverGradientDescentEV(solver_commons.SolverWrapper):
    """For EV the solver class is different as the problem is created in each solution instance."""

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: dict[str, type_defs.SysPars],
        controller_pars: type_defs.OptParameters,
        add_augmentation: Callable,
        add_par_value: Callable,
    ):
        super().__init__(horizon, sys_id, sys_parameters, controller_pars)

        self.add_augmentation = add_augmentation
        self.add_par_values = add_par_value

        self.ev_pars, self.ev_flex_id = [
            (pars, flex_id) for flex_id, pars in sys_parameters.items() if pars.system == "ev"
        ][0]
        self.sys_pars = sys_parameters
        self.solver = None

        self._has_cp = False

        if "bat" in self.model_configuration:
            self.bat_pars = [pars for pars in sys_parameters.values() if pars.system == "bat"][0]
            # get model config with "ev" for backup solver initialization
            mc = tuple(s for s in self.model_configuration if s != "ev")
            self.backup_solver_no_cp = self._create_problem(
                self.sys_id, self.horizon, mc, sys_parameters, controller_pars, {}
            )
        if "hp" in self.model_configuration:
            self.hp_pars = [pars for pars in sys_parameters.values() if pars.system == "hp"][0]
            # get model config with "ev" for backup solver initialization
            mc = tuple(s for s in self.model_configuration if s != "ev")
            self.backup_solver_no_cp = self._create_problem(
                self.sys_id, self.horizon, mc, sys_parameters, controller_pars, {}
            )

    def _create_problem(
        self,
        sys_id,
        horizon,
        model_configuration: tuple[str],
        sys_pars: dict[str, type_defs.SysPars],
        opt_pars: type_defs.OptParameters,
        charge_processes: dict[str, ocp_ev.ChargeProcess],
    ):
        ocp, grid_var = phys_commons.get_plain_ocp(
            sys_id, horizon, model_configuration, sys_pars, opt_pars, charge_processes
        )
        ocp = self.add_augmentation(sys_id, horizon, ocp, grid_var, self.controller_pars)
        return mycas.MyNLPSolver(ocp, opt_pars.solver_name)

    def solve(
        self,
        state: dict[str, Any],
        forecast: type_defs.Forecast,
        signal: sig_types.FirstOrderSignal,
        **kwargs,
    ):
        charge_process = phys_commons.get_charge_process(state, self.horizon, self.ev_pars)
        par_values = solver_commons.get_par_values(self.model_configuration, state, forecast)

        par_values = self.add_par_values(par_values, signal)

        if charge_process is None:
            self._has_cp = False
            if "hp" in self.model_configuration or "bat" in self.model_configuration:
                self.solver = self.backup_solver_no_cp
            else:
                self.solver = None
                self._yg = forecast.fc_res_load
                return
        else:
            self._has_cp = True
            self.solver = self._create_problem(
                self.sys_id,
                self.horizon,
                self.model_configuration,
                self.sys_pars,
                self.controller_pars,
                {self.ev_flex_id: charge_process},
            )

        self.solver.solve(par_values)

    def get_u(self) -> dict[str, np.ndarray]:
        """Return schedule just for controlled flexibility"""
        if self.solver is None:
            return {
                sub_sys_id: np.zeros(self.horizon) for sub_sys_id in self.map_type_to_id.values()
            }

        ret = solver_commons.get_controls(
            self.sys_id,
            self.horizon,
            self.model_configuration,
            self.map_type_to_id,
            self.solver,
            self._has_cp,
        )
        if set(ret.keys()) != set(self.map_type_to_id.values()) and self._has_cp:
            raise ValueError(f"No return implemented for system type {self.model_configuration}.")

        return ret

    def get_yg(self) -> np.ndarray:
        """Return schedule for schedule of grid power."""
        if self.solver is None:
            return self._yg
        else:
            return self.solver.opt_vector(f"y_g_a_{self.sys_id}")


class LocalSolverGradientDescentArbitrary(SolverWrapper):
    """This solver can be used for an arbitrary combination of flexibilities."""

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: dict[str, type_defs.SysPars],
        controller_pars: type_defs.OptParameters,
        add_augmentation: Callable,
        add_par_value: Callable,
    ):

        self.horizon = horizon
        self.sys_id = sys_id
        self.controller_pars = controller_pars

        self.add_augmentation = add_augmentation
        self.add_par_values = add_par_value

        self.sys_pars = sys_parameters
        self.solver = None

    def create_problem(
        self,
        sys_id,
        horizon,
        sys_pars: dict[str, type_defs.SysPars],
        opt_pars: type_defs.OptParameters,
        charge_processes: dict[str, ocp_ev.ChargeProcess],
    ) -> mycas.MyNLPSolver:
        """Create the OCP."""
        ocp, grid_var = ocp_all_flex.get_ocp(sys_id, horizon, sys_pars, opt_pars, charge_processes)
        if ocp.empty:
            return None

        self.add_augmentation(sys_id, horizon, ocp, grid_var, self.controller_pars)
        return mycas.MyNLPSolver(ocp, opt_pars.solver_name)

    def solve(
        self,
        state: dict[str, Any],
        forecast: type_defs.Forecast,
        signal: sig_types.FirstOrderSignal,
        **kwargs,
    ):

        charge_processes = {
            flex_id: phys_commons.get_charge_process(state, self.horizon, ev_pars)
            for flex_id, ev_pars in self.sys_pars.items()
            if ev_pars.system == "ev"
        }
        charge_processes = {key: val for key, val in charge_processes.items() if val is not None}
        self._charging_active = set(charge_processes.keys())

        self.solver = self.create_problem(
            self.sys_id,
            self.horizon,
            self.sys_pars,
            self.controller_pars,
            charge_processes,
        )
        if self.solver is None:
            self._yg = forecast.fc_res_load
            return

        par_values = solver_commons.get_par_values_arbitrary_naming(self.sys_pars, state, forecast)
        par_values = self.add_par_values(par_values, signal)

        self.solver.solve(par_values)

    def get_u(self) -> dict[str, np.ndarray]:
        """Return schedule just for controlled flexibility"""
        if self.solver is None:
            return {sub_sys_id: np.zeros(self.horizon) for sub_sys_id in self.sys_pars}

        ret = solver_commons.get_controls_arbitrary_naming(
            self.sys_pars, self.horizon, self.solver, self._charging_active
        )

        return ret

    def get_yg(self) -> np.ndarray:
        """Return schedule for schedule of grid power."""
        if self.solver is None:
            return self._yg
        else:
            return self.solver.opt_vector(f"{self.sys_id}_y_g")


class LocalSolverGradientDescent(solver_commons.SolverWrapper):
    """
    This solver extends a local_solver with the ADMM reference power term.

    This class should be generic to a range of different systems.
    """

    def __init__(
        self,
        horizon: int,
        sys_id: str,
        sys_parameters: dict[str, type_defs.SysPars],
        controller_pars: type_defs.OptParameters,
        add_augmentation: Callable,
        add_par_value: Callable,
    ):
        super().__init__(horizon, sys_id, sys_parameters, controller_pars)

        self.add_augmentation = add_augmentation
        self.add_par_values = add_par_value

        self.nlp_solver = self._create_problem(
            sys_id, horizon, self.model_configuration, sys_parameters, controller_pars, {}
        )

    def _create_problem(
        self,
        sys_id,
        horizon,
        model_configuration: tuple[str],
        sys_pars: dict[str, type_defs.SysPars],
        opt_pars: type_defs.OptParameters,
        charge_processes: dict[str, ocp_ev.ChargeProcess],
    ):
        ocp, grid_var = phys_commons.get_plain_ocp(
            sys_id, horizon, model_configuration, sys_pars, opt_pars, charge_processes
        )
        ocp = self.add_augmentation(sys_id, horizon, ocp, grid_var, self.controller_pars)
        return mycas.MyNLPSolver(ocp, opt_pars.solver_name)

    def solve(
        self,
        state: dict[str, Any],
        forecast: type_defs.Forecast,
        signal: sig_types.FirstOrderSignal,
        **kwargs,
    ):
        """
        Trigger solving the optimization problem for scheduling.

        The parameterization is depending on the controlled system's type.
        Maybe, in the future, this could be more generic with the OCP factory (local_solver_XXX)
        providing a method how to extract parameters from the state.
        """

        par_values = solver_commons.get_par_values(self.model_configuration, state, forecast)
        par_values = self.add_par_values(par_values, signal)

        self.nlp_solver.solve(par_values)
        # print(self.nlp_solver.sol)

    def get_u(self) -> dict[str, np.ndarray]:
        """Return schedule just for controlled flexibility"""
        ret = solver_commons.get_controls(
            self.sys_id,
            self.horizon,
            self.model_configuration,
            self.map_type_to_id,
            self.nlp_solver,
            False,
        )
        ret = {self.map_type_to_id["load"]: np.zeros(self.horizon)}

        if "pv" in self.model_configuration:
            ret[self.map_type_to_id["pv"]] = np.zeros(self.horizon)

        if "bat" in self.model_configuration:
            bat_control = self.nlp_solver.get_custom_function_value("u_bat")
            ret[self.map_type_to_id["bat"]] = bat_control

        if "hp" in self.model_configuration:
            hp_control = self.nlp_solver.opt_vector(f"p_el_hp_{self.sys_id}")
            ret[self.map_type_to_id["hp"]] = hp_control

        if set(ret.keys()) != set(self.map_type_to_id.values()):
            raise ValueError(f"No return implemented for system type {self.model_configuration}.")

        return ret

    def get_yg(self) -> np.ndarray:
        """Return schedule for schedule of grid power."""
        return self.nlp_solver.opt_vector(f"y_g_a_{self.sys_id}")


SOLVER_CLASSES = {
    ("hp", "load"): solver_commons.LocalSolverGradientDescent,
    ("hp", "load", "pv"): solver_commons.LocalSolverGradientDescent,
    ("bat", "load"): solver_commons.LocalSolverGradientDescent,
    ("bat", "load", "pv"): solver_commons.LocalSolverGradientDescent,
    ("ev", "load"): solver_commons.LocalSolverGradientDescentEV,
    ("ev", "load", "pv"): solver_commons.LocalSolverGradientDescentEV,
    ("bat", "hp", "load", "pv"): solver_commons.LocalSolverGradientDescent,
    ("bat", "ev", "load", "pv"): solver_commons.LocalSolverGradientDescentEV,
    ("ev", "hp", "load", "pv"): solver_commons.LocalSolverGradientDescentEV,
    # several EV
}
