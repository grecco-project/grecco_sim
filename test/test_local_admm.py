from grecco_sim.local_control.solver import common as solver_common
from grecco_sim.local_control.solver import local_admm
from grecco_sim.util import type_defs


def test_local_admm() -> bool:
    """Test if the local ADMM controller an solver tools work

    :return: True if the test passes
    :rtype: bool
    """

    sys_pars = {
        "bat": type_defs.SysParsPVBat("test_system", 0.25, 0.3, 0.1, 0.5, 5.0, 5.0),
        "load": type_defs.SysParsLoad("test_load", 0.25, 0.3, 0.1),
        "pv": type_defs.SysParsPV("test_pv", 0.25, 0.3, 0.1),
    }
    opt_pars = type_defs.OptParameters(solver_name="osqp", rho=2.0, mu=10.0, horizon=10, alpha=1.0)

    local_admm.get_solver(
        horizon=10,
        signal_lengths=[1],
        sys_id="ag_test",
        sys_parameters=sys_pars,
        controller_pars=opt_pars,
    )
    return True


if __name__ == "__main__":
    test_local_admm()
