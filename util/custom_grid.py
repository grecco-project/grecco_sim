import pandas as pd


def custom_transformer_types() -> pd.DataFrame:
    """ Define custom transformer types. """
    transformers = pd.DataFrame(
        index=["name", "s_nom", "v_nom_0", "v_nom_1", "vsc", "vscr", "pfe",
               "i0", "phase_shift", "tap_side", "tap_min", "tap_max", "tap_step",
               "references", "oltc", "capex", "opex", "code"])

    transformer_0 = {
        "name": 0,
        "s_nom": 0.4,
        "v_nom_0": 20.0,
        "v_nom_1": 0.4,
        "vsc": 3.9,
        "vscr": 1.2,
        "pfe": 0.0,
        "i0": 0.0,
        "phase_shift": 150.0,
        "tap_side": 1,
        "tap_min": -5,
        "tap_max": 5,
        "tap_step": 1.0,
        "references": "missing_type_auto_built",
        "oltc": False,
        "capex": None,
        "opex": None,
        "code": 6}

    transformer_1 = {
        "name": 0,
        "s_nom": 0.4,
        "v_nom_0": 20.0,
        "v_nom_1": 0.4,
        "vsc": 4.0,
        "vscr": 1.2,
        "pfe": 0.0,
        "i0": 0.0,
        "phase_shift": 150.0,
        "tap_side": 1,
        "tap_min": -5,
        "tap_max": 5,
        "tap_step": 1.0,
        "references": "missing_type_auto_built",
        "oltc": False,
        "capex": None,
        "opex": None,
        "code": 6}

    return pd.DataFrame([transformer_0 ,transformer_1])
