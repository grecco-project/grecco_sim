import pickle
from typing import Any, Dict
import pickle
from typing import Any, Dict
from pathlib import Path
import pytz
import json
import math

import numpy as np
import pandas as pd

from grecco_sim.util import config, type_defs
from grecco_sim.util import config, type_defs


UNIX_TIME = "unixtimestamp"


def write_ts(ts: pd.DataFrame, output_filename: Path):
    """Write a time series using unixtime to decode time info."""

    assert isinstance(ts.index, pd.DatetimeIndex), "Only usable for time series data"

    ts.loc[:, UNIX_TIME] = ts.index.astype(np.int64).values / 1e9

    # Sanity check
    unixtime = ts.loc[:, UNIX_TIME].values
    if (unixtime[1] - unixtime[0]) * (len(unixtime) - 1) != unixtime[-1] - unixtime[0]:
        print(f"{unixtime[-1]}, {unixtime[0]}, {unixtime[1]}, {len(unixtime)}")
        raise ValueError("Data has non-constant time step. Result is incorrect")

    ts.to_csv(output_filename, sep=";", index_label="timestamp", float_format="%f")

    ts.drop(columns=UNIX_TIME, inplace=True)


def read_ts(
    input_filename: Path,
    ambiguous: str = "raise",
    tz_info=pytz.timezone("Europe/Berlin"),
) -> pd.DataFrame:
    """Read a time series file in the format used in GreCCo sim output."""

    data_raw = pd.read_csv(input_filename, sep=";", index_col="timestamp")

    ind_start = pd.to_datetime(
        data_raw.loc[data_raw.index[0:2], UNIX_TIME].apply(pd.to_numeric).values, unit="s"
    )
    data_raw.index = pd.date_range(
        start=ind_start[0],
        freq=f"{int((ind_start[1] - ind_start[0]).total_seconds())}s",
        periods=len(data_raw.index),
    )

    data_tz = data_raw.tz_localize("UTC", ambiguous=ambiguous).tz_convert(tz_info)
    data_tz = data_tz.drop(columns=UNIX_TIME)

    return data_tz


def get_charging_data(ts_in, dt_h):
    # get parking duration ans soc data while ev is available for charging
    ts_out = pd.DataFrame(
        index=ts_in.index, columns=["until_departure", "initial_soc", "target_soc"]
    )

    # Filter non-zero values
    ts_ev_connected = ts_in["cp"][ts_in["cp"] != 0]

    # Create a grouping that increments if the time difference is not 15 minutes
    group = (ts_ev_connected.index.to_series().diff() != pd.Timedelta("15min")).cumsum()

    # Combine the series and group into a DataFrame
    df_temp = pd.concat([ts_ev_connected, group], axis=1)
    df_temp.columns = ["cp", "group"]

    df_temp.columns = ["cp", "group"]

    # Group by the block (using group and the value) and record start and end times
    availability = (
        df_temp.groupby(["group"])
        .apply(lambda x: pd.Series({"start_time": x.index[0], "end_time": x.index[-1]}))
        .reset_index(drop=True)
    )

    # Get time steps until departure (and interpolate soc)
    for idx, row in availability.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        init_soc = ts_in.loc[start_time, "soc_min_percent"] / 100
        target_soc = ts_in.loc[end_time, "soc_max_percent"] / 100
        if math.isnan(target_soc):
            target_soc = ts_in.loc[
                end_time - pd.Timedelta("15min"), "soc_max_percent"
            ]  # this seems to be a bug in the input data
        if target_soc < init_soc:
            target_soc = init_soc
        until_departure = (end_time - start_time) / dt_h
        ts_out.loc[start_time:end_time, "until_departure"] = np.arange(
            until_departure, until_departure - len(ts_in.loc[start_time:end_time]), -1
        )
        ts_out.loc[start_time:end_time, "initial_soc"] = init_soc
        ts_out.loc[start_time:end_time, "target_soc"] = target_soc

    return ts_out


def get_new_charging_data(ts_in, dt_h):
    """Constructs the charging processes data from the input time series.

    Input: ts_in is a DataFrame, snapshot index and soc values for a single EV.

    Output: ts_out is a DataFrame with the same index, containing the time until departure,
    initial state of charge (soc) and target soc."""
    ts_out = pd.DataFrame(
        index=ts_in.index, columns=["until_departure", "initial_soc", "target_soc"]
    )

    ts_ev_connected = ts_in["cp"][ts_in["cp"] != 0]

    # Create a grouping that increments if the time difference is not 15 minutes
    group = (ts_ev_connected.index.to_series().diff() != pd.Timedelta("15min")).cumsum()

    # Combine the series and group into a DataFrame
    df_temp = pd.concat([ts_ev_connected, group], axis=1)
    df_temp.columns = ["cp", "group"]

    # Group by the block (using group and the value) and record start and end times
    availability = (
        df_temp.groupby("group", group_keys=False)
        .apply(
            lambda x: pd.Series(
                {
                    "group": x["group"].iloc[0],
                    "start_time": x.index[0],
                    "end_time": x.index[-1],
                }
            ),
            include_groups=True,
        )
        .reset_index(drop=True)
    )

    ts_out = pd.DataFrame(
        index=ts_in.index, columns=["until_departure", "initial_soc", "target_soc"]
    )

    # Get time steps until departure (and interpolate soc)
    for idx, row in availability.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        until_departure = (end_time - start_time) / dt_h
        ts_out.loc[start_time:end_time, "until_departure"] = np.arange(
            until_departure, until_departure - len(ts_in.loc[start_time:end_time]), -1
        )
        ts_out.loc[start_time:end_time, "initial_soc"] = ts_in.loc[start_time, "soc"] / 100
        ts_out.loc[start_time:end_time, "target_soc"] = ts_in.loc[end_time, "soc"] / 100

    return ts_out


def pickle_something(filename: str, something: Any):
    """Write something as pickle to disc for inspection."""
    with open(config.data_path() / "pickled" / filename, "wb") as out_file:
        pickle.dump(something, out_file)


def unpickle(filename: str) -> Any:
    """Load pickled object."""
    with open(config.data_path() / "pickled" / filename, "rb") as out_file:
        return pickle.load(out_file)


def dump_parameterization(out_file_name: Path, parameters):
    """Write a json file of parameter classes"""
    with open(out_file_name, "w") as out_file:
        json.dump(parameters, out_file, cls=type_defs.EnhancedJSONEncoder)


def load_sizing(in_file_name: Path) -> dict[str, dict[str, type_defs.SysPars]]:
    """Read sizing json from file and create SysPars objects from it."""
    with open(in_file_name) as sizing_file:
        raw_sizing = json.load(sizing_file)

    SIZING_CLASS = {
        "hp": type_defs.SysParsHeatPump,
        "load": type_defs.SysParsLoad,
        "bat": type_defs.SysParsPVBat,
        "pv": type_defs.SysParsPV,
        "ev": type_defs.SysParsEV,
    }
    return {
        sys_id: {
            subsys: SIZING_CLASS[raw_dict[subsys]["_system"]](**raw_dict[subsys])
            for subsys in raw_dict
        }
        for sys_id, raw_dict in raw_sizing.items()
    }


def set_tz_index_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """df.tz_localize raises an Error on already localized data. This method
    implements it in a more robust fashion.

    Args:
        df: input dataframe.

    Returns:
        pd.DataFrame: The input dataframe with utc_localized index.
    """

    localized_df = df.copy()

    tz_index = pd.DatetimeIndex(df.index)

    if tz_index.tz is None:
        tz_index = tz_index.tz_localize("utc")
    else:
        tz_index = tz_index.tz_convert("Europe/Berlin")

    localized_df.index = tz_index

    return localized_df


def convert_to_unix(idx: pd.Index, assume_tz="UTC"):
    # Build naive/aware DatetimeIndex

    if type(idx) is not pd.DatetimeIndex:
        idx = pd.to_datetime(idx, utc=True)

    if idx.tz is None:
        # Interpret naive times as being in `assume_tz`
        # If you use a DST zone like Europe/Berlin, handle DST edge cases:
        idx = idx.tz_localize(
            assume_tz,
            ambiguous="infer",  # fall-back hour (e.g., Oct) -> infer
            nonexistent="shift_forward",  # spring-forward gap -> shift to next valid time
        )
    else:
        # Already tz-aware: good
        pass

    # Unix seconds (int64). For ms, divide by 1_000_000 instead.
    return pd.Index(idx.view("int64") // 1_000_000_000, name="unixtime", dtype="int64")


def get_csv_delimiter(file_path):
    """
    Identifies delimiter in a csv file for handling different formats.
    """
    with open(file_path, "r") as file:
        line = file.readline()
        if ";" in line:
            return ";"
        elif "," in line:
            return ","
        elif " " in line:
            return " "
        else:
            raise ValueError("Unknown delimiter in file: " + file_path)
