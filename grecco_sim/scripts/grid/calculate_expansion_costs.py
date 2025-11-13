import logging
import sys
import os
import json
import pandas as pd
import base64
import requests
from io import StringIO

logger = logging.getLogger(__name__)


#  FUNCTIONS FOR INDIGO OPTIMIZATION

def indigo_optimize_grid(username=None, token=None, grid_name=None, grid_to_send=None, config=None,
                         host="https://indigo.ise.fhg.de/", namespace="optimize_inatech",
                         populations=5):
    """
    optimize given grid as parametrized with config

    Parameters
    ----------
    username : str
        username to authenticate
    token : str
        token to authenticate
    grid_name : str
        name of grid
    grid_to_send : dict
        components of pypsa_extended.pypsa_extended.GridExt as json
    config : json
        Configfile for the optimization of the given grid
    host : str
        https://indigo.ise.fhg.de/
    namespace : str
        optimize_inatech

    Returns
    -------
    response

    """
    url = host + namespace
    logger.info(f"Sending grid to indigo api at {url}")

    data = {
        "username": username, "token": token, "grid_to_send": grid_to_send,
        "grid_name": grid_name, "config": config, "populations": populations
    }

    r = requests.post(url, json=data)

    return r


def collect_input_data(grid_folder, config_path):
    """collect input data

    Parameters
    ----------
    grid_folder : str
        path to grid folder
    example_folder : str
        path to example folder

    Returns
    -------
    input data

    """
    # get grid components as json to send to api
    grid_to_send = {}
    for file in os.listdir(grid_folder):
        if not file.endswith(".csv"):
            continue
        filename = os.fsdecode(file)
        df = pd.read_csv(os.path.join(grid_folder, filename), index_col=0)
        grid_to_send[filename] = df.to_json()

    # read config file
    with open(config_path, 'r') as file:
        config = json.load(file)

    return grid_to_send, config

# FUNCTIONS FOR RESULTS REQUEST
def get_state_by_sid(username: str=None, sid: int=None, token: str=None,
                     host: str="https://indigo.ise.fhg.de/",
                     namespace: str="get-optimized-by-sid"):
    """Send a request to API to get progress and result of gen-alg optimization. namespace need to be specified,
    e.g. namespace="get-optimized-by-sid".

    Print response from API: HTTP_Status_Code and message.

    Parameter
    --------
    username : str
        username to authenticate.
    sid : int
        Identifier created when a new computation is started.
        With sid progress and results can be requested.
    token : str
        Token to authenticate.
    host : str
        https://indigo.ise.fhg.de/
    namespace : str
        get-optimized-by-sid

    Returns
    -------
    response : dict
        response containing HTTP_status_code, sid

    """
    # specify data to send
    data = {"username": username, "sid": sid, "token": token, "inatech": True}
    r = requests.post(host + namespace, json=data)  # send request to API

    return r


def _write_csv_to_disc(fname, fdata, result_grid_name, result_grid_export_path):
    df = pd.read_json(StringIO(fdata), orient="split")
    if "network" in fname:
        df.index.name = "name"
        df.index = [result_grid_name]
        df["name"] = result_grid_name
    elif "snapshots" in fname:
        if df.empty:
            # initialize list elements
            data = [0]  # contains snapshot
            # Create the pandas DataFrame with column name is provided explicitly
            df = pd.DataFrame(data, columns=['name'])
    df.to_csv(os.path.join(result_grid_export_path, fname))


def save_grid_result(res, result_grid_name, result_grid_export_path):
    """
    Save the all files in res to the given folder result_grid_export_path
    under the given name result_grid_name

    Parameters
    ----------
    res : dict
        Result of optimization
    result_grid_name : str
        name of folder to be created with resulting grid
    result_grid_export_path : str
        directory to save the grid folder to

    Returns
    -------
    None.

    """
    if not os.path.exists(result_grid_export_path):
        os.makedirs(result_grid_export_path)
    if isinstance(res, str):
        res = json.loads(res)
    for fname, fdata in res.items():
        if "subdir_" in fname:
            fname = fname.split("subdir_")[-1]
            save_grid_result(fdata, result_grid_name, os.path.join(result_grid_export_path, fname))
        elif ".csv" in fname:
            _write_csv_to_disc(fname, fdata, result_grid_name, result_grid_export_path)
        elif "meta" in fname:
            with open(os.path.join(result_grid_export_path, fname), 'w') as f:
                json.dump(json.loads(base64.b64decode(fdata)), f)
        else:
            with open(os.path.join(result_grid_export_path, fname), 'wb') as f:
                f.write(base64.b64decode(fdata))


# --- Configuration ---
# path to config file (will be set by command line)
folder_with_config = ""  # Must be a json file

# grid object generated from coordinated profiles (will be set by command line)
grid_folder = ""  # Enter path to folder with grid object

# Credentials for authentication  -  NEVER PUSH THIS
username = "inatech_client"
token = ""  # Will be set by command line

# --- Argument Checking ---
if __name__ == "__main__":

    # Check if we got exactly 4 arguments (script name + 3 inputs)
    if len(sys.argv) != 4:
        print("ERROR: Missing arguments.")
        print(
            f"Usage: python {sys.argv[0]} <token> <grid_folder_path> <config_file_path>")
        sys.exit(1)

    # Set variables from command line arguments
    token = sys.argv[1]
    grid_folder = sys.argv[2]
    folder_with_config = sys.argv[3]

    print(f"Token set (length: {len(token)} characters)")
    print(f"Grid Folder set to: {grid_folder}")
    print(f"Config File set to: {folder_with_config}")

    # --- Main Execution Logic (Copied from Original) ---

    # get input data
    grid_to_send, config = collect_input_data(grid_folder, folder_with_config)
    grid_name = "coordinated_grid"

    # send request to api.
    print("\nSending request to indigo api...")
    response = indigo_optimize_grid(username=username, token=token,
                                    grid_name=grid_name,
                                    grid_to_send=grid_to_send,
                                    config=config,populations=5)
    import pdb; pdb.set_trace()
    result = json.loads(response.text)
    # log response (using print as a simple log)
    print("Response of optimization request:\n{}".format(result))

    try:
        sid = result["sid"]
        logger.info(f"Fetch results via sid {sid}")
        print(f"Fetch results via sid {sid}")
        print(result)
    except Exception as E:
        logger.warning(f"Cannot fetch sid due to {E}")

    print("DONE with initial request.")

    # --- Result Fetching (Simplified to show structure) ---

    sid = 284089639  # Use this to test if the request code below works with already simulated data

    response = get_state_by_sid(username, sid, token=token,
                                namespace="get-optimized-by-sid")
    response = json.loads(response.text)

    export_path = "results_output"  # Enter path to folder where results should be saved (Relative path used here)

    try:
        print(response["progress"])
        if r["progress"] == 100:
            print(f"Status: OK. Saving results to {export_path}/{sid}")
            save_grid_result(response["result_data"], response["grid_name"], export_path)
            print("--- Simulation: Results Saved ---")
    except Exception as E:
        print(f"Cannot fetch from result dct due to {E}")

    print("\nSCRIPT FINISHED")