import json
import pathlib


def result_dir() -> pathlib.Path:
    """Get results dir: result in main directory"""
    return pathlib.Path(__file__).parent.parent.parent / "results"


def data_path():
    """Get path of data in main dir."""
    return pathlib.Path(__file__).parent.parent.parent / "data"


def get_config() -> dict:
    """Get user config if existing."""
    conf_path = data_path() / "conf.json"
    if not conf_path.exists():
        # No config present
        return {}

    with open(conf_path, "r", encoding="utf-8") as fp:
        return json.load(fp)
