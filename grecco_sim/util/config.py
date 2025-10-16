import json
from pathlib import Path


def result_dir() -> Path:
    """Get results dir: result in main directory"""
    return Path(__file__).parents[2] / "results"


def data_root() -> Path:
    """ Assume data is stored in repository root. """
    return Path(__file__).parents[2] / "data"


def get_config() -> dict:
    """Get user config if existing."""
    conf_path = data_root() / "conf.json"
    if not conf_path.exists():
        # No config present
        return {}

    with open(conf_path, "r", encoding="utf-8") as fp:
        return json.load(fp)
