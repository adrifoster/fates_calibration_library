"""Helper methods
"""
import configparser
from dask_jobqueue import PBSCluster


def config_to_dict(config_file: str) -> dict:
    """Convert a config file to a python dictionary

    Args:
        config_file (str): full path to config file

    Returns:
        dictionary: dictionary of config file
    """
    config = configparser.ConfigParser()
    config.read(config_file)

    dictionary = {}
    for section in config.sections():
        dictionary[section] = {}
        for option in config.options(section):
            dictionary[section][option] = config.get(section, option)

    return dictionary

def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to True or False.

    Args:
        val (str): input string

    Raises:
        ValueError: can't figure out what the string should be converted to

    Returns:
        bool: True or False
    """
    if val.lower() in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val.lower() in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value {val}")