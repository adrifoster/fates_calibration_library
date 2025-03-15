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
