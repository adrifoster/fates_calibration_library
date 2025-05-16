"""Helper methods"""

import operator
import os
import yaml
from functools import reduce


def get_config_file(file_name: str) -> dict:
    """Reads in a YAML config file to a dictionary

    Args:
        file_name (str): path to config file

    Returns:
        dict: dictionary output
    """

    with open(file_name, "r") as f:
        config = yaml.safe_load(f)

    return config


def evaluate_conversion_factor(factor) -> float:
    """Evaluates the conversion factor if it's an equation, otherwise returns the number."""
    if isinstance(factor, dict) and "operation" in factor and "operands" in factor:
        op = factor["operation"]
        operands = factor["operands"]
        if op == "multiply":
            result = 1
            for num in operands:
                result *= num
            return result
        elif op == "add":
            return sum(operands)
        elif op == "divide":
            if len(operands) < 2:
                raise ValueError("Divide operation requires at least two operands.")
            return reduce(operator.truediv, operands)
    return factor


def join_nonempty(*parts: list[str]) -> str:
    """Joins a list of strings but does not include emtpy strings.

    Returns:
        str: concatenated string
    """
    return " ".join(p for p in parts if p and p.strip())


def should_skip_file(path: str, clobber: bool) -> bool:
    """Checks whether file should be skipped

    Args:
        path (str): file path
        clobber (bool): whether or not to overwrite file that does exist

    Returns:
        bool: whether to skip file
    """
    return os.path.isfile(path) and not clobber

def validate_dict_keys(d: dict, required_keys: set, dict_name: str):
    """Checks to make sure required dictionary keys are present

    Args:
        d (dict): input dictionary
        required_keys (set): set of required keys
        dict_name (str): name of dictionary

    Raises:
        ValueError: Missing keys
    """
    missing = required_keys - d.keys()
    if missing:
        raise ValueError(f"Missing keys in {dict_name}: {', '.join(missing)}")