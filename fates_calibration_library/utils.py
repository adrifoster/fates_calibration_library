"""Helper methods"""

import yaml


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
    return factor


def join_nonempty(*parts: list[str]) -> str:
    """Joins a list of strings but does not include emtpy strings.

    Returns:
        str: concatenated string
    """
    return " ".join(p for p in parts if p and p.strip())

# def join_nonempty(lat, alt, biome):
#     """Join logic:
#     - If altitude is an empty string or NaN, return "latitude biome"
#     - Otherwise return "altitude biome"
#     """
#     def is_empty(val):
#         return val.strip() == ""

#     if is_empty(alt):
#         return f"{lat.strip()} {biome.strip()}"
#     else:
#         return f"{alt.strip()} {biome.strip()}"
