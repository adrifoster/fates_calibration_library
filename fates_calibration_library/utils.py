"""Helper methods
"""
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