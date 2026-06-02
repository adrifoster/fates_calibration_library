"""Utility functions"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

def read_param_list(
    param_data_file: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read in the excel file that sets up all the parameters

    Args:
        param_data_file (Path): path to excel file

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: main dataframe, dictionary of
        sheets with pft-specific parameter values
    """

    xl = pd.ExcelFile(param_data_file, engine="xlrd")
    main = pd.read_excel(xl, sheet_name="main")

    pft_sheets = {}
    for sheet in xl.sheet_names:
        if sheet != "main":
            pft_sheets[f"{sheet}"] = pd.read_excel(xl, sheet_name=sheet)
    return main, pft_sheets

