"""ParamSpec class - fully self-describing calibratable FATES parameter."""

from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Optional
import xarray as xr
import numpy as np

import pandas as pd
from fates_calibration_library.param_gen.bounds import ParamBounds

VALID_PARAM_TYPES = {"default", "array_index", "multi_param", "scale_from_root"}
VALID_STRATEGIES = {"default", "posterior"}


@dataclass
class ParamSpec:
    """All metadata for a single calibratable FATES parameter.

    Attributes
    ----------
    name : str
        Calibration handle — the parameter_name from the spreadsheet.
        This is what you use to refer to the parameter everywhere. For
        'default' and 'array_index' types it matches the netCDF variable
        name directly. For 'multi_param' and 'scale_from_root' types the
        actual netCDF variable(s) are in root_params.
    long_name : str
        Human-readable description from the spreadsheet.
    units : str
        Units string from the spreadsheet.
    dims : list[str]
        NetCDF dimension names for this variable, e.g. ['fates_pft'] or
        ['fates_leafage_class', 'fates_pft']. Empty list for scalars.
    param_type : str
        How this parameter gets written to the netCDF file:
        - 'default'         : written directly to the variable named `name`
        - 'array_index'     : written to one index of an array variable
        - 'multi_param'     : a calibration handle for multiple netCDF vars
                              (the actual variable names are in root_params)
        - 'scale_from_root' : value is expressed as a delta from a root param
                              (root param netCDF name is in root_params)
    strategy : str
        How the parameter value is generated during sampling:
        - 'default'   : scaled between a min and max
        - 'posterior' : drawn from an external posterior distribution
    bounds : ParamBounds
        Min and max bounds for this parameter (FixedBound, PercentBound,
        or PFTBound). Call bounds.resolve(default_value) to get concrete
        values at sample time.
    array_index : int | None
        For 'array_index' param_type only: which index of the array to write.
        None for all other types.
    root_params : list[str]
        NetCDF variable names this parameter is linked to. Meaning depends
        on param_type:
        - 'default'                 : empty
        - 'array_index'             : single entry - the original parameter name
        - 'scale_from_root'         : single entry — the root parameter
        - 'multi_param'             : all parameters this handle writes to
    """

    name: str
    long_name: str
    units: str
    dims: list[str]
    param_type: str
    strategy: str
    bounds: ParamBounds
    array_index: Optional[int]
    root_params: list[str]

    def __post_init__(self):
        """Catch errors in parameter set up that would cause failures
        Raises:
            ValueError: Invalid param_type
            ValueError: Invalid strategy
            ValueError: array_index type with no index
            ValueError: scale_from_root/multi_param with no root_params
        """
        if self.param_type not in VALID_PARAM_TYPES:
            raise ValueError(
                f"Invalid param_type '{self.param_type}' for parameter '{self.name}'. "
                f"Must be one of: {sorted(VALID_PARAM_TYPES)}"
            )
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}' for parameter '{self.name}'. "
                f"Must be one of: {sorted(VALID_STRATEGIES)}"
            )
        if self.param_type == "array_index" and self.array_index is None:
            raise ValueError(
                f"Parameter '{self.name}' has param_type 'array_index' "
                "but no array_index value was provided."
            )
        if (
            self.param_type in ["scale_from_root", "multi_param"]
            and not self.root_params
        ):
            raise ValueError(
                f"Parameter '{self.name}' has param_type '{self.param_type}' "
                "but root_params is empty."
            )

    @property
    def is_pft_param(self) -> bool:
        """True if this parameter varies per PFT."""
        return "fates_pft" in self.dims
    
    
    def get_default_value(
        self, default_ds: xr.Dataset
    ) -> float | np.ndarray | list[np.ndarray] | list[float]:
        """Extract the relevant default value(s) from a FATES netCDF parameter dataset.
        
            The correct variable to look up depends on param_type:
 
            - 'default'         : default_ds[name]
            - 'array_index'     : default_ds[root_params[0]][array_index]  (single slice)
            - 'scale_from_root' : default_ds[root_params[0]]     (the root variable)
            - 'multi_param'     : [default_ds[p] for p in root_params]

        Args:
            default_ds (xr.Dataset): The default FATES parameter dataset

        Raises:
            ValueError: Can't get default value for an unknown param_type

        Returns:
            float | np.ndarray | list[np.ndarray] | list[float]: Default parameter value. 
            Scalar or array for most types; list of scalars/arrays for 'multi_param'.
        """
        if self.param_type == "default":
            return default_ds[self.name].values
 
        elif self.param_type == "array_index":
            return default_ds[self.root_params[0]].values[self.array_index]
 
        elif self.param_type == "scale_from_root":
            return default_ds[self.root_params[0]].values
 
        elif self.param_type == "multi_param":
            return [default_ds[p].values for p in self.root_params]
 
        # unreachable given __post_init__ validation, but explicit is better
        raise ValueError(
            f"Cannot get default value for unknown param_type "
            f"'{self.param_type}' on parameter '{self.name}'."
        )


    @classmethod
    def from_row(cls, row: pd.Series, pft_sheet: pd.DataFrame | None = None) -> ParamSpec:
        """Construct a ParamSpec from a single row of the main spreadsheet.

        Args:
            row (pd.Series): A row from the 'main' sheet DataFrame, indexed by column name.
            pft_sheet : (pd.DataFrame | None, Optional): per-parameter PFT sheet for 
            this parameter, if param_min or param_max is 'pft'. Pass None for scalar bounds.

        Returns:
            ParamSpec: ParamSpec instance
        """
        return cls(
            name=str(row["parameter_name"]),
            long_name=str(row.get("long_name", "")),
            units=str(row.get("units", "")),
            dims=_parse_dims(row.get("coord", "")),
            param_type=str(row.get("param_type", "default")).strip(),
            strategy=str(row.get("strategy", "default")).strip(),
            bounds=ParamBounds.from_row_and_sheet(row, pft_sheet),
            array_index=_parse_optional_int(row.get("array_index")),
            root_params=_parse_list(row.get("root_params", "")),
        )


def _parse_dims(value: str | None) -> list[str]:
    """Parse a coord cell like \"['fates_pft']\" into a list of strings.

    Args:
        value (str | None): input coord

    Returns:
        list[str]: list of coordinate strings
    """
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = ast.literal_eval(value.strip())
        if isinstance(result, list):
            return [str(d) for d in result]
        return [str(result)]
    except (ValueError, SyntaxError):
        return [value.strip().strip("[]'\"")]


def _parse_list(value: str | float | None) -> list[str]:
    """Parse a comma-separated list cell into a list of strings

    Accepts Python literals (['a', 'b']), comma-separated strings,
    or a single name. Returns an empty list for blank/null values.

    Args:
        value (str | float | None): input cell string

    Returns:
        list[str]: list of strings
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    stripped = value.strip()
    try:
        result = ast.literal_eval(stripped)
        if isinstance(result, list):
            return [str(r).strip() for r in result]
        return [str(result).strip()]
    except (ValueError, SyntaxError):
        # plain comma-separated fallback
        return [r.strip() for r in stripped.split(",") if r.strip()]


def _parse_optional_int(value: str | None) -> Optional[int]:
    """Return int if value is a valid integer, else None.

    Args:
        value (str | None): intput cell

    Returns:
        Optional[int]: output integer or None
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
