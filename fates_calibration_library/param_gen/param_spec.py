"""ParamSpec class - fully self-describing calibratable FATES parameter."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional
import xarray as xr
import numpy as np
import pandas as pd

from fates_calibration_library.param_gen.bounds import ParamBounds
from .param_type import (
    ParamType,
    SlicedParamType,
    ScaleFromRootParamType,
    MultiParamType,
)

VALID_STRATEGIES = {"default", "posterior"}


@dataclass
class DimIndex:
    """A pinned position in a single netCDF dimension.

    Used on expanded ParamSpec objects to record which dimension and index
    this spec is responsible for writing to.

    Attributes
    ----------
    dim : str
        The netCDF dimension name, e.g. 'fates_pft'.
    index : int
        The 0-based index along that dimension.
    """

    dim: str
    index: int


@dataclass
class ParamSpec:
    """All metadata for a single calibratable FATES parameter.

    Attributes
    ----------
    name : str
        Calibration handle — the parameter_name from the spreadsheet.
        This is what you use to refer to the parameter everywhere. For
        'default' and some 'sliced' types it matches the netCDF variable
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
        - 'sliced'          : one specific index of one dimension is targeted;
                              see slice_dim and slice_index
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
    slice_dim : str | None
        For 'sliced' param_type: which dimension is being indexed into,
        e.g. 'fates_leafage_class' or 'fates_plant_organs'.
        None for all other param types.
    slice_index : int | None
        For 'sliced' param_type: which index along slice_dim to target.
        None for all other types.
    root_params : list[str]
        NetCDF variable names this parameter is linked to. Meaning depends
        on param_type:
        - 'default'                 : empty
        - 'sliced'                  : single entry - the original parameter name
        - 'scale_from_root'         : single entry — the root parameter
        - 'multi_param'             : all parameters this handle writes to
    expand_by_index : bool
        If True, this parameter will be expanded into one independent spec
        per active index during the expansion step (e.g. one per PFT, one
        per plant organ). Each index gets its own LH dimension and can be
        sampled independently.
        If False (default), a single LH value is applied across all indices
        and they move together.
    fixed_indices : dict[str, list[int]]
        Set by the expansion step. Maps dimension name to 0-based indices
        that should be held at their default value when writing. The writer
        uses this to skip those positions regardless of expansion mode.
        Empty dict means no indices are fixed.
    """

    name: str
    long_name: str
    units: str
    dims: list[str]
    param_type: ParamType
    strategy: str
    bounds: ParamBounds
    slice_dim: Optional[str]
    slice_index: Optional[int]
    root_params: list[str]
    expand_by_index: bool = False
    fixed_indices: dict | None = None

    def __post_init__(self):
        """Catch errors in parameter set up that would cause failures
        Raises:
            ValueError: Invalid param_type
            ValueError: Invalid strategy
            ValueError: sliced type with no slice_index, slice_dim, or root_params
            ValueError: slice_dim, slice_index, and root_params set but not sliced_type
            ValueError: scale_from_root/multi_param with no root_params
        """
        if self.fixed_indices is None:
            self.fixed_indices = {}

        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{self.strategy}' for parameter '{self.name}'. "
                f"Must be one of: {sorted(VALID_STRATEGIES)}"
            )

        if isinstance(self.param_type, SlicedParamType):
            slice_parts = [
                self.slice_dim is not None,
                self.slice_index is not None,
                bool(self.root_params),
            ]
            if not (all(slice_parts) or not any(slice_parts)):
                raise ValueError(
                    f"Parameter '{self.name}': slice_dim, slice_index, and root_params "
                    "must all be set or all be None/empty."
                )
        if isinstance(self.param_type, SlicedParamType) and self.slice_dim is None:
            raise ValueError(
                f"Parameter '{self.name}' has param_type 'sliced' "
                "slice_dim, slice_index, and root_params are not set."
            )
        if (
            not isinstance(self.param_type, SlicedParamType)
            and self.slice_dim is not None
        ):
            raise ValueError(
                f"Parameter '{self.name}' has slice_dim set but param_type "
                f"is '{self.param_type}', not 'sliced'."
            )
        if (
            isinstance(self.param_type, (ScaleFromRootParamType, MultiParamType))
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

    @property
    def free_dims(self) -> list[str]:
        """Dimensions that are not pinned by a slice.

        For most parameters this is the same as dims. For 'sliced' params
        slice_dim is removed, leaving only the dimensions that vary freely.

        Returns:
            list[str]: list of dimensions not pinned by a slice
        """
        if self.slice_dim is None:
            return self.dims
        return [d for d in self.dims if d != self.slice_dim]

    def get_default_value(
        self, default_ds: xr.Dataset
    ) -> float | np.ndarray | list[np.ndarray] | list[float]:
        """Extract the relevant default value(s) from a netCDF parameter dataset.
        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray] | list[float]: Default parameter value.
            Scalar or array for most types; list of scalars/arrays for 'multi_param'.
        """
        return self.param_type.get_default(self, default_ds)

    def write(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray | list[np.ndarray],
    ):
        """Write a scaled value into the working parameter dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
        """
        self.param_type.write(self, ds, default_ds, value)

    @classmethod
    def from_row(
        cls, row: pd.Series, pft_sheet: pd.DataFrame | None = None
    ) -> ParamSpec:
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
            param_type=ParamType.from_str(str(row.get("param_type", "default"))),
            strategy=str(row.get("strategy", "default")).strip(),
            bounds=ParamBounds.from_row_and_sheet(row, pft_sheet),
            slice_dim=_parse_optional_str(row.get("slice_dim")),
            slice_index=_parse_optional_int(row.get("slice_index")),
            root_params=_parse_list(row.get("root_params", "")),
            expand_by_index=bool(row.get("expand_by_index", False)),
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


def _parse_optional_str(value: str | None) -> Optional[str]:
    """Return stripped string if non-empty, else None.

    Args:
        value (str | None): input cell

    Returns:
        Optional[str]: stripped string or None
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s if s else None
