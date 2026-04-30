"""ParamSpec class - fully self-describing calibratable FATES parameter."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class ParamSpec:
    """All metadata for a single calibratable parameter. Belongs to a Parameter object.
    
    Note: 
    This is a pure data container which mirrors exactly one row (and potentially an extra 
    sheet) of the input calibration spreadsheet. It has no knowledge of parameter 
    datasets or distribution resolution — those concerns belong to Parameter, 
    which owns a ParamSpec instance.

    Attributes
    ----------
    name : str
        Calibration handle — the parameter_name from the spreadsheet.
        This is what you use to refer to the parameter everywhere. For
        'default' and some 'sliced' types it matches the netCDF variable
        name directly. For 'joint' and 'scale_from_root' types the
        actual parameter(s) are in base_params.
    category: str
        Parameter category; useful description for grouping parameters
    subcategory: str
        Parameter subcategory; useful description for grouping parameters
    long_name : str
        Human-readable description from the spreadsheet.
    units : str
        Units string from the spreadsheet.
    dims : list[str]
        Dimension names for this parameter, e.g. ['fates_pft'],
        ['fates_leafage_class', 'fates_pft'], or [] for scalars.
    param_type : str
        How this parameter gets scaled and written to parameter file
    slice_dim : str | None
        For 'sliced' param_type: which dimension is being indexed into,
        e.g. 'fates_leafage_class' or 'fates_plant_organs'.
        None for all other param types.
    slice_index : int | None
        For 'sliced' param_type: which index along slice_dim to target.
        None for all other types.
    base_params : list[str]
        Parameter names this parameter is linked to. Meaning depends on param_type:
        - 'default'                 : empty
        - 'sliced'                  : single entry - the original parameter name
        - 'scale_from_root'         : single entry — the original parameter name
        - 'multi_param'             : all parameters this handle writes to
    root_param: str | None
        For 'scale_from_root' param_type: the parameter to scale from
        None for all other types.
    """

    name: str
    long_name: str
    category: str
    subcategory: str
    units: str
    dims: list[str]
    param_type: str
    slice_dim: Optional[str]
    slice_index: Optional[int]
    root_param: Optional[str]
    base_params: list[str]

    def __post_init__(self):
        """Catch errors in parameter set up that would cause failures
        Raises:
            ValueError: sliced type with no slice_index, slice_dim, or root_params
            ValueError: slice_dim, slice_index, and root_params set but not sliced_type
            ValueError: scale_from_root/joint with no root_params
        """
        # slice_dim, slice_index, and base_params must always be set together
        if self.param_type == "sliced":
            slice_parts = [
                self.slice_dim is not None,
                self.slice_index is not None,
                bool(self.base_params),
            ]
            if not (all(slice_parts) or not any(slice_parts)):
                raise ValueError(
                    f"Parameter '{self.name}': slice_dim, slice_index, and base_params "
                    "must all be set or all be None/empty."
                )
        if self.param_type == "sliced" and self.slice_dim is None:
            raise ValueError(
                f"Parameter '{self.name}' has param_type 'sliced' "
                "slice_dim, slice_index, and base_params are not set."
            )
        if self.param_type != "sliced" and self.slice_dim is not None:
            raise ValueError(
                f"Parameter '{self.name}' has slice_dim set but param_type "
                f"is '{self.param_type}', not 'sliced'."
            )

        if self.param_type == "scale_from_root":
            root_parts = [
                self.root_param is not None,
                bool(self.base_params),
            ]
            if not (all(root_parts) or not any(root_parts)):
                raise ValueError(
                    f"Parameter '{self.name}': root_param and base_params "
                    "must all be set or all be None/empty."
                )

        if self.param_type == "scale_from_root" and self.root_param is None:
            raise ValueError(
                f"Parameter '{self.name}' has param_type 'scale_from_root' "
                "root_param and base_params are not set."
            )

        if self.param_type != "scale_from_root" and self.root_param is not None:
            raise ValueError(
                f"Parameter '{self.name}' has root_param set but param_type "
                f"is '{self.param_type}', not 'scale_from_root'."
            )

        if self.param_type == "joint" and not self.base_params:
            raise ValueError(
                f"Parameter '{self.name}' has param_type 'joint' "
                "base_params are not set."
            )

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

    @classmethod
    def from_row(cls, row: pd.Series) -> ParamSpec:
        """Construct a ParamSpec from a single row of the main spreadsheet.

        Args:
            row (pd.Series): A row from the 'main' sheet DataFrame, indexed by column name.

        Returns:
            ParamSpec: ParamSpec instance
        """

        return cls(
            name=str(row["parameter_name"]),
            category=str(row.get("category", "")),
            subcategory=str(row.get("subcategory", "")),
            long_name=str(row.get("long_name", "")),
            units=str(row.get("units", "")),
            dims=_parse_dims(row.get("coord", "")),
            param_type=str(row.get("param_type", "")).strip(),
            slice_dim=_parse_optional_str(row.get("slice_dim")),
            slice_index=_parse_optional_int(row.get("slice_index")),
            base_params=_parse_list(row.get("base_params", "")),
            root_param=_parse_optional_str(row.get("root_param")),
        )

# ---------------------------------------------------------------------------
# Private parsing helpers
# ---------------------------------------------------------------------------

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
