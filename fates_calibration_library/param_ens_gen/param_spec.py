"""ParamSpec class - metadata for a parameter."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional
import math

import pandas as pd

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"parameter_name", "coord", "param_type"})


@dataclass
class ParamSpec:  # pylint: disable=too-many-instance-attributes
    """All metadata for a single calibratable parameter. Belongs to a Parameter object.

    Note:
    This is a pure data container which mirrors exactly one row of the input calibration
    spreadsheet. It has no knowledge of parameter datasets or distribution
    resolution — those concerns belong to Parameter, which owns a ParamSpec instance.

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
        Parameter names this parameter is linked to. Meaning depends on param_type, which
        determines the concrete class of Parameter
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
        """Validate field-level invariants that hold regardless of param_type.

        These are constraints on the fields themselves — a slice field being set
        on a non-sliced type is always wrong, regardless of what the type-specific
        logic does. Type-specific required-field validation (e.g. 'sliced needs
        slice_dim') belongs on the Parameter subclass that owns that logic.
        Raises:
            ValueError: If slice_dim or slice_index are set on a non-sliced param.
            ValueError: If root_param is set on a non-scale_from_root param.
        """
        if self.param_type != "sliced":
            if self.slice_dim is not None:
                raise ValueError(
                    f"Parameter '{self.name}' has slice_dim set but param_type "
                    f"is '{self.param_type}', not 'sliced'."
                )
            if self.slice_index is not None:
                raise ValueError(
                    f"Parameter '{self.name}' has slice_index set but param_type "
                    f"is '{self.param_type}', not 'sliced'."
                )
        if self.param_type != "scale_from_root" and self.root_param is not None:
            raise ValueError(
                f"Parameter '{self.name}' has root_param set but param_type "
                f"is '{self.param_type}', not 'scale_from_root'."
            )

    @classmethod
    def from_row(cls, row: pd.Series) -> ParamSpec:
        """Construct a ParamSpec from a single row of the main spreadsheet.

        Args:
            row (pd.Series): A row from the 'main' sheet DataFrame, indexed by column name.

        Returns:
            ParamSpec: ParamSpec instance
        """

        missing = _REQUIRED_COLUMNS - set(row.index)
        if missing:
            raise KeyError(f"Row is missing required columns: {sorted(missing)}")
        blank = [col for col in _REQUIRED_COLUMNS if not str(row[col]).strip()]
        if blank:
            raise ValueError(
                f"Row has blank values in required columns: {sorted(blank)}"
            )

        return cls(
            name=str(row["parameter_name"]),
            category=str(row.get("category", "")),
            subcategory=str(row.get("subcategory", "")),
            long_name=str(row.get("long_name", "")),
            units=str(row.get("units", "")),
            dims=_parse_dims(row.get("coord", "")),
            param_type=str(row["param_type"]).strip(),
            slice_dim=_parse_optional_str(row.get("slice_dim")),
            slice_index=_parse_optional_int(row.get("slice_index")),
            base_params=_parse_list(row.get("base_params", "")),
            root_param=_parse_optional_str(row.get("root_param")),
        )


# ----------------------------------------------------------------------------------------
# Private parsing helpers
# ----------------------------------------------------------------------------------------


def _is_nan(value: float) -> bool:
    """Return True if *value* is a float NaN."""
    try:
        return math.isnan(value)
    except TypeError:
        return False


def _parse_dims(value: str | None) -> list[str]:
    """Parse a coord cell like \"['fates_pft']\" into a list of strings.

    An explicit empty-list string (``"[]"``) is the correct way to signal
    a scalar parameter. A missing / NaN value raises ``ValueError`` so that
    accidental blank cells in the spreadsheet are caught early rather than
    silently treated as scalars.

    Args:
        value (str | None): Raw coord cell value from the spreadsheet.

    Returns:
        list[str]: List of dimension-name strings, empty for scalars.
    """
    if value is None or (isinstance(value, float) and _is_nan(value)):
        raise ValueError(
            "coord cell is missing (NaN/None). "
            "Use an explicit empty list '[]' for scalar parameters."
        )
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        result = ast.literal_eval(value.strip())
        if isinstance(result, list):
            return [str(d) for d in result]
        # bare non-list literal (e.g. a bare string without brackets)
        return [str(result)]
    except (ValueError, SyntaxError):
        return [value.strip().strip("[]'\"")]


def _parse_list(value: str | float | None) -> list[str]:
    """Parse a comma-separated list cell into a list of strings

    Accepts Python literals (['a', 'b']), comma-separated strings,
    or a single name. Returns an empty list for blank/null values.

    Args:
        value (str | float | None): Raw cell value from the spreadsheet.

    Returns:
        list[str]: List of stripped strings, empty list for blank/null input.
    """
    if value is None or (isinstance(value, float) and _is_nan(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    stripped = value.strip()
    try:
        result = ast.literal_eval(stripped)
        if isinstance(result, list):
            return [str(r).strip() for r in result]
        # bare non-list literal (e.g. a single quoted string or integer)
        return [str(result).strip()]
    except (ValueError, SyntaxError):
        # plain comma-separated fallback
        return [r.strip() for r in stripped.split(",") if r.strip()]


def _parse_optional_int(value: str | None) -> Optional[int]:
    """Return int if value is a valid integer, else None.

    Args:
        value (str | None): Raw cell value from the spreadsheet.

    Returns:
        Optional[int]: Parsed integer, or ``None`` for blank/null/non-numeric input.
    """
    if value is None or (isinstance(value, float) and _is_nan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_str(value: str | None) -> Optional[str]:
    """Return stripped string if non-empty, else None.

    Args:
        value (str | None): Raw cell value from the spreadsheet.

    Returns:
        Optional[str]: Stripped string, or ``None`` for blank/null/whitespace-only input.
    """
    if value is None or (isinstance(value, float) and _is_nan(value)):
        return None
    s = str(value).strip()
    return s if s else None
