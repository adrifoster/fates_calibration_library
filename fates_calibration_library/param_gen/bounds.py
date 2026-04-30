"""
Bound types for parameter min/max values

A parameter's min and max can each be one of three things:
    - FixedBound    : a plain number, resolved immediately
    - PercentBound  : a percentage of the default value (e.g. "50percent")
    - PFTBound      : per-PFT fixed values loaded from a parameter-specific sheet

All three share a common resolve() interface. The sampler calls resolve() to
get the actual float/array it needs — it never needs to know which type
it's dealing with.

NullBound is returned for posterior parameters — strategy.requires_bounds()
is False for those, so bounds are never resolved at sample time. NullBound
exists only as a safe placeholder so Parameter.__init__ can unconditionally
construct a ParamBounds without special-casing posterior params.

PFTBound is always used for both min and max together (never mixed with
Fixed or Percent on the same parameter).  PFT-specific values must be
fixed numbers — no percent syntax is allowed in per-parameter sheets.

Usage
-----
    # at load time (parsing the spreadsheet)
    min_bound = Bound.parse("50percent", bound_side="min")
    max_bound = Bound.parse("0.9", bound_side="max")

    # at sample time
    min_val = min_bound.resolve(default_value)
    max_val = max_bound.resolve(default_value)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy import Strategy

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Bound(ABC):
    """Abstract base for a single parameter bound (min or max)."""

    @abstractmethod
    def resolve(
        self, default_value: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        """Return the concrete bound value.

        Args:
            default_value (float | np.ndarray | None, optional): default parameter value.
                Required for PercentBound; ignored by FixedBound and PFTBound.
                Defaults to None.

        Returns:
            float | np.ndarray: min/max value
        """

    @staticmethod
    def parse(cell_value: str | float | int, bound_side: str) -> Bound:
        """Parse a single param_min or param_max cell into a Bound object.

        Returns a NullBound for posterior parameters (i.e. when the cell
        value matches Strategy.POSTERIOR). This keeps Parameter.__init__
        unconditional — it always constructs a ParamBounds, and callers
        gate resolution on strategy.requires_bounds().

        Accepted formats:
            - Plain number : '0.9', '1', '-0.5'
            - Percent      : '50percent', '50%', '50 percent', '50 %'
            - Posterior    : 'posterior' (returns NullBound)

        Args:
            cell_value (str | foat | int): Raw value from the spreadsheet cell.
            bound_side (str): 'min' or 'max' — needed to apply percent change in the right
            direction.

        Raises:
            ValueError
            if bound_side is not 'min' or 'max', if the cell is empty,
            if the value is 'pft' (must go through PFTBound.from_sheet),
            if a percent value is zero (would make min == max == default),
            or if the value cannot be parsed as a number.

        Returns:
            Bound: A FixedBound or PercentBound.
            (PFTBound must be constructed via PFTBound.from_sheet()).
        """
        if bound_side not in ("min", "max"):
            raise ValueError(f"bound_side must be 'min' or 'max', got '{bound_side}'")

        if cell_value is None or (
            isinstance(cell_value, float) and pd.isna(cell_value)
        ):
            raise ValueError("Bound cell is empty.")

        as_str = str(cell_value).strip().lower()

        # check strategy to see if this is a posterior marker
        try:
            strategy = Strategy.parse(as_str)
            if strategy.requires_posterior():
                return NullBound(value=None)
        except ValueError:
            pass  # not a strategy string; continue parsing as a bound value

        # wrong class method
        if as_str == "pft":
            raise ValueError(
                "Cannot parse 'pft' bound with Bound.parse(). "
                "Use PFTBound.from_sheet() instead."
            )

        # we accept "50%" or "50percent"
        normalised = as_str.replace(" ", "").replace("%", "percent")

        if "percent" in normalised:
            percent_str = normalised.replace("percent", "").strip()
            try:
                percent = float(percent_str)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse percent bound '{cell_value}': "
                    f"expected a number before 'percent' or '%', got '{percent_str}'."
                ) from exc

            if percent == 0.0:
                raise ValueError(
                    f"Percent bound of 0 for param_{bound_side}='{cell_value}' would "
                    "make min == max == default. Use a non-zero percentage."
                )
            return PercentBound(percent=percent, bound_side=bound_side)

        # Otherwise must be a plain number
        try:
            return FixedBound(value=float(as_str))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse bound '{cell_value}' for param_{bound_side}. "
                "Expected a number, a percent (e.g. '50percent', '50perc', or '50%'), "
                "or 'posterior'."
            ) from exc


# ---------------------------------------------------------------------------
# Concrete bound types
# ---------------------------------------------------------------------------


@dataclass
class NullBound(Bound):
    """Placeholder bound for parameters whose strategy does not use bounds.

    Never resolved at sample time — callers must check
    strategy.requires_bounds() before calling resolve().
    """

    value: None

    def resolve(self, default_value: float | np.ndarray | None = None) -> None:
        """Return the concrete bound value.

        Args:
            default_value (float | np.ndarray | None, optional): Default value from
            parameter file. Defaults to None.

        Returns:
            None: Nothing
        """
        return None


@dataclass
class FixedBound(Bound):
    """A plain numeric bound, fully resolved at parse time."""

    value: float

    def resolve(self, default_value: float | np.ndarray | None = None) -> float:
        """Return the concrete bound value.

        Args:
            default_value (float | np.ndarray | None): Default value from
            parameter file. Defaults to None.

        Returns:
            float: Scalar value
        """
        return self.value


@dataclass
class PercentBound(Bound):
    """A bound defined as a percentage change from the default value.

    e.g. "50percent" for min = default_value - abs(default_value * 0.50)
         "50percent" for max = default_value + abs(default_value * 0.50)
    """

    percent: float
    bound_side: str  # 'min' or 'max'

    def resolve(
        self, default_value: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        """Return the concrete bound value.

        Args:
            default_value (float | np.ndarray): default value from parameter file

        Raises:
            ValueError: Didn't get a default value

        Returns:
            float | np.ndarray: bound
        """
        if default_value is None:
            raise ValueError(
                "PercentBound.resolve() requires a default_value but got None."
            )
        delta = np.abs(default_value * (self.percent / 100.0))
        if self.bound_side == "min":
            return default_value - delta
        return default_value + delta


@dataclass
class PFTBound(Bound):
    """Per-PFT fixed bounds loaded from a parameter-specific sheet.

    Values are indexed by PFT (0-based internally, 1-based in the sheet).
    Always used for both min and max together — never mixed with other types.
    """

    values: np.ndarray  # shape: (n_pfts,), dtype: float

    def resolve(self, default_value: float | np.ndarray | None = None) -> np.ndarray:
        return self.values

    @classmethod
    def from_sheet(cls, sheet: pd.DataFrame, col: str) -> PFTBound:
        """Construct a PFTBound from a per-parameter sheet column.

        Args:
            sheet (pd.DataFrame): The per-parameter sheet, with columns: pft_index,
            pft_name, param_min, param_max. Rows are in PFT order (1-indexed).
            col (str): 'param_min' or 'param_max'.

        Raises:
            ValueError: PFT-specific bounds must be fixed numbers

        Returns:
            PFTBound: PFT bound
        """
        raw = sheet[col].values

        values = []
        for i, v in enumerate(raw):
            as_str = str(v).strip().lower()
            try:
                value = float(as_str)
            except ValueError as exc:
                raise ValueError(
                    f"PFT-specific bounds must be fixed numbers, but found "
                    f"'{v}' in row {i} of column '{col}'. "
                    "Use a plain number for per-PFT bounds."
                ) from exc

            values.append(value)

        return cls(values=np.array(values, dtype=float))


# ---------------------------------------------------------------------------
# ParamBounds container
# ---------------------------------------------------------------------------


@dataclass
class ParamBounds:
    """Min and max bounds for a single parameter.

    Attributes
    ----------
    min_bound : Bound
        Lower bound (FixedBound, PercentBound, or PFTBound).
    max_bound : Bound
        Upper bound (FixedBound, PercentBound, or PFTBound).
    """

    min_bound: Bound
    max_bound: Bound

    def resolve(
        self,
        default_value: float | np.ndarray | None = None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Resolve both bounds and return (min_val, max_val).

        Args:
            default_value (float | np.ndarray | None, optional): default value from
            parameter file. Required if either bound is a PercentBound. Defaults to None.

        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: (min_val, max_val)
        """
        return (
            self.min_bound.resolve(default_value),
            self.max_bound.resolve(default_value),
        )

    @classmethod
    def from_row_and_sheet(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
    ) -> ParamBounds:
        """Construct ParamBounds from a main sheet row.

        Args:
            row (pd.Series): A row from the main sheet.
            pft_sheet (pd.DataFrame | None, optional): The per-parameter PFT sheet,
            required when param_min or param_max is 'pft'. Ignored otherwise.
            Defaults to None.

        Raises:
            ValueError: No pft_sheet supplied if param_min/param_max is pft
            ValueError: Mixing of pft and some other bounds type

        Returns:
            ParamBounds: Bounds
        """
        min_raw = str(row.get("param_min", "")).strip().lower()
        max_raw = str(row.get("param_max", "")).strip().lower()

        if min_raw == "pft" or max_raw == "pft":
            if min_raw != max_raw:
                raise ValueError(
                    f"Parameter '{row.get('parameter_name')}': param_min and param_max "
                    "must both be 'pft' or neither — mixing is not supported."
                )
            if pft_sheet is None:
                raise ValueError(
                    f"Parameter '{row.get('parameter_name')}' has "
                    "param_min or param_max == 'pft' but no pft_sheet was supplied."
                )
            return cls(
                min_bound=PFTBound.from_sheet(pft_sheet, "param_min"),
                max_bound=PFTBound.from_sheet(pft_sheet, "param_max"),
            )

        return cls(
            min_bound=Bound.parse(row.get("param_min"), bound_side="min"),
            max_bound=Bound.parse(row.get("param_max"), bound_side="max"),
        )
