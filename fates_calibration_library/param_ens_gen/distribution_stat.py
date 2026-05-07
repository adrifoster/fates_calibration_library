"""
Data classes for parsing and storing parameters for sampling

Distribution statistics can each be one of three things:
    - FixedStat    : a plain number, resolved immediately
    - PercentStat  : a percentage of the default value (e.g. "50percent")
    - PFTStat      : per-PFT fixed values loaded from a parameter-specific sheet

All three share a common resolve() interface. The caller uses resolve() to
get the actual float/array it needs — it never needs to know which type
it's dealing with.

PFTStat is always used for all stats needed (never mixed with
Fixed or Percent on the same parameter).  PFT-specific values must be
fixed numbers — no percent syntax is allowed in per-parameter sheets.

Usage
-----
    # at load time (parsing the spreadsheet)
    min_bound = DistributionStat.parse("50percent", stat_type="min")
    max_bound = DistributionStat.parse("0.9", stat_type="max")

    # at sample time
    min_val = min_bound.resolve(default_value)
    max_val = max_bound.resolve(default_value)

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

ACCEPTED_STATS = {"min", "max", "mean", "sd"}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DistributionStat(ABC):
    """Abstract base class for a set of stats for a single parameter.

    Subclasses must implement:
    - resolve(self, default_value): return the actual stat value
    """

    @staticmethod
    def parse(row: pd.Series, stat_type: str, pft_sheet: pd.DataFrame | None=None) -> DistributionStat:
        """Factory method for DistributionStat

        Accepted formats:
            - Plain number : '0.9', '1', '-0.5'
            - Percent      : '50percent', '50%', '50 percent', '50 %'

        Args:
            row (pd.Series): one row of the input spreadsheet
            stat_type (str): needed to apply percent change in the right
            direction.
            pft_sheet (pd.DataFrame | None, optional): per-parameter sheet for
                PFT-specific stats. Defaults to None.

        Raises:
            ValueError
            if stat_type is not in ACCEPTED_STATS, if the cell is empty,
            if a percent value is zero (would make min == max == default),
            if a percent value is negative
            or if the value cannot be parsed as a number.

        Returns:
            DistributionStat: A FixedStat, PFTStat, or PercentStat.
        """
        
        # get raw value from cell and make sure it exists
        raw = str(row.get(f"param_{stat_type}")).strip().lower()
        if raw is None or (
            isinstance(raw, float) and pd.isna(raw)
        ):
            raise ValueError(
                f"Parameter '{row.get('parameter_name')}': param_{stat_type} is empty."
            )
        
        if stat_type not in ACCEPTED_STATS:
            raise ValueError(
                f"Unknown stat_type '{stat_type}'. " f"Valid stats: {ACCEPTED_STATS}"
            )

        # convert to string
        as_str = str(raw).strip().lower()

        # PFTStat
        if as_str == "pft":
            if pft_sheet is None:
                raise ValueError(
                    f"stat_type='{stat_type}' is 'pft' but no pft_sheet was supplied."
                )
            return PFTStat.from_sheet(pft_sheet, f"param_{stat_type}")

        # we accept "50%" or "50percent"
        normalised = as_str.replace(" ", "").replace("%", "percent")
        
        # PercentStat
        if "percent" in normalised:

            # 50% of default for mean doesn't make much sense
            # but this could be removed if someone actually wants to do this
            if stat_type == "mean":
                raise ValueError(
                    f"stat_type '{stat_type}' does not support percent stats. "
                    f"Use a fixed or PFT-specific value."
                )
            percent_str = normalised.replace("percent", "").strip()
            try:
                percent = float(percent_str)
            except ValueError as exc:
                raise ValueError(
                    f"Could not parse percent stat '{raw}': "
                    f"expected a number before 'percent' or '%', got '{percent_str}'."
                ) from exc

            if percent == 0.0:
                raise ValueError(
                    f"Percent stat of 0 for param_{stat_type}='{raw}' is "
                    "not allowed. Use a non-zero percentage or a fixed value."
                )
            if percent < 0.0:
                raise ValueError(
                    f"Negative percents (param_{stat_type}='{raw}') are "
                    "not allowed. Use a positive percentage."
                )
            return PercentStat(percent=percent, stat_type=stat_type)

        # FixedStat
        try:
            return FixedStat(value=float(as_str))
        except ValueError as exc:
            raise ValueError(
                f"Could not parse stat '{raw}' for param_{stat_type}. "
                "Expected a number, a percent (e.g. '50percent' or '50%'), "
                "or 'pft'."
            ) from exc
        
    @abstractmethod
    def resolve(
        self, default_value: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        """Return the concrete stat value.

        Args:
            default_value (float | np.ndarray | None, optional): default parameter value.
                Required for PercentStat; ignored by FixedStat and PFTStat.
                Defaults to None.

        Returns:
            float | np.ndarray: min/max value
        """


# ---------------------------------------------------------------------------
# Concrete types
# ---------------------------------------------------------------------------


@dataclass
class FixedStat(DistributionStat):
    """A plain numeric stat, fully resolved at parse time."""

    value: float

    def resolve(self, default_value: float | np.ndarray | None = None) -> float:
        """Return the concrete stat value"""
        return self.value


@dataclass
class PercentStat(DistributionStat):
    """A stat defined as a percentage change from the default value.

    e.g. "50percent" for min = default_value - abs(default_value * 0.50)
         "50percent" for max = default_value + abs(default_value * 0.50)
         "50percent" for sd = abs(default_value + abs(default_value * 0.50))
    
    We do not currently allow PercentStat to be used with "mean"
        
    """

    percent: float
    stat_type: str

    def resolve(
        self, default_value: float | np.ndarray | None = None
    ) -> float | np.ndarray:
        """Return the concrete stat value."""
        assert self.stat_type != "mean", (
            "PercentStat should never be constructed with stat_type='mean' — "
            "parse() blocks this. If you removed that check, also remove PercentStat support "
            "for mean."
        )
        if default_value is None:
            raise ValueError(
                "PercentStat.resolve() requires a default_value but got None."
            )
        delta = np.abs(default_value * (self.percent / 100.0))
        if self.stat_type == "min":
            return default_value - delta
        if self.stat_type == "max":
            return default_value + delta
        if self.stat_type == "sd":
            return np.abs(default_value + delta)
        assert False, (
            f"Unhandled stat_type '{self.stat_type}' in PercentStat.resolve(). "
            f"Add a branch here if you've added a new stat_type to ACCEPTED_STATS."
        )


@dataclass
class PFTStat(DistributionStat):
    """Per-PFT fixed stats loaded from a parameter-specific sheet.

    Values are indexed by PFT (0-based internally, 1-based in the sheet).
    Always used for both min and max together — never mixed with other types.
    """

    values: np.ndarray  # shape: (n_pfts,), dtype: float

    def resolve(self, default_value: float | np.ndarray | None = None) -> np.ndarray:
        """Return the concrete stat value."""
        return self.values

    @classmethod
    def from_sheet(cls, sheet: pd.DataFrame, col: str) -> PFTStat:
        """Construct a PFTStat from a per-parameter sheet column.

        Args:
            sheet (pd.DataFrame): The per-parameter sheet, with columns: pft_index,
            pft_name, param_min, param_max. Rows are in PFT order (1-indexed).
            col (str): e.g., 'param_min' or 'param_max'.

        Raises:
            ValueError
                PFT sheet missing "pft_index" column
                PFT sheet missing requested parameter stat column
                PFT-specific stats must be fixed numbers

        Returns:
            PFTStat: PFT stat
        """
        if "pft_index" not in sheet.columns:
            raise ValueError(
                f"PFT sheet is missing a 'pft_index' column. "
                f"Found columns: {list(sheet.columns)}"
            )
        sheet = sheet.sort_values(by='pft_index', ascending=True)
        
        row = sheet.get(col)
        if row is None:
            raise ValueError(
                f"PFT sheet is missing a {col} column. "
                f"Found columns: {list(sheet.columns)}"
            )
        raw = row.values

        values = []
        for i, v in enumerate(raw):
            as_str = str(v).strip().lower()
            try:
                value = float(as_str)
            except ValueError as exc:
                raise ValueError(
                    f"PFT-specific stats must be fixed numbers, but found "
                    f"'{v}' in row {i} of column '{col}'. "
                    "Use a plain number for per-PFT stats."
                ) from exc

            values.append(value)

        return cls(values=np.array(values, dtype=float))
