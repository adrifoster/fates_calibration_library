"""
PosteriorSource: data class for managing posterior distributions
    for parameter sampling.

PosteriorSource owns one file covering one or more array indices.

Notes
-----
- Column names in each text must match the parameter names in `parameters`.
- `array_indices` can be a list of 0-based integers or the string "all".

Lazy loading
-------------
text files are not read until draw_row() is first called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_DEFAULT_SORT_INDEX = 0


@dataclass
class PosteriorSource:
    """Posterior samples from one text file, covering one or more array indices.

     Attributes
    ----------
    path : Path
        Path to the text file. Columns must match parameter file parameter names.
    array_indices : list[int] | str
        Array indices (0-based) this file covers. "all" means apply to
        every index (broadcast mode).
    parameters : list[str]
        parameter names. Must match column names in the text file.
    sort_param_index: int
        0-based indiex into ``parameters`` selecting the column used to sort
    """

    path: Path
    array_indices: list[int] | str
    parameters: list[str]
    sort_param_index: int = _DEFAULT_SORT_INDEX

    # lazy-loaded state — not part of the public interface or constructor
    _draws: Optional[pd.DataFrame] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        """Validate attributes and set path to be an actual Path

        Raises:
            ValueError: 
                array_indices is a string that is not "all"
                array indices cannot be converted to an int list
        """
        self.path = Path(self.path)

        if isinstance(self.array_indices, str):
            if self.array_indices != "all":
                raise ValueError(
                    f"array_indices must be a list of ints or 'all', "
                    f"got '{self.array_indices}'."
                )
        else:
            if not isinstance(self.array_indices, list):
                self.array_indices = [self.array_indices]
            try:
                self.array_indices = [int(v) for v in self.array_indices]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"array_indices must be a list of ints, "
                    f"but found values that could not be converted: {e}."
                ) from e

    @property
    def is_broadcast(self) -> bool:
        """True if this source applies to all array indices"""
        return self.array_indices == "all"

    def _load(self):
        """Load, validate, and sort the posterior draws from disk.

        Loads the file and then sorts by the chosen parameter column so
        that a [0, 1] input to draw_row() acts as a true quantile index. Sorting preserves
        joint structure — all variables in a row stay together.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If any variable name is missing from the data columns.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"PosteriorSource: cannot find input file '{self.path}'."
            )

        df = pd.read_table(self.path, sep=" ")

        missing = [p for p in self.parameters if p not in df.columns]
        if missing:
            raise ValueError(
                f"PosteriorSource '{self.path}': columns {missing} not found. "
                f"Available columns: {list(df.columns)}"
            )
        sort_col = self.parameters[self.sort_param_index]
        self._draws = (
            df[self.parameters].sort_values(by=sort_col).reset_index(drop=True)
        )

    def draw_row(self, normalized_value: float) -> pd.Series:
        """Return one row using normalized_value as a quantile index.

        Args:
            normalized_value (float): Value in [0, 1]. Maps to a row position in the 
            sorted pre-drawn data frame

        Raises:
            ValueError: normalized_value outside allowed range
            RuntimeError: Dataframe is empty

        Returns:
            pd.Series: One row of posterior draws, indexed by variable name.
        """
        if normalized_value > 1.0:
            raise ValueError (
                f"normalized_value={normalized_value}. "
                "Cannot use a normalized_value greater than 1.0"
            )
        if normalized_value < 0.0:
            raise ValueError (
                f"normalized_value={normalized_value}. "
                "Cannot use a normalized_value less than 0.0"
            )
        if self._draws is None:
            self._load()
        n = len(self._draws)
        if n == 0:
            raise RuntimeError (
                f"PosteriorSource '{self.path}' is empty  - cannot sample."
            )
        idx = min(int(normalized_value * n), n - 1)
        return self._draws.iloc[idx]


    def unscale(self, value: float) -> float:
        """Find the quantile corresponding to a drawn value.

        Args:
            value (float): actual parameter input

        Raises:
            RuntimeError: Dataframe is empty
            ValueError: input ``value`` is outside range of distribution

        Returns:
            pd.Series: One row of posterior draws, indexed by variable name.
        """
        if self._draws is None:
            self._load()
        n = len(self._draws)
        if n == 0:
            raise RuntimeError (
                f"PosteriorSource '{self.path}' is empty  - cannot sample."
            )
        if n == 1:
            return 1.0
        
        sort_col = self.parameters[self.sort_param_index]
        max_val = self._draws[sort_col].iloc[-1]
        min_val = self._draws[sort_col].iloc[0]
        if value > max_val:
            raise ValueError (
                f"value: {value} > maximum value on dataset. "
                f"maximum value is {max_val}"
            )
        if value < min_val:
            raise ValueError (
                f"value: {value} < minimum value on dataset. "
                f"minimum value is {min_val}"
            )
        
        col = self._draws[sort_col].values
        idx = np.searchsorted(col, value)  # col is already sorted
        idx = np.clip(idx, 0, len(col) - 1)
        return float(idx) / (len(col) - 1)
