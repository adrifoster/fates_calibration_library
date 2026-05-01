"""
PosteriorSource: data classes for managing posterior distributions
    for parameter sampling.

PosteriorSource owns one file covering one or more array indices.

Notes
-----
- Column names in each text must match the parameter names in `parameters`.
- `array_indices` can be a list of 0-based integers or the string "all".
- Paths are resolved relative to the YAML file's directory unless absolute.

Lazy loading
-------------
text files are not read until draw() is first called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


_DEFAULT_SORT_INDEX = 0


@dataclass
class PosteriorSource:
    """Posterior samples from one text file, covering one or more indices.

     Attributes
    ----------
    path : Path
        Path to the text file. Columns must match parameter file parameter names.
    array_indices : list[int] | "all"
        array indices (0-based) this file covers. "all" means apply to
        every index (broadcast mode).
    parameters : list[str]
        parameter names. Must match column names in the text file.
    _draws : pd.DataFrame | None
        Cached sample rows. None until prepare() is called.
    """

    path: Path
    array_indices: Union[list[int], str]  # list of ints or "all"
    parameters: list[str]
    sort_index: int = _DEFAULT_SORT_INDEX
    _draws: Optional[pd.DataFrame] = field(default=None, repr=False)
    _n_rows: Optional[int] = field(default=None, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise IOError(f"Cannot find input file {self.path}.")
        if not isinstance(self.array_indices, str):
            self.array_indices = list(self.array_indices)
        else:
            if self.array_indices != "all":
                raise ValueError(
                    f"array_indices must be 'all' or a list of ints, not {self.array_indices}"
                )

    @property
    def is_broadcast(self) -> bool:
        """True if this source applies to all array indices"""
        return self.array_indices == "all"

    def prepare(self):
        """Pre-draw n_samples rows from the file.

        Randomly samples n_samples rows then sorts by the first variable
        so that input [0-1] acts as a true quantile index. Sorting preserves
        joint structure — all variables in a row stay together.

        Raises:
            ValueError: If any variable name is missing from the data columns.
        """
        df = pd.read_table(self.path, sep=" ")

        missing = [v for v in self.parameters if v not in df.columns]
        if missing:
            raise ValueError(
                f"PosteriorSource '{self.path}': columns {missing} not found. "
                f"Available columns: {list(df.columns)}"
            )
        self._draws = (
            df[self.parameters]
            .sort_values(by=self.parameters[self.sort_index])
            .reset_index(drop=True)
        )

    def draw_row(self, value: float) -> pd.Series:
        """Return one row using value as a quantile index.

        Args:
            value (float): Value in [0, 1]. Maps to a row position in the sorted
            pre-drawn subsample

        Raises:
            RuntimeError: If prepare() has not been called yet.

        Returns:
            pd.Series: One row of posterior draws, indexed by variable name.
        """
        if self._draws is None:
            raise RuntimeError(
                f"PosteriorSource '{self.path}': prepare() must be called "
                "before draw_row()."
            )
        n = len(self._draws)
        idx = min(int(value * n), n - 1)
        return self._draws.iloc[idx]
