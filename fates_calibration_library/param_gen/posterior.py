"""
PosteriorSource and PosteriorConfig: data classes for managing posterior distributions 
    for parameter sampling.

PosteriorSource owns one file covering one or more array indices.
PosteriorConfig owns all PosteriorSource objects for one parameter set
and knows how to draw values from them.

YAML format
-----------
    fates_allom_agb:
      variables: [fates_allom_agb1, fates_allom_agb2]
      files:
        - path: "jags_out/allom_agb_pft1.csv"
          pft_indices: [0]
        - path: "jags_out/allom_agb_pft2.csv"
          pft_indices: [1, 2]
 
    fates_leafn_vert_scaler:
      variables: [fates_leafn_vert_scaler_coeff1, fates_leafn_vert_scaler_coeff2]
      files:
        - path: "jags_out/leafn_scaler.csv"
          pft_indices: "all"

Notes
-----
- Column names in each text must match the parameter names in `parameters`.
- `array_indices` can be a list of 0-based integers or the string "all".
- Paths are resolved relative to the YAML file's directory unless absolute.

Lazy loading
-------------
text files are not read until draw() is first called. Only n_samples rows are cached. The
full file is never fully loaded into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml
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
            raise IOError(
                f"Cannot find input file {self.path}."
            )
        if not isinstance(self.array_indices, str):
            self.array_indices = list(self.array_indices)
        else:
            if self.array_indices != 'all':
                raise ValueError(
                    f"array_indices must be 'all' or a list of ints, not {self.array_indices}"
                )
            
    @property
    def is_broadcast(self) -> bool:
        """True if this source applies to all array indices"""
        return self.array_indices == 'all'

    def prepare(self, n_samples: int) -> None:
        """Pre-draw n_samples rows from the file.
        
        Randomly samples n_samples rows then sorts by the first variable
        so that input [0-1] acts as a true quantile index. Sorting preserves
        joint structure — all variables in a row stay together.

        Args:
            n_samples (int): Number of ensemble members to pre-draw for.

        Raises:
            ValueError: If any variable name is missing from the data columns.
        """
        df = pd.read_table(self.path, sep = " ")
 
        missing = [v for v in self.parameters if v not in df.columns]
        if missing:
            raise ValueError(
                f"PosteriorSource '{self.path}': columns {missing} not found. "
                f"Available columns: {list(df.columns)}"
            )
 
        self._draws = (
            df[self.parameters]
            .sample(n=n_samples, replace=n_samples > len(df))
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
    
@dataclass
class PosteriorConfig:
    """All posterior sources for one calibration parameter set.
 
    Attributes
    ----------
    param_name : str
        Calibration handle — matches parameter_name in the spreadsheet.
    sources : list[PosteriorSource]
        One PosteriorSource per text file.
    parameters : list[str]
        parameter names this parameter set writes to.
    """
    param_name: str
    sources: list[PosteriorSource]
    parameters: list[str]
    
    def prepare(self, n_samples: int) -> None:
        """Pre-draw samples from all sources.

        Args:
            n_samples (int): number of samples
        """
        for source in self.sources:
            source.prepare(n_samples)
            
    def draw(self, value: float, array_index: Optional[int], n_indices: int) -> list[np.ndarray]:
        if array_index is not None:
            return self._draw_for_index(value, array_index)
        else:
            return self._draw_broadcast(value, n_indices)
    
    
    def _draw_for_index(
        self, value: float, array_index: int
    ) -> list[np.ndarray]:
        source = self._source_for_pft(array_index)
        row = source.draw_row(value)
        return [np.array([row[v]]) for v in self.parameters]
    
    def _source_for_pft(self, array_index: int) -> PosteriorSource:
        for source in self.sources:
            if source.is_broadcast or array_index in source.array_indices:
                return source
        raise ValueError(
            f"PosteriorConfig '{self.param_name}': no source found for "
            f"PFT index {array_index}. Check your posterior_sources.yaml."
        )
    
    def _draw_broadcast(
        self, value: float, n_indices: int
    ) -> list[np.ndarray]:
        result = [np.zeros(n_indices) for _ in self.parameters]
 
        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            row = self.sources[0].draw_row(value)
            for k, var in enumerate(self.parameters):
                result[k][:] = row[var]
        else:
            for source in self.sources:
                row = source.draw_row(value)
                indices = range(n_indices) if source.is_broadcast else source.array_indices
                for array_idx in indices:
                    for k, var in enumerate(self.parameters):
                        result[k][array_idx] = row[var]
 
        return result
    
    @classmethod
    def from_yaml(
        cls, yaml_path: Union[str, Path]
    ) -> dict[str, PosteriorConfig]:
        """Load all posterior configurations from a YAML file.

        Args:
            yaml_path (Union[str, Path]): Path to the posterior_sources.yaml file. 
                File paths within the YAML are resolved relative to the YAML file's directory
                unless they are absolute.

        Returns:
            dict[str, PosteriorConfig]: Mapping of parameter name to PosteriorConfig.
        """
        yaml_path = Path(yaml_path)
        base_dir = yaml_path.parent
 
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
 
        if not raw:
            return {}
 
        configs = {}
        for param_name, entry in raw.items():
            parameters = entry["parameters"]
            sort_index = entry.get('sort_index', _DEFAULT_SORT_INDEX)
            sources = [
                PosteriorSource(
                    path=(
                        Path(fe["path"])
                        if Path(fe["path"]).is_absolute()
                        else base_dir / fe["path"]
                    ),
                    array_indices=fe["array_indices"],
                    parameters=parameters,
                    sort_index=sort_index
                )
                for fe in entry["files"]
            ]
            configs[param_name] = cls(
                param_name=param_name,
                sources=sources,
                parameters=parameters,
            )
 
        return configs