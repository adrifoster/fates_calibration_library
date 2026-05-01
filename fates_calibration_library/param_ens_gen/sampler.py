"""Sampler class"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np

from .distribution_stat import DistributionStat, PFTStat
from .posterior import PosteriorSource, _DEFAULT_SORT_INDEX


class Sampler(ABC):
    """Abstract base for Sampler class."""

    _registry: dict[str, type[Sampler]] = {}

    def __init_subclass__(cls, sampler_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Sampler._registry[sampler_type] = cls

    @abstractmethod
    def __init__(
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):
        """Concrete class must implement this"""

    @abstractmethod
    def sample(
        self,
        normalized_value: float,
        default_value: float | np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ):
        """Generate a sample for a parameter"""

    @abstractmethod
    def normalize(
        self,
        value: float | np.ndarray,
        default_value: float | np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""

    @classmethod
    def from_row_and_sheet(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ) -> Sampler:
        """Construct Sampler from a main sheet row and pft_sheet.

        Args:
            row (pd.Series): A row from the main sheet.
            pft_sheet (pd.DataFrame | None, optional): The per-parameter PFT sheet,
            required when a relevant column is 'pft'. Ignored otherwise.
            Defaults to None.

        Raises:
            ValueError: No pft_sheet supplied if param_min/param_max is pft
            ValueError: Mixing of pft and some other bounds type

        Returns:
            Sampler: Sampler
        """
        param_strategy = str(row.get("strategy", "")).strip()
        subclass = cls._registry.get(param_strategy)
        if subclass is None:
            raise ValueError(
                f"Unknown param_type '{param_strategy}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(row, pft_sheet, posterior_config)


class UniformSampler(Sampler, sampler_type="uniform"):
    """Uniform Sampler - scales between a minimum and a maximum given an input [0-1 value]

    Attributes
    ===========
    min_stat: DistributionStat
        Minimum parameter bound
    max_stat: DistributionStat
        Maximum parameter bound
    """

    def __init__(
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):

        min_raw = str(row.get("param_min", "")).strip().lower()
        max_raw = str(row.get("param_max", "")).strip().lower()

        if min_raw == "" or max_raw == "":
            raise ValueError(
                "Parameters with strategy=uniform must supply a 'param_min' and 'param_max' value"
            )

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
            self.min_stat = PFTStat.from_sheet(pft_sheet, "param_min")
            self.max_stat = PFTStat.from_sheet(pft_sheet, "param_max")
        else:
            self.min_stat = DistributionStat.parse(min_raw, stat_type="min")
            self.max_stat = DistributionStat.parse(max_raw, stat_type="max")

    def resolve_bounds(
        self,
        mask: np.ndarray | None,
        default_value: float | np.ndarray | None,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Resolve min and max bounds and return (min_val, max_val).

        Args:
            default_value (float | np.ndarray | None, optional): default value from
            parameter file. Required if either bound is a PercentBound. Defaults to None.

        Returns:
            tuple[float | np.ndarray, float | np.ndarray]: (min_val, max_val)
        """

        min_val = self.min_stat.resolve(default_value)
        max_val = self.min_stat.resolve(default_value)

        _validate_bounds(min_val, max_val, mask=mask)

        return min_val, max_val

    def sample(
        self,
        normalized_value: float,
        mask: np.ndarray | None = None,
        default_value: float | np.ndarray | None = None,
        array_index: int | None = None,
        n_indices: int | None = None,
    ):

        min_val, max_val = self.resolve_bounds(mask, default_value)

        return min_val + normalized_value * (max_val - min_val)

    def normalize(
        self,
        value: float,
        mask: np.ndarray | None = None,
        default_value: float | np.ndarray | None = None,
        array_index: int | None = None,
        n_indices: int | None = None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""

        min_val, max_val = self.resolve_bounds(mask, default_value)

        # normalize
        return (value - min_val) / (max_val - min_val)


class PosteriorSampler(Sampler, sampler_type="posterior"):
    """Posterior Sampler - pulls from a posterior distribution

    Attributes
    ===========
    parameters: list[str]
        list of column/parameter names in each PosteriorSource
    sources: list[PosteriorSource]
        list of PosteriorSources that can be used to draw from a distribution
    """

    def __init__(
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):
        if posterior_config is None:
            raise ValueError(
                f"Parameter '{row.get('parameter_name')}' has "
                "strategy='posterior' but no posterior_sources yaml was supplied."
            )

        self.parameters = posterior_config["parameters"]
        sources = [
            PosteriorSource(
                path=Path(file_entry["path"]),
                array_indices=file_entry["array_indices"],
                parameters=self.parameters,
                sort_index=posterior_config.get("sort_index", _DEFAULT_SORT_INDEX),
            )
            for file_entry in posterior_config["files"]
        ]
        self.sources = sources

        for source in self.sources:
            source.prepare()

    def sample(
        self,
        normalized_value: float,
        mask: np.ndarray | None = None,
        default_value: float | np.ndarray | None = None,
        array_index: int | None = None,
        n_indices: int | None = None,
    ):
        if array_index is not None:
            return self._draw_for_index(normalized_value, array_index)
        else:
            return self._draw_broadcast(normalized_value, n_indices)

    def normalize(
        self,
        value: float | np.ndarray,
        mask: np.ndarray | None = None,
        default_value: float | np.ndarray | None = None,
        array_index: int | None = None,
        n_indices: int | None = None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""
        pass

    def _draw_for_index(self, value: float, array_index: int) -> list[np.ndarray]:
        source = self._source_for_index(array_index)
        row = source.draw_row(value)
        return [np.array([row[v]]) for v in self.parameters]

    def _source_for_index(self, array_index: int) -> PosteriorSource:
        for source in self.sources:
            if source.is_broadcast or array_index in source.array_indices:
                return source
        raise ValueError(
            f"No source found for array index {array_index}"
            f"Check your posterior_sources.yaml."
        )

    def _draw_broadcast(self, value: float, n_indices: int) -> list[np.ndarray]:
        result = [np.zeros(n_indices) for _ in self.parameters]

        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            row = self.sources[0].draw_row(value)
            for k, var in enumerate(self.parameters):
                result[k][:] = row[var]
        else:
            for source in self.sources:
                row = source.draw_row(value)
                indices = (
                    range(n_indices) if source.is_broadcast else source.array_indices
                )
                for array_idx in indices:
                    for k, var in enumerate(self.parameters):
                        result[k][array_idx] = row[var]

        return result


def _validate_bounds(
    min_val: float | np.ndarray,
    max_val: float | np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    """Raise error if any min > max after resolution

    Args:
        min_val (float | np.ndarray): minimum value
        max_val (float | np.ndarray): maximum value

    Raises:
        ValueError: Parameter min > max
    """
    if min_val is None or max_val is None:
        raise ValueError(
            f"Parameter min or max is None  - cannot scale"
            f"(min={min_val}, max={max_val}). Check inputs"
        )

    min_arr = np.asarray(min_val)
    max_arr = np.asarray(max_val)
    if mask is not None and min_arr.ndim > 0:
        min_arr = min_arr[mask]
        max_arr = max_arr[mask]
    if np.any(min_arr > max_arr):
        raise ValueError(
            f"Parameter has min > max " f"(min={min_val}, max={max_val}). Check inputs"
        )
