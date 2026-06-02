"""
Sampler class

This class in in charge of sampling a parameter given an input normalized_value, and
normalizing a concrete parameter to a normalized value

"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np

from .distribution_stat import DistributionStat
from .posterior_source import PosteriorSource, _DEFAULT_SORT_INDEX

@dataclass
class SampleContext:
    """Sampling context passed to Sampler.
 
    Each Sampler subclass uses only the fields relevant to it and ignores
    the rest. This avoids a mixed-concern signature on the abstract interface
    where some arguments are only meaningful to one subclass.
 
    Attributes
    ----------
    default_value : float | np.ndarray | list[np.ndarray] | None
        Default value(s) for this parameter from the dataset.
        Required by UniformSampler to resolve percent bounds.
        Ignored by PosteriorSampler.
    array_index : int | None
        The active array index when sampling an expanded (per-index) parameter.
        None when sampling all free indices together (broadcast mode).
        Used by PosteriorSampler to select the correct PosteriorSource.
        Ignored by UniformSampler.
    n_indices : int
        Total number of positions along the free dimension (1 for scalars).
        Used by PosteriorSampler in broadcast mode to size the output arrays.
        Ignored by UniformSampler.
    """
 
    default_value: float | np.ndarray | list[np.ndarray] | None = None
    array_index: int | None = None
    n_indices: list[int] | None = None


class Sampler(ABC):
    """Class for parameter sampling.

    Subclasses must implement:
        - __init__(self, row, pft_sheet, posterior_config): parse all
          sampling configuration from the spreadsheet row at construction time.
        - sample(self, normalized_value, context): draw a value given a normalised input.
        - normalize(self, value, context): convert a concrete value back to normalised [0, 1].
    """

    _registry: dict[str, type[Sampler]] = {}

    def __init_subclass__(cls, sampler_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Sampler._registry[sampler_type] = cls
        
    @classmethod
    def from_row_and_sheet(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ) -> Sampler:
        """Construct Sampler from a main sheet row, pft_sheet, or posterior_config

        Args:
            row (pd.Series): A row from the main sheet.
            pft_sheet (pd.DataFrame | None, optional): The per-parameter PFT sheet,
                required when a relevant column is 'pft'. Ignored otherwise.
                Defaults to None.

        Raises:
            Unknown strategy

        Returns:
            Sampler: Sampler
        """
        param_strategy = str(row.get("strategy", "")).strip()
        subclass = cls._registry.get(param_strategy)
        if subclass is None:
            raise ValueError(
                f"Unknown strategy '{param_strategy}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(
            row, pft_sheet, posterior_config
        )  # pylint: disable=not-callable


    @abstractmethod
    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter

        Args:
            normalized_value (float): normalized [0-1] input value
            context: Sampling context. Each subclass uses the fields relevant to its 
                strategy and ignores the rest.
        Returns:
            float | np.ndarray: sampled parameter value.
        """

    @abstractmethod
    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Args:
            value (float | np.ndarray): concrete parameter to normalize
            context: Sampling context. Each subclass uses the fields relevant to its 
                strategy and ignores the rest.
            
        Returns:
            float | np.ndarray: normalized value(s) in [0 to 1]
        """

class UniformSampler(Sampler, sampler_type="uniform"):
    """Uniform Sampler - scales between a minimum and a maximum given an input [0-1] value

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

        if (min_raw == "pft") != (max_raw == "pft"):
            raise ValueError(
                f"Parameter '{row.get('parameter_name')}': param_min and param_max "
                "must both be 'pft' or neither — mixing is not supported."
            )

        self.min_stat = DistributionStat.parse(row, "min", pft_sheet)
        self.max_stat = DistributionStat.parse(row, "max", pft_sheet)

    def resolve_bounds(
        self,
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
        max_val = self.max_stat.resolve(default_value)

        _validate_bounds(min_val, max_val)

        return min_val, max_val

    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter"""
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
        min_val, max_val = self.resolve_bounds(context.default_value)
        return min_val + normalized_value * (max_val - min_val)

    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""
        min_val, max_val = self.resolve_bounds(context.default_value)
        
        if np.any(value > max_val):
            raise ValueError (
                f"value: {value} exceeds maximum value(s). "
                f"maximum value is {max_val}"
            )
        if np.any(value < min_val):
            raise ValueError (
                f"value: {value} is below minimum value(s). "
                f"minimum value is {min_val}"
            )
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
                sort_param_index=posterior_config.get("sort_index", _DEFAULT_SORT_INDEX),
            )
            for file_entry in posterior_config["files"]
        ]
        self.sources = sources
        
    def sample(
        self,
        normalized_value: float,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Generate a sample for a parameter"""
        if context.array_index is not None:
            return self._draw_for_index(normalized_value, context.array_index)
        return self._draw_broadcast(normalized_value, context.n_indices)

    def normalize(
        self,
        value: float | np.ndarray,
        context: SampleContext,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value."""
        if context.array_index is not None:
            return self._unscale_for_index(value, context.array_index)
        return self._unscale_broadcast(value, context.n_indices)
        
    
    def _draw_for_index(self, normalized_value: float, array_index: int) -> list[np.ndarray]:
        """draws from a PosteriorSource given an input array_index

        Args:
            normalized_value (float): input normalized [0-1] value to use as quantile
            array_index (int): array index we are pulling for

        Returns:
            list[np.ndarray]: output parameter value(s)
        """
        source = self._source_for_index(array_index)
        row = source.draw_row(normalized_value)
        return [np.array([row[v]]) for v in self.parameters]
    
    def _unscale_for_index(self, value: float, array_index: int) -> float:
        """convert a concrete parameter value into a normalized [0-1] value for a 
        specific array index

        Args:
            value (float): concrete parameter value
            array_index (int): array index to use

        Returns:
            float: normalized value
        """
        source = self._source_for_index(array_index)
        return source.unscale(value)

    def _source_for_index(self, array_index: int) -> PosteriorSource:
        """Return the correct PosteriorSource given an input index

        Args:
            array_index (int): input array_index for PosteriorSource

        Raises:
            ValueError: No source found for the input array index

        Returns:
            PosteriorSource: Correct PosteriorSource
        """
        for source in self.sources:
            if source.is_broadcast or array_index in source.array_indices:
                return source
        raise ValueError(
            f"No source found for array index {array_index}"
            f"Check your posterior_sources.yaml."
        )

    def _unscale_broadcast(self, value: float, n_indices: list[int] | None) -> list[np.ndarray]:
        result = [np.zeros(n_indices) for _ in self.parameters]
        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            unscaled = self.sources[0].unscale(value)
            for k, var in enumerate(self.parameters):
                result[k][:] = unscaled
        else:
            for source in self.sources:
                row = source.unscale(0.01)
                indices = (
                    range(n_indices) if source.is_broadcast else source.array_indices
                )
                for array_idx in indices:
                    for k, var in enumerate(self.parameters):
                        result[k][array_idx] = row
        return result
    
    def _draw_broadcast(self, normalized_value: float, n_indices: list[int] | None) -> list[np.ndarray]:
        result = [np.zeros(n_indices) for _ in self.parameters]

        if len(self.sources) == 1 and self.sources[0].is_broadcast:
            row = self.sources[0].draw_row(normalized_value)
            for k, var in enumerate(self.parameters):
                result[k][:] = row[var]
        else:
            for source in self.sources:
                row = source.draw_row(normalized_value)
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
            f"Parameter min or max is None  - cannot scale "
            f"(min={min_val}, max={max_val}). Check inputs"
        )

    min_arr = np.asarray(min_val)
    max_arr = np.asarray(max_val)
    if np.any(min_arr > max_arr):
        raise ValueError(
            f"Parameter has min > max " f"(min={min_val}, max={max_val}). Check inputs"
        )
