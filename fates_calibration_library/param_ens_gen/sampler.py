"""Sampler class"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import numpy as np
import xarray as xr

from .bounds import Bound, PFTBound

class Sampler(ABC):
    """Abstract base for Sampler class.
    """
    
    _registry: dict[str, type[Sampler]] = {}

    def __init_subclass__(cls, sampler_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Sampler._registry[sampler_type] = cls

    @abstractmethod
    def __init__(self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        ):
        """Concrete class must implement this"""
        
    @abstractmethod
    def sample(self, normalized_value: float,
            default_value: float | np.ndarray | None = None,
            mask: np.ndarray | None=None):
        """Generate a sample for a parameter
        """
        
    @abstractmethod
    def normalize(self, value: float) -> float | np.ndarray | list[np.ndarray]:
        """Normalize a value"""
        
    @classmethod
    def from_row_and_sheet(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        default_ds: xr.DataFrame | None = None,
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
        return subclass(row, pft_sheet)


class UniformSampler(Sampler, sampler_type='uniform'):
    """Uniform Sampler - scales between a minimum and a maximum given an input [0-1 value]
    
    Attributes
    ===========
    min_bound: Bound
    max_bound: Bound
    """
    
    def __init__(
        self,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        ):
        
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
            self.min_bound=PFTBound.from_sheet(pft_sheet, "param_min")
            self.max_bound=PFTBound.from_sheet(pft_sheet, "param_max")
        
        self.min_bound=Bound.parse(row.get("param_min"), bound_side="min")
        self.max_bound=Bound.parse(row.get("param_max"), bound_side="max")
        
    def resolve_bounds(
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
    
    def sample(self, normalized_value: float,
               default_value: float | np.ndarray | None = None,
               mask: np.ndarray | None=None):
        
        min_val, max_val = self.resolve_bounds(default_value)
        
        _validate_bounds(min_val, max_val, mask=mask)
        
        return min_val + normalized_value * (max_val - min_val)
    
    
    def normalize(self, value: float):
        pass
    

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
            f"Parameter has min > max "
            f"(min={min_val}, max={max_val}). Check inputs"
        )
