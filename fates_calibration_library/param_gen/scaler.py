"""
Scaler - abstract base class for parameter scaling strategies
A Scaler takes a min and max bounds, a normalized value in [0, 1], and returns a 
concrete parameter value ready to be written to a parameter file.

Concrete implementations
------------------------
DefaultScaler : linearly interpolates between resolved min and max bounds.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Scaler(ABC):
    """Abstract base for parameter scaling strategies.

    A Scaler is stateless — the same instance can be reused across any
    number of parameters and samples.
    """

    @abstractmethod
    def scale(
        self,
        min_bound: float | np.ndarray,
        max_bound: float | np.ndarray,
        normalized_val: float,
        mask: np.ndarray | None,
    ) -> float | np.ndarray:
        """Convert a normalized [0, 1] value into a concrete parameter value.

        Args:
            min_bound (float | np.ndarray): minimum bound
            max_bound (float | np.ndarray): maximum bound
            normalized_val (float): A value in [0, 1] from the sampler (e.g. Latin Hypercube).
            mask (np.ndarray | None) optional mask of fixed indices. 
            We only validate non-fixed indices

        Returns:
            float | np.ndarray: The scaled parameter value.
        """
    
    @abstractmethod
    def normalize(
        self,
        min_bound: float | np.ndarray,
        max_bound: float | np.ndarray,
        value: float | np.ndarray,
        mask: np.ndarray | None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Args:
            min_bound (float | np.ndarray): minimum bound
            max_bound (float | np.ndarray): maximum bound
            value (float | np.ndarray): Parameter value
            mask (np.ndarray | None) optional mask of fixed indices. 
            We only validate non-fixed indices

        Returns:
            float | np.ndarray: The normalized parameter value.
        """

class DefaultScaler(Scaler):
    """Linearly interpolates between resolved min and max bounds.

    For a given normalized_val in [0, 1]:
        result = min_val + normalized_val * (max_val - min_val)

    Works for all three bound types (Fixed, Percent, PFT) — the bounds
    handle their own resolution, this class just does the interpolation.
    """

    def scale(
        self,
        min_bound: float | np.ndarray,
        max_bound: float | np.ndarray,
        normalized_val: float,
        mask: np.ndarray | None,
    ) -> float | np.ndarray:
        """Convert a normalized [0, 1] value into a concrete parameter value.

        Args:
            min_bound (float | np.ndarray): minimum bound
            max_bound (float | np.ndarray): maximum bound
            normalized_val (float): A value in [0, 1] from the sampler (e.g. Latin Hypercube).
            mask (np.ndarray | None) optional mask of fixed indices. 
            We only validate non-fixed indices

        Returns:
            float | np.ndarray: The scaled parameter value.
        """
        # validate the bounds
        _validate_bounds(min_bound, max_bound, mask=mask)

        # scale
        return min_bound + normalized_val * (max_bound - min_bound)
    
    def normalize(
        self,
        min_bound: float | np.ndarray,
        max_bound: float | np.ndarray,
        value: float | np.ndarray,
        mask: np.ndarray | None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Args:
            min_bound (float | np.ndarray): minimum bound
            max_bound (float | np.ndarray): maximum bound
            value (float | np.ndarray): Parameter value
            mask (np.ndarray | None) optional mask of fixed indices. 
            We only validate non-fixed indices

        Returns:
            float | np.ndarray: The normalized parameter value.
        """

        # validate the bounds
        _validate_bounds(min_bound, max_bound, mask=mask)

        # normalize
        return (value - min_bound) / (max_bound - min_bound)

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
