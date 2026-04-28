"""
Scaler - abstract base class for parameter scaling strategies
A Scaler takes a ParamSpec, a normalized value in [0, 1], and an optional
default_value, and returns a concrete parameter value ready to be written
to a parameter file.

Concrete implementations
------------------------
DefaultScaler : linearly interpolates between resolved min and max bounds.

Adding a new strategy:
----------------------
Subclass Scaler, implement scale(), and it's ready to use.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .param_spec import ParamSpec

class Scaler(ABC):
    """Abstract base for parameter scaling strategies.

    A Scaler is stateless — the same instance can be reused across any
    number of parameters and samples.
    """

    @abstractmethod
    def scale(
        self,
        spec: ParamSpec,
        lh_value: float,
        default_value: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Convert a normalized [0, 1] value into a concrete parameter value.

        Args:
            spec (ParamSpec): The parameter being scaled. Provides bounds and metadata.
            lh_value (float): A value in [0, 1] from the sampler (e.g. Latin Hypercube).
            default_value (float | np.ndarray | None, optional): The default parameter
            value from the parameter file. Required if either bound is a PercentBound;
            ignored otherwise. Defaults to None.

        Returns:
            float | np.ndarray: The scaled parameter value. Returns a float for
            non-PFT parameters with fixed/percent bounds; returns np.ndarray for
            PFTBound parameters.
        """


class DefaultScaler(Scaler):
    """Linearly interpolates between resolved min and max bounds.

    For a given lh_value in [0, 1]:
        result = min_val + lh_value * (max_val - min_val)

    Works for all three bound types (Fixed, Percent, PFT) — the bounds
    handle their own resolution, this class just does the interpolation.
    """

    def scale(
        self,
        spec: ParamSpec,
        lh_value: float,
        default_value: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Convert a normalized [0, 1] value into a concrete parameter value.

        Args:
            spec (ParamSpec): The parameter being scaled. Provides bounds and metadata.
            lh_value (float): A value in [0, 1] from the sampler (e.g. Latin Hypercube).
            default_value (float | np.ndarray | None, optional): The default parameter
            value from the parameter file. Required if either bound is a PercentBound;
            ignored otherwise. Defaults to None.

        Returns:
            float | np.ndarray: The scaled parameter value. Returns a float for
            non-PFT parameters with fixed/percent bounds; returns np.ndarray for
            PFTBound parameters
        """

        # get the bounds
        min_val, max_val = spec.bounds.resolve(default_value)
        if min_val is None or max_val is None:
            raise ValueError(
                f"Parameter '{spec.name}' has NullBounds — cannot scale. "
        )
        self._validate_bounds(spec.name, min_val, max_val)

        # scale
        return min_val + lh_value * (max_val - min_val)
    
    def normalize(
        self,
        spec: ParamSpec,
        value: float | np.ndarray,
        default_value: float | np.ndarray | None = None,
    ) -> float | np.ndarray:
        """Convert a concrete parameter value into a normalized [0, 1] value.

        Args:
            spec (ParamSpec): The parameter being scaled. Provides bounds and metadata.
            value (float | np.ndarray): Parameter value
            default_value (float | np.ndarray | None, optional): The default parameter
            value from the parameter file. Required if either bound is a PercentBound;
            ignored otherwise. Defaults to None.

        Returns:
            float | np.ndarray: The normalized parameter value. Returns a float for
            non-PFT parameters with fixed/percent bounds; returns np.ndarray for
            PFTBound parameters
        """

        # get the bounds
        min_val, max_val = spec.bounds.resolve(default_value)
        self._validate_bounds(spec.name, min_val, max_val)

        # normalize
        return (value - min_val) / (max_val - min_val)

    def _validate_bounds(
        self,
        name: str,
        min_val: float | np.ndarray,
        max_val: float | np.ndarray,
    ) -> None:
        """Raise error if any min >= max after resolution

        Args:
            name (str): parameter name
            min_val (float | np.ndarray): minimum value
            max_val (float | np.ndarray): maximum value

        Raises:
            ValueError: Parameter min > max
        """
        if np.any(np.asarray(min_val) > np.asarray(max_val)):
            raise ValueError(
                f"Parameter '{name}' has min > max after resolving bounds "
                f"(min={min_val}, max={max_val}). Check the spreadsheet."
            )
