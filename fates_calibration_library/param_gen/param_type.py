"""
ParamType - abstract base class for parameter type-specific logic

Each subclass owns two operations for one param_type:
    - get_default() : extract relevant default value from a dataset
    - write()       : write a scaled value into a working dataset

Adding a new param_type
-----------------------
1. Subclass ParamType and implement get_default() and write().
2. Add an entry to PARAM_TYPE_REGISTRY at the bottom of this file.
Nothing else needs to change.

Registry
--------
PARAM_TYPE_REGISTRY maps the string from the spreadsheet to a ParamType
instance. Constructed once at module load; all instances are stateless and
shared.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr


if TYPE_CHECKING:
    from .param_spec import ParamSpec
class ParamType(ABC):
    """Abstract base for parameter type-specific logic.

    ParamType instances are stateless — one instance per subclass is
    shared across all ParamSpec objects of that type.
    """

    @abstractmethod
    def get_default(
        self,
        spec: ParamSpec,
        default_ds: xr.Dataset,
    ) -> float | np.ndarray | list[np.ndarray]:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            spec (ParamSpec): The parameter whose default value is needed.
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """

    @abstractmethod
    def write(
        self,
        spec: ParamSpec,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]] | None=None,
    ) -> None:
        """Write a scaled value into the working dataset.

        Args:
            spec (ParamSpec): The parameter being written.
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """

    @staticmethod
    def from_str(param_type_str: str) -> ParamType:
        """Look up a ParamType instance from a spreadsheet string.

        Args:
            param_type_str (str): The param_type string from the spreadsheet.

        Raises:
            ValueError: If the string is not in PARAM_TYPE_REGISTRY.

        Returns:
            ParamType: ParamType istance
        """
        key = param_type_str.strip().lower()
        if key not in PARAM_TYPE_REGISTRY:
            raise ValueError(
                f"Unknown param_type '{param_type_str}'. "
                f"Valid types: {sorted(PARAM_TYPE_REGISTRY)}"
            )
        return PARAM_TYPE_REGISTRY[key]


# ----------------------------------------------------------------------------------------
# Concrete param types
# ----------------------------------------------------------------------------------------


class DefaultParamType(ParamType):
    """Standard parameter: written directly to ds[name]."""

    def get_default(self, spec: ParamSpec, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            spec (ParamSpec): The parameter whose default value is needed.
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return default_ds[spec.name].values

    def write(
        self,
        spec: ParamSpec,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None=None,
    ) -> None:
        """Write a scaled value into the working dataset.

        Args:
            spec (ParamSpec): The parameter being written.
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """
        arr = ds[spec.name].values.copy()
        free_dim = spec.free_dims[0] if spec.free_dims else None
        fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []

        if spec.active_index is not None:
            arr[spec.active_index.index] = _as_scalar(value, spec.name)
        else:
            arr = _broadcast_to_array(arr, value, fixed, spec.name)

        ds[spec.name].values = arr


class SlicedParamType(ParamType):
    """Parameter that targets one slice of a dimension."""

    def get_default(self, spec: ParamSpec, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            spec (ParamSpec): The parameter whose default value is needed.
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return (
            default_ds[spec.root_params[0]]
            .isel({spec.slice_dim: spec.slice_index})
            .values
        )

    def write(
        self,
        spec: ParamSpec,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None=None,
    ) -> None:
        """Write a scaled value into the working dataset.

        Args:
            spec (ParamSpec): The parameter being written.
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """
        arr = ds[spec.root_params[0]].values.copy()
        da_dims = list(ds[spec.root_params[0]].dims)
        slice_axis = da_dims.index(spec.slice_dim)
        free_dim = spec.free_dims[0] if spec.free_dims else None

        # build index tuple pointing at the slice
        idx = [slice(None)] * arr.ndim
        idx[slice_axis] = spec.slice_index

        if spec.active_index is not None:
            # expanded — single cell
            idx[da_dims.index(spec.active_index.dim)] = spec.active_index.index
            arr[tuple(idx)] = _as_scalar(value, spec.root_params[0])
        else:
            # not expanded — broadcast across free dim at this slice
            fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []
            slice_arr = arr[tuple(idx)].copy()
            slice_arr = _broadcast_to_array(
                slice_arr, value, fixed, spec.root_params[0]
            )
            arr[tuple(idx)] = slice_arr

        ds[spec.root_params[0]].values = arr


class ScaleFromRootParamType(ParamType):
    """Parameter whose value is root + delta."""

    def get_default(self, spec: ParamSpec, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            spec (ParamSpec): The parameter whose default value is needed.
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return default_ds[spec.root_params[0]].values

    def write(
        self,
        spec: ParamSpec,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None=None,
    ) -> None:
        """Write a scaled value into the working dataset.

        Args:
            spec (ParamSpec): The parameter being written.
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """
        root_arr = ds[spec.root_params[0]].values.copy()  # already written
        arr = ds[spec.name].values.copy()
        delta = value

        if spec.active_index is not None:
            i = spec.active_index.index
            arr[i] = root_arr[i] + delta
        else:
            free_dim = spec.free_dims[0] if spec.free_dims else None
            fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []

            if arr.ndim == 0 or not spec.free_dims:
                arr = root_arr + delta
            else:
                default_arr = default_ds[spec.name].values
                for i in range(len(arr)):
                    if i in fixed:
                        arr[i] = default_arr[i]
                    else:
                        arr[i] = root_arr[i] + delta

        ds[spec.name].values = arr


class MultiParamType(ParamType):
    """Calibration handle for multiple parameters (e.g. posterior draws)."""

    def get_default(self, spec: ParamSpec, default_ds: xr.Dataset) -> list[np.ndarray]:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            spec (ParamSpec): The parameter whose default value is needed.
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return [default_ds[p].values for p in spec.root_params]

    def write(
        self,
        spec: ParamSpec,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: list[np.ndarray],
        fixed_indices: dict[str, list[int]] | None=None,
    ) -> None:
        """Write a scaled value into the working dataset.

        Args:
            spec (ParamSpec): The parameter being written.
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Scaled value from the sampler.
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        Raises:
            ValueError: incorrect number of values given
        """
        if len(value) != len(spec.root_params):
            raise ValueError(
                f"Parameter '{spec.name}': expected {len(spec.root_params)} "
                f"arrays (one per root_param) but got {len(value)}."
            )
        for var_name, arr_value in zip(spec.root_params, value):
            ds[var_name].values = arr_value


# ----------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------

PARAM_TYPE_REGISTRY: dict[str, ParamType] = {
    "default": DefaultParamType(),
    "sliced": SlicedParamType(),
    "scale_from_root": ScaleFromRootParamType(),
    "multi_param": MultiParamType(),
}

# ----------------------------------------------------------------------------------------
# Shared write helpers
# ----------------------------------------------------------------------------------------


def _as_scalar(value: float | np.ndarray, name: str) -> float:
    """Return value as a Python float; raise if it is a non-scalar array.

    Args:
        value (float | np.ndarray): input value
        name (str): parameter name

    Raises:
        ValueError: incorrect input shape (non-scalar array)

    Returns:
        float: output float
    """
    arr = np.asarray(value)
    if arr.ndim > 0 and arr.size != 1:
        raise ValueError(
            f"Parameter '{name}': expected a scalar value but got an "
            f"array of shape {arr.shape}. Pass a scalar or expand the spec."
        )
    return float(arr)


def _broadcast_to_array(
    arr: np.ndarray,
    value: float | np.ndarray,
    fixed: list[int],
    name: str,
) -> np.ndarray:
    """Write value into arr at all non-fixed positions.
    
    Scalar value: broadcast to every non-fixed position.
    Array value: must match arr shape; fixed positions are skipped.

    Args:
        arr (np.ndarray): default array from dataset
        value (float | np.ndarray): input value
        fixed (list[int]): list of indices to keep at default
        name (str): parameter name

    Raises:
        ValueError: incorrect shape

    Returns:
        np.ndarray: output array
    """

    value_arr = np.asarray(value)

    if value_arr.ndim == 0:
        for i in range(len(arr)):
            if i not in fixed:
                arr[i] = float(value_arr)
    else:
        if value_arr.shape != arr.shape:
            raise ValueError(
                f"Parameter '{name}': value shape {value_arr.shape} does not "
                f"match target array shape {arr.shape}."
            )
        for i in range(len(arr)):
            if i not in fixed:
                arr[i] = value_arr[i]

    return arr
