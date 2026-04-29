"""
Parameter - classes for parameter logic
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np
import xarray as xr

from .param_spec import ParamSpec
from .bounds import ParamBounds


@dataclass
class DimIndex:
    """A pinned position in a single dimension.

    Used on expanded Parameter objects to record which dimension and index

    Attributes
    ----------
    dim : str
        The dimension name, e.g. 'fates_pft'.
    index : int
        The 0-based index along that dimension.
    """

    dim: str
    index: int

class Parameter(ABC):
    """Abstract base for parameter logic.

    Parameters
    ----------
    spec : ParamSpec
        All metadata for this parameter
    bounds : ParamBounds
        Unresolved min/max bounds. Call bounds.resolve(default_value) at
        sample time to get concrete values.
    active_index: DimIndex | None
        Set by the expansion step on expanded Parameters. Records which dimension
        and index this Parameter is responsible for. None on unexpanded Parameters.
    """

    _registry: dict[str, type[Parameter]] = {}

    def __init_subclass__(cls, param_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        Parameter._registry[param_type] = cls

    def __init__(
            self,
            row: pd.Series,
            pft_sheet: pd.DataFrame | None = None,
            default_ds: xr.Dataset | None = None
            ):
        self.spec = ParamSpec.from_row(row)
        self.bounds = ParamBounds.from_row_and_sheet(row, pft_sheet)
        self.active_index: Optional[DimIndex] = None
        if default_ds is not None:
            self.validate(default_ds)

    @classmethod
    def from_row(
        cls,
        row: pd.Series,
        pft_sheet: pd.DataFrame | None = None,
        default_ds: xr.Dataset | None = None,
    ) -> Parameter:
        """Construct the correct Parameter subclass from a spreadsheet row."""
        param_type = str(row.get("param_type", "")).strip()
        subclass = cls._registry.get(param_type)
        if subclass is None:
            raise ValueError(
                f"Unknown param_type '{param_type}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(row, pft_sheet, default_ds)
    
    def validate(self, default_ds: xr.Dataset) -> None:
        variables_to_check = self._variables_to_validate()
        
        for var in variables_to_check:
            if var not in default_ds:
                raise ValueError(
                    f"Parameter '{self.spec.name}': variable '{var}' not found "
                    f"in default dataset. Available variables: "
                    f"{sorted(default_ds.data_vars)}"
                )
            actual_dims = list(default_ds[var].dims)
            if actual_dims != self.spec.dims:
                raise ValueError(
                    f"Parameter '{self.spec.name}': variable '{var}' has dims "
                    f"{actual_dims} in default dataset but spec.dims is "
                    f"{self.spec.dims}. Dimensions must match exactly."
                )
        
    def _variables_to_validate(self) -> list[str]:
        """Returns parameter names this parameter touches.
        
        The default implementation covers DefaultParameter (spec.name) and any
        type that uses base_params. Subclasses override only if they need different
        logic (ScaleFromRootParameter also reads root_params)

        Returns:
            list[str]: list of actual parameters this parameter handle touches
        """
        if self.spec.base_params:
            return self.spec.base_params
        return [self.spec.name]

    @abstractmethod
    def get_default(
        self,
        default_ds: xr.Dataset,
    ) -> float | np.ndarray | list[np.ndarray]:
        """Extract the relevant default value(s) from a netCDF dataset.

        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """

    @abstractmethod
    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray | list[np.ndarray],
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """


# ----------------------------------------------------------------------------------------
# Concrete Parameter classes
# ----------------------------------------------------------------------------------------


class DefaultParameter(Parameter, param_type="default"):
    """Standard parameter: written directly to ds[name]."""

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a parameter dataset.

        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return default_ds[self.spec.name].values

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """

        arr = ds[self.spec.name].values.copy()

        if self.active_index is not None:
            arr[self.active_index.index] = _as_scalar(value, self.spec.name)
        else:
            free_dim = self.spec.free_dims[0] if self.spec.free_dims else None
            fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []
            arr = _broadcast_to_array(arr, value, fixed, self.spec.name)

        ds[self.spec.name].values = arr


class SlicedParameter(Parameter, param_type="sliced"):
    """Parameter that targets one slice of a dimension."""

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a parameter dataset.

        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return (
            default_ds[self.spec.base_params[0]]
            .isel({self.spec.slice_dim: self.spec.slice_index})
            .values
        )

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """

        arr = ds[self.spec.base_params[0]].values.copy()
        da_dims = list(ds[self.spec.base_params[0]].dims)
        slice_axis = da_dims.index(self.spec.slice_dim)

        # build index tuple pointing at the slice
        idx = [slice(None)] * arr.ndim
        idx[slice_axis] = self.spec.slice_index

        if self.active_index is not None:
            idx[da_dims.index(self.active_index.dim)] = self.active_index.index
            arr[tuple(idx)] = _as_scalar(value, self.spec.base_params[0])
        else:
            free_dim = self.spec.free_dims[0] if self.spec.free_dims else None
            fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []
            slice_arr = arr[tuple(idx)].copy()
            slice_arr = _broadcast_to_array(
                slice_arr, value, fixed, self.spec.base_params[0]
            )
            arr[tuple(idx)] = slice_arr

        ds[self.spec.base_params[0]].values = arr


class ScaleFromRootParameter(Parameter, param_type="scale_from_root"):
    """Parameter whose value is root + delta."""
    
    
    def _variables_to_validate(self) -> list[str]:
        """Include root_param in validation in addition to base_params."""
        variables = list(self.spec.base_params)
        if self.spec.root_param and self.spec.root_param not in variables:
            variables.append(self.spec.root_param)
        return variables

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
        """Extract the relevant default value(s) from a parameter dataset.

        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return default_ds[self.spec.base_params[0]].values

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """

        root_arr = ds[self.spec.root_param].values.copy()  # already written
        arr = ds[self.spec.base_params[0]].values.copy()
        delta = value

        if self.active_index is not None:
            i = self.active_index.index
            arr[i] = root_arr[i] + delta
        else:
            free_dim = self.spec.free_dims[0] if self.spec.free_dims else None
            fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []

            if arr.ndim == 0 or not self.spec.free_dims:
                arr = root_arr + delta
            else:
                default_arr = default_ds[self.spec.base_params[0]].values
                for i in range(len(arr)):
                    if i in fixed:
                        arr[i] = default_arr[i]
                    else:
                        arr[i] = root_arr[i] + delta

        ds[self.spec.base_params[0]].values = arr


class JointParameter(Parameter, param_type="joint"):
    """Calibration handle for multiple connected parameters (e.g. posterior draws)."""

    def get_default(self, default_ds: xr.Dataset) -> list[np.ndarray]:
        """Extract the relevant default value(s) from a parameter dataset.

        Args:
            default_ds (xr.Dataset): The default parameter dataset

        Returns:
            float | np.ndarray | list[np.ndarray]: default value
        """
        return [default_ds[p].values for p in self.spec.base_params]

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        """Write a value into the working dataset.

        Args:
            ds (xr.Dataset): Working copy of the parameter dataset. Modified in place.
            default_ds (xr.Dataset): Unchanging default dataset. Used to restore fixed positions.
            value (float | np.ndarray | list[np.ndarray]): Value to write
            fixed_indices (dict[str, list[int]] | None): Run-level mapping of dimension to
                0-based indices to hold at default. None means no indices are fixed
        """
        value_arr = np.asarray(value)
        if len(value_arr) != len(self.spec.base_params):
            raise ValueError(
                f"Parameter '{self.spec.name}': expected {len(self.spec.base_params)} "
                f"arrays (one per base_param) but got {len(value_arr)}."
            )
        for parameter, val in zip(self.spec.base_params, value_arr):
            arr = ds[parameter].values.copy()
            if self.active_index is not None:
                arr[self.active_index.index] = _as_scalar(val, parameter)
            else:
                free_dim = self.spec.free_dims[0] if self.spec.free_dims else None
                fixed = (fixed_indices or {}).get(free_dim, []) if free_dim else []
                arr = _broadcast_to_array(arr, val, fixed, parameter)
            
            ds[parameter].values = arr


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
        if arr.ndim == 0:
            arr = float(value_arr)
        else:
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
