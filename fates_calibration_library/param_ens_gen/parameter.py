"""
Parameter - classes for parameter logic
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional
import pandas as pd
import numpy as np
import xarray as xr

from .param_spec import ParamSpec
from .sampler import Sampler, SampleContext


class DimIndex(NamedTuple):
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
    sampler : Sampler
        Class for parameter sampling
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
        default_ds: xr.Dataset,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ):
        self.spec = ParamSpec.from_row(row)
        self.sampler = Sampler.from_row_and_sheet(row, pft_sheet, posterior_config)
        self.active_index: Optional[DimIndex] = None

        # store only sizes so we don't  hold a reference to the full dataset.
        # this is used by the n_indices property without ordering constraints.
        self._dim_sizes: dict[str, int] = dict(default_ds.sizes)

        self._validate_spec()
        self._validate_dataset(default_ds)

    @classmethod
    def from_row(
        cls,
        row: pd.Series,
        default_ds: xr.Dataset,
        pft_sheet: pd.DataFrame | None = None,
        posterior_config: dict | None = None,
    ) -> Parameter:
        """Construct the correct Parameter subclass from a spreadsheet row.

        Args:
            row (pd.Series): A row from the 'main' sheet DataFrame.
            default_ds (xr.Dataset): The default parameter dataset.
            pft_sheet (pd.DataFrame | None, optional): Optional per-PFT bounds sheet.
                Defaults to None.
            posterior_config (dict | None, optional): Optional posterior sampling
                configuration. Defaults to None.

        Raises:
            ValueError: If param_type is not registered.

        Returns:
            Parameter: An instance of the appropriate Parameter subclass.
        """
        param_type = str(row.get("param_type", "")).strip()
        subclass = cls._registry.get(param_type)
        if subclass is None:
            raise ValueError(
                f"Unknown param_type '{param_type}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass(row, default_ds, pft_sheet, posterior_config)
    
    @property
    def free_dim(self)-> str | None:
        """The single free dimension for this parameter, or None for scalars."""
        if self.spec.slice_dim is None:
            free_dims = self.spec.dims
        else:
            free_dims = [d for d in self.spec.dims if d != self.spec.slice_dim]
        return free_dims[0] if free_dims else None
    

    @property
    def n_indices(self) -> int:
        """Number of positions along the free dimension (1 for scalars)."""
        return self._dim_sizes.get(self.free_dim, 1) if self.free_dim else 1

    def _validate_spec(self) -> None:
        """Validate type-specific required fields on self.spec.

        Called from __init__ after self.spec is set. The base implementation
        is a no-op; subclasses override to assert the fields they require.
        This is intentionally not abstract — types with no extra required
        fields (e.g. DefaultParameter) need not override.
        """

    def _validate_dataset(self, default_ds: xr.Dataset) -> None:
        """Check that all variables this parameter touches exist in default_ds
        with the correct dimensions.

        For sliced parameters, spec.dims includes the slice dimension because
        it reflects the full variable shape in the dataset. The slice is an
        access pattern, not a dataset shape change.

        Args:
            default_ds (xr.Dataset): The default parameter dataset.

        Raises:
            ValueError: If a variable is missing or has unexpected dimensions.
        """
        for var in self._variables_to_validate():
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

    @abstractmethod
    def _variables_to_validate(self) -> list[str]:
        """Return the variable names in the dataset that this parameter touches.

        Each subclass must implement this explicitly so that _validate_dataset
        checks exactly the right variables. Abstract rather than a shared
        default to prevent new subclasses from silently inheriting incorrect
        behaviour.

        Returns:
            List of variable name strings.
        """

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

    def sample(
        self,
        normalized_value: float,
        default_ds: xr.Dataset,
    ) -> float | np.ndarray:
        """Sample a parameter given an input normalized value

        Builds a SampleContext from the dataset and fixed_indices, then
        delegates to self.sampler. Subclasses override _build_context if
        they need different context behaviour (e.g. JointParameter passes
        mask=None).

        Args:
            normalized_value (float): normalized value [0-1] used to sample
            default_ds (xr.Dataset): default parameter dataset. used for validating
                dimensions and indices
            fixed_indices (dict[str, list[int]]): 0-based indices to hold at default.

        Returns:
            float: Sampled parameter value.
        """
        default_value = self.get_default(default_ds)
        context = self._build_context(default_value)
        return self.sampler.sample(normalized_value, context)
    
    def _build_context(
        self,
        default_value: float | np.ndarray | list[np.ndarray],
    ) -> SampleContext:
        """Build a SampleContext for this parameter.
 
        The default implementation populates all fields. Subclasses override
        this when their sampler needs different context (e.g. JointParameter
        passes mask=None since its posterior sampler handles masking internally).
 
        Args:
            default_value: Default value(s) for this parameter.
            fixed_indices: Mapping of dimension name to fixed 0-based indices.
 
        Returns:
            SampleContext populated for this parameter.
        """
        return SampleContext(
            default_value=default_value,
            array_index=self.active_index.index if self.active_index is not None else None,
            n_indices=self.n_indices,
        )

# ----------------------------------------------------------------------------------------
# Concrete Parameter classes
# ----------------------------------------------------------------------------------------


class DefaultParameter(Parameter, param_type="default"):
    """Standard parameter: written directly to ds[name]."""

    def _variables_to_validate(self) -> list[str]:
        return [self.spec.name]

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
        return default_ds[self.spec.name].values

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        arr = ds[self.spec.name].values.copy()

        if self.active_index is not None:
            arr[self.active_index.index] = _as_scalar(value, self.spec.name)
        else:
            fixed = (
                (fixed_indices or {}).get(self.free_dim, []) if self.free_dim else []
            )
            arr = _broadcast_to_array(arr, value, fixed, self.spec.name)

        ds[self.spec.name].values = arr


class SlicedParameter(Parameter, param_type="sliced"):
    """Parameter that targets one slice of a dimension."""

    def _validate_spec(self) -> None:
        """Require slice_dim, slice_index, and base_params to all be set."""
        missing = []
        if self.spec.slice_dim is None:
            missing.append("slice_dim")
        if self.spec.slice_index is None:
            missing.append("slice_index")
        if not self.spec.base_params:
            missing.append("base_params")
        if missing:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'sliced' but the "
                f"following required fields are not set: {missing}."
            )

    def _variables_to_validate(self) -> list[str]:
        return self.spec.base_params

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
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
            fixed = (
                (fixed_indices or {}).get(self.free_dim, []) if self.free_dim else []
            )
            slice_arr = arr[tuple(idx)].copy()
            slice_arr = _broadcast_to_array(
                slice_arr, value, fixed, self.spec.base_params[0]
            )
            arr[tuple(idx)] = slice_arr

        ds[self.spec.base_params[0]].values = arr


class ScaleFromRootParameter(Parameter, param_type="scale_from_root"):
    """Parameter whose value is root + delta."""

    def _validate_spec(self) -> None:
        """Require root_param and base_params to both be set."""
        missing = []
        if self.spec.root_param is None:
            missing.append("root_param")
        if not self.spec.base_params:
            missing.append("base_params")
        if missing:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'scale_from_root' "
                f"but the following required fields are not set: {missing}."
            )

    def _variables_to_validate(self) -> list[str]:
        """Include both base_params and root_param in dataset validation."""
        variables = list(self.spec.base_params)
        if self.spec.root_param not in variables and self.spec.root_param is not None:
            variables.append(self.spec.root_param)
        return variables

    def get_default(self, default_ds: xr.Dataset) -> np.ndarray:
        return default_ds[self.spec.base_params[0]].values

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
        root_arr = ds[self.spec.root_param].values.copy()  # already written
        arr = ds[self.spec.base_params[0]].values.copy()
        delta = value

        if self.active_index is not None:
            i = self.active_index.index
            arr[i] = root_arr[i] + delta
        else:
            fixed = (
                (fixed_indices or {}).get(self.free_dim, []) if self.free_dim else []
            )

            if arr.ndim == 0 or not self.free_dim:
                arr = root_arr + delta
            else:
                default_arr = default_ds[self.spec.base_params[0]].values
                arr = root_arr + delta
                if fixed:
                    arr[fixed] = default_arr[fixed]

        ds[self.spec.base_params[0]].values = arr


class JointParameter(Parameter, param_type="joint"):
    """Parameter which stands for multiple connected parameters (e.g. posterior draws)."""

    def _validate_spec(self) -> None:
        """Require base_params to be non-empty."""
        if not self.spec.base_params:
            raise ValueError(
                f"Parameter '{self.spec.name}' has param_type 'joint' but "
                "base_params is not set."
            )

    def _variables_to_validate(self) -> list[str]:
        return self.spec.base_params

    def get_default(self, default_ds: xr.Dataset) -> list[np.ndarray]:
        return [default_ds[p].values for p in self.spec.base_params]

    def set_value(
        self,
        ds: xr.Dataset,
        default_ds: xr.Dataset,
        value: float | np.ndarray,
        fixed_indices: dict[str, list[int]] | None = None,
    ) -> None:
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
                fixed = (
                    (fixed_indices or {}).get(self.free_dim, [])
                    if self.free_dim
                    else []
                )
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
    if value_arr.ndim > 0 and value_arr.shape != arr.shape:
        raise ValueError(
            f"Parameter '{name}': value shape {value_arr.shape} does not "
            f"match target array shape {arr.shape}."
        )
    
    result = arr.copy()
    free = [i for i in range(len(result)) if i not in fixed] if arr.ndim > 0 else None
    
    if free is None:
        result = float(value_arr)
    elif value_arr.ndim == 0:
        result[free] = float(value_arr)
    else:
        result[free] = value_arr[free]

    return result


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

