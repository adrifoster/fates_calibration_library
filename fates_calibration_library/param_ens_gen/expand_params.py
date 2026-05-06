from __future__ import annotations
 
import copy
from typing import Optional
import xarray as xr
 
from .parameter import Parameter, DimIndex

def expand(
        params: list[Parameter],
        default_ds: xr.Dataset,
        fixed_indices: Optional[dict[str, list[int]]] = None,
    ) -> list[Parameter]:
        """Expand list of Parameter objects into one Parameter per active index.

        Args:
            fixed_indices (Optional[dict[str, list[int]]], optional): Mapping of dimension
                name to 0-based indices to hold at default. These are never expanded into
                specs. If None, no indices are fixed and all are expanded over.

        Returns:
            list[Parameter]: Expanded Parameter list. Unexpanded Parameters are returned
            unchanged. Expanded Parameters are shallow copies with active_index set to a
            DimIndex.

        Raises:
            ValueError
                If fixed_indices references unknown dimensions or out-of-range indices.
            ValueError
                If a spec with expand_by_index=True has no free_dims.
        """
        fixed = fixed_indices or {}
        full_index_map = _build_full_index_map(default_ds)
        _validate_fixed(fixed, full_index_map)

        result = []
        for param in params:
            result.extend(expand_param(param, fixed, full_index_map))
        return result

def expand_param(
    param: Parameter,
    fixed: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
) -> list[Parameter]:
    """Return one expanded copy of Paremeter per active index of free_dims[0].

    Args:
        param (Parameter): parameter to expand
        fixed (dict[str, list[int]]): mapping of dim to indices of indices to fix
        full_index_map (dict[str, list[int]]): full available dim: indes mapping

    Raises:
        ValueError: no free dims to expand over
        ValueError: dimension not fouond in default_ds

    Returns:
        list[Parameter]: expanded copy of Parameters
    """
    if not param.spec.free_dims:
        raise ValueError(
            f"Parameter '{param.spec.name}' has no " "free_dims to expand over."
        )

    # expand over the first free dimension
    expand_dim = param.spec.free_dims[0]

    if expand_dim not in full_index_map:
        # dimension exists on the spec but not in default_ds — shouldn't
        # happen if the netCDF file and spreadsheet are consistent
        raise ValueError(
            f"Parameter '{param.spec.name}': free dimension '{expand_dim}' not "
            f"found in default_ds. Available dimensions: {sorted(full_index_map)}"
        )

    fixed_for_dim = fixed.get(expand_dim, [])
    active = [i for i in full_index_map[expand_dim] if i not in fixed_for_dim]

    expanded = []
    for idx in active:
        # here clone.spec points to the same ParamSpec as original,
        # which is intentional
        clone = copy.copy(param)
        clone.active_index = DimIndex(dim=expand_dim, index=idx)
        expanded.append(clone)

    return expanded

def _validate_fixed(
    fixed: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
):
    """Raise if fixed_indices references unknown dims or out-of-range indices.

    Args:
        fixed (dict[str, list[int]]): input mapping of dim to indices of fixed indices
        full_index_map (dict[str, list[int]]): full available mapping of dim to indices

    Raises:
        ValueError: fixed_indices has dimension which does not exist in default_ds
        ValueError: out of range index
    """
    for dim, idxs in fixed.items():
        if dim not in full_index_map:
            raise ValueError(
                f"fixed_indices contains dimension '{dim}' which does not "
                f"exist in default_ds. Available dimensions: {sorted(full_index_map)}"
            )
        valid = full_index_map[dim]
        invalid = [i for i in idxs if i not in valid]
        if invalid:
            raise ValueError(
                f"fixed_indices['{dim}'] contains out-of-range indices {invalid}. "
                f"Valid range for '{dim}' is 0–{len(valid) - 1}."
            )

def _build_full_index_map(default_ds: xr.Dataset) -> dict[str, list[int]]:
    """Build a map of all dimension names to all valid 0-based indices.

    Args:
        default_ds (xr.Dataset): input default parameter dataset

    Returns:
        dict[str, list[int]]: output dictionary mapping
    """
    return {dim: list(range(default_ds.sizes[dim])) for dim in default_ds.dims}