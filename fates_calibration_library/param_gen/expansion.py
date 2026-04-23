"""
    expand() — fans out ParamSpec objects with expand_by_index=True into one
    spec per active index along their first free dimension.

    Specs with expand_by_index=False pass through unchanged.

    The expansion step is a preprocessing step that runs once before sampling.
    After expansion, every spec in the returned list has:
    - expand_by_index = False  (already expanded; prevents double-expansion)
    - active_index set to a DimIndex  (if it was expanded)
    - active_index = None             (if it was not expanded)

    Usage
    -----
    expanded = expand(specs, default_ds)

    # or with specific active indices (e.g. to fix PFT 4 at default):
    expanded = expand(specs, default_ds, active_indices={"fates_pft": [0, 1, 2]})

    Index convention
    ----------------
    Indices are 0-based throughout, matching numpy/xarray conventions.
    The FATES netCDF files use 1-based PFT indices in their metadata, but
    by the time values reach this module they have been read into numpy arrays
    where indexing is 0-based.
"""

from __future__ import annotations
 
import copy
from typing import Optional
 
import xarray as xr

from fates_calibration_library.param_spec import DimIndex, ParamSpec


def expand(
    specs: list[ParamSpec],
    default_ds: xr.Dataset,
    active_indices: Optional[dict[str, list[int]]] = None,
) -> list[ParamSpec]:
    """Expand specs with expand_by_index=True into one spec per active index.

    Args:
        specs (list[ParamSpec]): Specs as loaded from the spreadsheet. May contain a 
            mix of expand_by_index=True and expand_by_index=False.
        default_ds (xr.Dataset): Default FATES parameter dataset. Used to determine the 
            full set of valid indices for each dimension, and to validate active_indices.
        active_indices (Optional[dict[str, list[int]]], optional): Optional mapping of 
            dimension name to list of 0-based indices to expand over. 
            If a dimension is not in this dict, all indices for that dimension are used. 
            If None, all indices for all dimensions are used.
            Validated against default_ds — passing an out-of-range index raises a 
            ValueError.

    Returns:
        list[ParamSpec]: Expanded spec list. Unexpanded specs are the same objects as in 
            the input list. Expanded specs are shallow copies with expand_by_index
            set to False and active_index set to a DimIndex.
    """
    full_index_map = _build_full_index_map(default_ds)
    resolved = _resolve_active_indices(active_indices, full_index_map)
 
    result = []
    for spec in specs:
        if not spec.expand_by_index:
            result.append(spec)
            continue
        result.extend(_expand_spec(spec, resolved, full_index_map))
 
    return result
    

def _build_full_index_map(default_ds: xr.Dataset) -> dict[str, list[int]]:
    """Build a map of all dimension names to all valid 0-based indices.

    Args:
        default_ds (xr.Dataset): input default parameter dataset

    Returns:
        dict[str, list[int]]: output dictionary mapping
    """
    
    return {
        dim: list(range(default_ds.sizes[dim]))
        for dim in default_ds.dims
    }

def _resolve_active_indices(active_indices: Optional[dict[str, list[int]]],
                            full_index_map: dict[str, list[int]]) -> dict[str, list[int]]:
    """"Merge active_indices with full_index_map, validating as we go.

    Args:
        active_indices (Optional[dict[str, list[int]]]): optional input dictionary of 
            active indices
        full_index_map (dict[str, list[int]]): a full index mapping from the 
            default dataset

    Returns:
        dict[str, list[int]]: dict that has an entry for every dimension in full_index_map,
        using active_indices values where provided and full indices otherwise.
    """

    if active_indices is None:
            return full_index_map.copy()
    
    resolved = full_index_map.copy()
    
    for dim, indices in active_indices.items():
        if dim not in full_index_map:
            raise ValueError(
                f"active_indices contains dimension '{dim}' which does not "
                f"exist in default_ds. Available dimensions: {sorted(full_index_map)}"
            )
        valid = full_index_map[dim]
        invalid = [i for i in indices if i not in valid]
        if invalid:
            raise ValueError(
                f"active_indices['{dim}'] contains out-of-range indices {invalid}. "
                f"Valid range for '{dim}' is 0–{len(valid) - 1}."
            )
        resolved[dim] = indices
 
    return resolved

def _expand_spec(
    spec: ParamSpec,
    resolved: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
) -> list[ParamSpec]:
    """Return one expanded copy of spec per active index of free_dims[0]."""
    if not spec.free_dims:
        raise ValueError(
            f"Parameter '{spec.name}' has expand_by_index=True but no "
            "free_dims to expand over. Set expand_by_index=False or add "
            "a dimension to this parameter."
        )
 
    # expand over the first free dimension
    expand_dim = spec.free_dims[0]
 
    if expand_dim not in resolved:
        # dimension exists on the spec but not in default_ds — shouldn't
        # happen if the netCDF file and spreadsheet are consistent
        raise ValueError(
            f"Parameter '{spec.name}': free dimension '{expand_dim}' not "
            f"found in default_ds. Available dimensions: {sorted(full_index_map)}"
        )
 
    indices = resolved[expand_dim]
    expanded = []
    for idx in indices:
        clone = copy.copy(spec)
        clone.expand_by_index = False
        clone.active_index = DimIndex(dim=expand_dim, index=idx)
        expanded.append(clone)
 
    return expanded