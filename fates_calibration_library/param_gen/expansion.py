"""
expand() — fans out ParamSpec objects with expand_by_index=True into one
spec per active index along their first free dimension.

Specs with expand_by_index=False pass through unchanged, but their
fixed_indices are recorded so the writer knows which positions to skip.

Index conventions
------------------
All incides are 0-based, matching numpy/xarray conventions.

active_indices vs. fixed_indices
---------------------------------
- fixed_indices  : indices always held at default value; never expanded into specs
- active_indices : indices to vary (expanded if expand_by_index=True; else uniform)
- neither        : treated as active (default behavior)
- both           : raises ValueError

The two dicts do not need to be exhaustive - any index in neither is treated as active
"""

from __future__ import annotations

import copy
from typing import Optional

import xarray as xr

from fates_calibration_library.param_gen.param_spec import DimIndex, ParamSpec


def expand(
    specs: list[ParamSpec],
    default_ds: xr.Dataset,
    active_indices: Optional[dict[str, list[int]]] = None,
    fixed_indices: Optional[dict[str, list[int]]] = None,
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
        fixed_indices (Optional[dict[str, list[int]]], optional): Mapping of dimension
            name to 0-based indices to hold at default. These are never expanded into
            specs. If a dimension is absent, no indices are fixed for that dimension.

    Returns:
        list[ParamSpec]: Expanded spec list. Unexpanded specs are the same objects as in
            the input list. Expanded specs are shallow copies with expand_by_index
            set to False and active_index set to a DimIndex.

    Raises:
        ValueError
            If any index appears in both active_indices and fixed_indices.
        ValueError
            If active_indices or fixed_indices reference unknown dimensions
            or out-of-range indices.
        ValueError
            If a spec with expand_by_index=True has no free_dims.
    """
    full_index_map = _build_full_index_map(default_ds)
    fixed = _validate_and_normalize(
        fixed_indices or {}, full_index_map, "fixed_indices"
    )
    active = _resolve_active(active_indices, fixed, full_index_map)
    _check_overlap(active, fixed)

    result = []
    for spec in specs:
        if not spec.expand_by_index:
            clone = copy.copy(spec)
            clone.fixed_indices = fixed
            result.append(clone)
        else:
            result.extend(_expand_spec(spec, active, fixed, full_index_map))

    return result


def _build_full_index_map(default_ds: xr.Dataset) -> dict[str, list[int]]:
    """Build a map of all dimension names to all valid 0-based indices.

    Args:
        default_ds (xr.Dataset): input default parameter dataset

    Returns:
        dict[str, list[int]]: output dictionary mapping
    """

    return {dim: list(range(default_ds.sizes[dim])) for dim in default_ds.dims}


def _validate_and_normalize(
    indices: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
    label: str,
) -> dict[str, list[int]]:
    for dim, idxs in indices.items():
        if dim not in full_index_map:
            raise ValueError(
                f"{label} contains dimension '{dim}' which does not exist "
                f"in default_ds. Available dimensions: {sorted(full_index_map)}"
            )
        valid = full_index_map[dim]
        invalid = [i for i in idxs if i not in valid]
        if invalid:
            raise ValueError(
                f"{label}['{dim}'] contains out-of-range indices {invalid}. "
                f"Valid range for '{dim}' is 0–{len(valid) - 1}."
            )
    return indices


def _resolve_active(
    active_indices: Optional[dict[str, list[int]]],
    fixed: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
) -> dict[str, list[int]]:
    """Merge active_indices with full_index_map, validating as we go.

    Args:
        active_indices (Optional[dict[str, list[int]]]): optional input dictionary of
            active indices
        full_index_map (dict[str, list[int]]): a full index mapping from the
            default dataset

    Returns:
        dict[str, list[int]]: dict that has an entry for every dimension in full_index_map,
        using active_indices values where provided and full indices otherwise.
    """

    if active_indices is not None:
        _validate_and_normalize(active_indices, full_index_map, "active_indices")

    resolved = {}
    for dim, all_idxs in full_index_map.items():
        if active_indices is not None and dim in active_indices:
            resolved[dim] = active_indices[dim]
        else:
            fixed_for_dim = fixed.get(dim, [])
            resolved[dim] = [i for i in all_idxs if i not in fixed_for_dim]

    return resolved


def _check_overlap(
    active: dict[str, list[int]],
    fixed: dict[str, list[int]],
):
    for dim in set(active) & set(fixed):
        overlap = set(active[dim]) & set(fixed[dim])
        if overlap:
            raise ValueError(
                f"Dimension '{dim}' has indices {sorted(overlap)} in both "
                "active_indices and fixed_indices. Each index must be in "
                "one or the other, not both."
            )


def _expand_spec(
    spec: ParamSpec,
    active: dict[str, list[int]],
    fixed: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
) -> list[ParamSpec]:
    if not spec.free_dims:
        raise ValueError(
            f"Parameter '{spec.name}' has expand_by_index=True but no "
            "free_dims to expand over. Set expand_by_index=False or add "
            "a dimension to this parameter."
        )

    # expand over the first free dimension
    expand_dim = spec.free_dims[0]

    if expand_dim not in full_index_map:
        # dimension exists on the spec but not in default_ds — shouldn't
        # happen if the netCDF file and spreadsheet are consistent
        raise ValueError(
            f"Parameter '{spec.name}': free dimension '{expand_dim}' not "
            f"found in default_ds. Available dimensions: {sorted(full_index_map)}"
        )

    indices = active.get(expand_dim, full_index_map[expand_dim])
    expanded = []
    for idx in indices:
        clone = copy.copy(spec)
        clone.expand_by_index = False
        clone.active_index = DimIndex(dim=expand_dim, index=idx)
        clone.fixed_indices = fixed
        expanded.append(clone)

    return expanded
