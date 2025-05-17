"""Tests for analysis_functions"""

import xarray as xr
import numpy as np
import pytest
from fates_calibration_library.analysis_functions import (
    cyclic_month_difference,
    adjust_lon,
    calculate_zonal_mean,
)


def test_cyclic_month_difference_same_month():
    """Tests cyclic difference for two arrays with the same values is 0"""
    da = xr.DataArray([1, 6, 12])
    result = cyclic_month_difference(da, da)
    assert np.all(result == 0)


def test_cyclic_month_difference_forward_and_wrap():
    """Tests that the cyclic month difference can forward wrap"""
    da1 = xr.DataArray([1, 6, 11])
    da2 = xr.DataArray([2, 12, 1])
    result = cyclic_month_difference(da1, da2)
    expected = xr.DataArray([1, 6, 2])
    assert result.equals(expected)


def test_cyclic_month_difference_backward_and_wrap():
    """Tests that the cyclic month difference can backwards wrap"""
    da1 = xr.DataArray([12, 6, 3])
    da2 = xr.DataArray([1, 1, 1])
    result = cyclic_month_difference(da1, da2)
    expected = xr.DataArray([1, -5, -2])
    assert result.equals(expected)


def test_cyclic_month_difference_input_shape_mismatch():
    """Tests that the function can properly catch DataArray shape mis-matches"""
    da1 = xr.DataArray([1, 2, 3])
    da2 = xr.DataArray([1])
    with pytest.raises(ValueError, match="Input arrays must have the same shape."):
        cyclic_month_difference(da1, da2)


def test_adjust_lon_basic():
    """Tests a basic example of adjust_lon"""
    # example: 0, 90, 180, 270 should become 0, 90, 180, -90
    lon = [0, 90, 180, 270]
    ds = xr.Dataset({"data": (("lon",), [1, 2, 3, 4])}, coords={"lon": lon})
    result = adjust_lon(ds, "lon")
    expected_lon = [-90, 0, 90, 180]
    np.testing.assert_array_equal(result.lon.values, sorted(expected_lon))
    assert list(result.data.values) == [4, 1, 2, 3]  # data should match reordered lon


def test_adjust_lon_missing_var():
    """Tests that the function can catch the input lon name not exists"""
    ds = xr.Dataset({"temp": ("x", [1, 2, 3])})
    with pytest.raises(ValueError, match="not found in dataset."):
        adjust_lon(ds, "lon")


def test_adjust_lon_no_change_needed():
    """Tests that function still works properly when lon already in -180 to 180"""
    # all already in -180 to 180
    lon = [-180, -90, 0, 90, 180]
    ds = xr.Dataset({"var": (("lon",), range(5))}, coords={"lon": lon})
    out = adjust_lon(ds, "lon")
    np.testing.assert_array_equal(out.lon.values, lon)


def test_zonal_mean_basic():
    """Tests basic implementation of calculating zonal mean"""
    # create a 3x4 grid (lat x lon) with known values
    da = xr.DataArray([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]], dims=["lat", "lon"])

    # constant land area (weight=1)
    land_area = xr.ones_like(da)

    expected = da.mean(dim="lon")
    result = calculate_zonal_mean(da, land_area)
    xr.testing.assert_allclose(result, expected)


def test_zonal_mean_with_conversion():
    """Tests basic use with a conversion"""
    da = xr.DataArray([[10, 10], [20, 20]], dims=["lat", "lon"])
    land_area = xr.DataArray([[1, 1], [2, 2]], dims=["lat", "lon"])
    conversion = 0.5

    # expected zonal mean: weighted average, then scale
    weighted_sum = (da * land_area).sum(dim="lon")
    area_sum = land_area.sum(dim="lon")
    expected = conversion * (weighted_sum / area_sum)

    result = calculate_zonal_mean(da, land_area, conversion_factor=conversion)
    xr.testing.assert_allclose(result, expected)
