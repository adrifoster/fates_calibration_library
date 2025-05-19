"""Tests for analysis_functions"""

import xarray as xr
import numpy as np
import pytest
from fates_calibration_library.analysis_functions import (
    cyclic_month_difference,
    adjust_lon,
    calculate_zonal_mean,
    calculate_month_of_max,
    calculate_annual_mean,
    calculate_monthly_mean,
    area_mean,
    fix_infl_months,
    compute_infl
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
    expected = conversion * weighted_sum

    result = calculate_zonal_mean(da, land_area, conversion_factor=conversion)
    xr.testing.assert_allclose(result, expected)


def test_month_of_max_basic():
    """Tests basic functionality of calculate_month_of_max with known data"""
    data = xr.DataArray([1, 3, 2], coords={"month": [1, 2, 3]}, dims="month")
    result = calculate_month_of_max(data)
    assert result.item() == 2


def test_month_of_max_flat():
    """Tests basic functionality of calculate_month_of_max with flat data"""
    data = xr.DataArray([5, 5, 5], coords={"month": [1, 2, 3]}, dims="month")
    result = calculate_month_of_max(data)
    assert np.isnan(result.item())


def test_month_of_max_nan():
    """Tests basic functionality of calculate_month_of_max with nan data"""
    data = xr.DataArray(
        [np.nan, np.nan, np.nan], coords={"month": [1, 2, 3]}, dims="month"
    )
    result = calculate_month_of_max(data)
    assert np.isnan(result.item())


def test_calculate_month_of_max_invalid_dim():
    """Tests calculate_month_of_max catches that month_dim not on dataset"""
    da = xr.DataArray(np.random.rand(3, 4), dims=["lat", "lon"])
    with pytest.raises(ValueError, match="Dimension 'month' not found"):
        calculate_month_of_max(da)


def test_annual_mean_basic():
    """Basic test of calculate_annual_mean"""
    time = xr.cftime_range(str(2000), periods=12, freq="MS")
    data = xr.DataArray(np.ones(12), coords=[time], dims="time")
    data["time"] = time

    result = calculate_annual_mean(data)
    assert np.isclose(result.values, 1.0)


def test_annual_mean_with_conversion():
    """Tests calculate_annual_mean with a conversion factor"""
    time = xr.cftime_range(str(2000), periods=12, freq="MS")
    da = xr.DataArray(np.full(12, 2.0), coords=[time], dims="time")

    result = calculate_annual_mean(da, conversion_factor=2.0, new_units="g/m2/s")
    assert np.isclose(
        result.values, 2.0 * (da["time.daysinmonth"] * da).groupby("time.year").sum()
    )
    assert result.attrs["units"] == "g/m2/s"


def test_annual_mean_with_missing_data():
    """Test calculate_annual_mean with missing datda"""
    time = xr.cftime_range(str(2002), periods=12, freq="MS")
    values = np.ones(12)
    values[5] = np.nan
    da = xr.DataArray(values, coords=[time], dims="time")

    result = calculate_annual_mean(da)
    assert np.isclose(
        result.values,
        np.nansum(values * time.days_in_month)
        / np.nansum(time.days_in_month[~np.isnan(values)]),
    )


def test_annual_mean_missing_time_dim():
    """Tests calculate_annual_mean errors with no time dimension"""
    da = xr.DataArray([1, 2, 3], dims="x")
    with pytest.raises(ValueError, match="Input must have a 'time' dimension"):
        calculate_annual_mean(da)


def test_monthly_mean_typical_data():
    """Tests basic usage of calculate_monthly_mean"""
    data = np.arange(24).astype(float)
    time = xr.cftime_range("2000-01-01", periods=len(data), freq="MS")
    da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="test_var")
    result = calculate_monthly_mean(da)
    assert result.size == 12
    assert not result.isnull().any()


def test_monthly_mean_all_nan():
    """Tests calculate_monthly_mean with all NaN data"""
    data = np.full(24, np.nan)
    time = xr.cftime_range("2000-01-01", periods=len(data), freq="MS")
    da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="test_var")
    result = calculate_monthly_mean(da)
    assert result.isnull().all()


def test_monthly_mean_some_nan():
    """Tests calculate monthly mean with some NaNs"""
    data = np.array([np.nan if i % 2 == 0 else i for i in range(24)])
    time = xr.cftime_range("2000-01-01", periods=len(data), freq="MS")
    da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="test_var")
    result = calculate_monthly_mean(da)
    assert (result.notnull().sum() > 0).item()


def test_monthly_mean_constant_data():
    """Tests calculate_monthly_mean with constant dadta"""
    data = np.ones(24)
    time = xr.cftime_range("2000-01-01", periods=len(data), freq="MS")
    da = xr.DataArray(data, coords={"time": time}, dims=["time"], name="test_var")
    result = calculate_monthly_mean(da)
    np.testing.assert_allclose(result, 1.0)


def test_mothly_mean_missing_time_raises():
    """Tests that calculate_monthly_mean properly checks for time dimension."""
    da = xr.DataArray([1, 2, 3], dims="x")
    with pytest.raises(ValueError, match="Input must have a 'time' dimension."):
        calculate_monthly_mean(da)

def test_area_mean():
    """Test basic usage of area_mean where cf=None
    """
    
    data = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="example_var"
    )
    land_area = xr.DataArray(
        [[1.0, 1.0], [1.0, 1.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]}
    )

    # expected area-weighted mean with equal land area: mean of all values
    expected = (1 + 2 + 3 + 4) / 4

    result = area_mean(data, None, land_area)

    assert np.isclose(result.item(), expected)
    
def test_area_mean_with_nans():
    """Test usage of area_mean where cf=None and NaNs are in input data array
    """
    
    da = xr.DataArray(
        [[1.0, np.nan], [3.0, 4.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="example_var"
    )
    land_area = xr.DataArray(
        [[1.0, 1.0], [1.0, 1.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]},
    )

    # should ignore NaN in data array and use remaining valid cells
    valid_values = [1.0, 3.0, 4.0]
    expected = np.mean(valid_values)

    result = area_mean(da, cf=None, land_area=land_area)
    assert np.isclose(result.item(), expected)

def test_area_mean_with_explicit_cf():
    """Test use of area_mean with cf=0.5
    """
    da = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="example_var"
    )
    land_area = xr.DataArray(
        [[2.0, 2.0], [2.0, 4.0]],
        dims=["lat", "lon"],
        coords={"lat": [0, 1], "lon": [0, 1]},
    )

    # manually compute area-weighted mean
    cf = 0.5
    weighted_sum = (da * land_area).sum().item()
    expected = cf * weighted_sum

    result = area_mean(da, cf=cf, land_area=land_area)
    assert np.isclose(result.item(), expected)

def test_fix_infl_months_three_values():
    input_months = np.array([3, 7, 11])
    result = fix_infl_months(input_months)
    np.testing.assert_array_equal(result, np.array([3, 7, 11]))

def test_fix_infl_months_two_values_add_december():
    input_months = np.array([3, 6])
    result = fix_infl_months(input_months)
    np.testing.assert_array_equal(result, np.array([3, 6, 12]))

def test_fix_infl_months_two_values_add_january():
    input_months = np.array([7, 10])
    result = fix_infl_months(input_months)
    np.testing.assert_array_equal(result, np.array([1, 7, 10]))

def test_fix_infl_months_one_value():
    input_months = np.array([6])
    result = fix_infl_months(input_months)
    np.testing.assert_array_equal(result, np.array([1, 6, 12]))

def test_fix_infl_months_invalid_input():
    input_months = np.array([])
    with pytest.raises(ValueError, match="inflection_months must contain 1 to 3 elements"):
        fix_infl_months(input_months)

def test_single_peak():
    months = np.arange(1, 13)
    values = [1, 2, 4, 6, 9, 7, 5, 3, 2, 1, 1, 1]
    da = xr.DataArray(values, coords={"month": months}, dims="month")
    infl = compute_infl(da)
    assert set(infl) == set([12, 6, 1])