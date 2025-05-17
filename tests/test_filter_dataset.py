"""Tests for ilamb_functions.filter_dataset"""

import pytest
import xarray as xr
import numpy as np
from fates_calibration_library.ilamb_functions import filter_dataset


@pytest.fixture
def test_ds() -> xr.Dataset:
    """Creates a test dataset

    Returns:
        xr.Dataset: test datset
    """
    data = xr.Dataset(
        {"var1": (("time",), [1, 5, 10, 20])}, coords={"time": [0, 1, 2, 3]}
    )
    return data


def test_filter_min_only(test_ds):
    """Tests filtering only minimum

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    filtered = filter_dataset(test_ds.copy(), "var1", min_val=10)
    expected = [np.nan, np.nan, 10, 20]
    assert np.allclose(filtered["var1"].values, expected, equal_nan=True)


def test_filter_max_only(test_ds):
    """Tests filtering only maximum

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    filtered = filter_dataset(test_ds.copy(), "var1", max_val=5)
    expected = [1, 5, np.nan, np.nan]
    assert np.allclose(filtered["var1"].values, expected, equal_nan=True)


def test_filter_min_and_max(test_ds):
    """Tests filtering both minimum and maximum

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    filtered = filter_dataset(test_ds.copy(), "var1", min_val=5, max_val=10)
    expected = [np.nan, 5, 10, np.nan]
    assert np.allclose(filtered["var1"].values, expected, equal_nan=True)


def test_filter_none(test_ds):
    """Tests filtering when inputs are both None

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    filtered = filter_dataset(test_ds.copy(), "var1")
    assert np.array_equal(filtered["var1"].values, test_ds["var1"].values)


def test_filter_var_not_found(test_ds):
    """Tests that function catches that variable doesn't exist on dataset

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    with pytest.raises(KeyError, match="Variable 'var2' not found"):
        filter_dataset(test_ds.copy(), "var2", min_val=5)


def test_filter_all_filtered_out(test_ds):
    """Tests filtering when when all data would be masked out

    Args:
        test_ds (xr.Dataset): input test dataset
    """
    filtered = filter_dataset(test_ds.copy(), "var1", min_val=100)
    assert np.all(np.isnan(filtered["var1"].values))
