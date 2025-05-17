"""Tests for ilamb_functions.read_ilamb_data"""

from unittest.mock import patch
import pytest
import xarray as xr
from fates_calibration_library.ilamb_functions import read_ilamb_data


@pytest.fixture
def attributes() -> dict:
    """Creates a test attributes dict

    Returns:
        dict: attributes dict
    """
    return {
        "sub_dir": "some_subdir",
        "model": "some_model",
        "filename": "data.nc",
        "in_var": "gpp",
        "min_val": 0,
        "max_val": 100,
    }


def test_read_ilamb_data_success(attributes):
    """Tests that a basic read in works correctly

    Args:
        attributes (dict): test dictionary
    """
    mock_ds = xr.Dataset({"gpp": (("time",), [10, 20, 200])})

    with patch(
        "fates_calibration_library.ilamb_functions.xr.open_dataset",
        return_value=mock_ds,
    ) as mock_open:
        filtered_ds, path = read_ilamb_data("/base_dir", attributes)

    assert isinstance(filtered_ds, xr.Dataset)
    assert "gpp" in filtered_ds
    assert path == "/base_dir/some_subdir/some_model/data.nc"
    mock_open.assert_called_once_with(path)


def test_read_ilamb_data_missing_key():
    """Tests that function catches that a key is missing from the input
        dictionary

    Args:
        attributes (dict): test dictionary
    """
    incomplete_attrs = {
        "model": "some_model",  # missing sub_dir, filename, in_var
    }
    with pytest.raises(ValueError, match="Missing keys in attributes_dict:"):
        read_ilamb_data("/base_dir", incomplete_attrs)


def test_read_ilamb_data_filters_applied(attributes):
    """Tests that read_ilamb_data appropriately filters data

    Args:
        attributes (dict): test dictionary
    """
    # input with one value out of bounds
    raw_ds = xr.Dataset({"gpp": (("time",), [-5, 10, 200])})

    with patch(
        "fates_calibration_library.ilamb_functions.xr.open_dataset", return_value=raw_ds
    ):
        filtered_ds, _ = read_ilamb_data("/base_dir", attributes)

    assert filtered_ds["gpp"].values[0] != -5  # filtered out
    assert filtered_ds["gpp"].values[-1] != 200  # filtered out
