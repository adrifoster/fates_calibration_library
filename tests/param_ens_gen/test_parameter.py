"""Tests for param_gen.parameter — Parameter subclasses and helpers."""

import numpy as np
import pytest

from fates_calibration_library.param_ens_gen.bounds import FixedBound, PFTBound, NullBound, PercentBound
from fates_calibration_library.param_ens_gen.parameter import (
    DimIndex,
    DefaultParameter,
    JointParameter,
    Parameter,
    ScaleFromRootParameter,
    SlicedParameter,
    _as_scalar,
    _broadcast_to_array,
)

# ===========================================================================
# DimIndex
# ===========================================================================


def test_dimindex_stores_dim_and_index():
    """DimIndex correctly stores dim name and 0-based index.

    Args:
        None
    """
    di = DimIndex(dim="fates_pft", index=2)
    assert di.dim == "fates_pft"
    assert di.index == 2


# ===========================================================================
# Parameter registry / from_row
# ===========================================================================


def test_from_row_returns_default_parameter(default_row):
    """from_row returns a DefaultParameter for param_type='default'.

    Args:
        default_row (pd.Series): fixture
    """
    param = Parameter.from_row(default_row)
    assert isinstance(param, DefaultParameter)


def test_from_row_returns_sliced_parameter(sliced_row):
    """from_row returns a SlicedParameter for param_type='sliced'.

    Args:
        sliced_row (pd.Series): fixture
    """
    param = Parameter.from_row(sliced_row)
    assert isinstance(param, SlicedParameter)


def test_from_row_returns_scale_from_root_parameter(scale_from_root_row):
    """from_row returns a ScaleFromRootParameter for param_type='scale_from_root'.

    Args:
        scale_from_root_row (pd.Series): fixture
    """
    param = Parameter.from_row(scale_from_root_row)
    assert isinstance(param, ScaleFromRootParameter)


def test_from_row_returns_joint_parameter(joint_param_row):
    """from_row returns a JointParameter for param_type='joint'.

    Args:
        joint_param_row (pd.Series): fixture
    """
    param = Parameter.from_row(joint_param_row)
    assert isinstance(param, JointParameter)


def test_from_row_raises_for_unknown_param_type(default_row):
    """from_row raises ValueError for an unregistered param_type.

    Args:
        default_row (pd.Series): fixture
    """
    default_row["param_type"] = "unknown_type"
    with pytest.raises(ValueError, match="Unknown param_type"):
        Parameter.from_row(default_row)


def test_active_index_is_none_on_construction(default_row):
    """active_index is None on a freshly constructed Parameter.

    Args:
        default_row (pd.Series): fixture
    """
    param = Parameter.from_row(default_row)
    assert param.active_index is None


# ===========================================================================
# Bounds gating
# ===========================================================================
 
 
def test_uniform_parameter_has_bounds(default_row):
    """A uniform strategy parameter has a ParamBounds instance."""
    param = Parameter.from_row(default_row)
    assert param.bounds is not None
    assert isinstance(param.bounds.min_bound, FixedBound)
    assert isinstance(param.bounds.max_bound, FixedBound)
 
 
def test_posterior_parameter_has_no_bounds(joint_param_row):
    """A posterior strategy parameter has bounds=None."""
    param = Parameter.from_row(joint_param_row)
    assert isinstance(param.bounds.min_bound, NullBound)
    assert isinstance(param.bounds.max_bound, NullBound)
    
def test_percent_has_percent_bounds(percent_row):
    """from_row correctly constructs a parameter with percent bounds.

    Args:
        percent_row (pd.Series): fixture
    """
    param = Parameter.from_row(percent_row)
    assert isinstance(param.bounds.min_bound, PercentBound)
    assert isinstance(param.bounds.max_bound, PercentBound)
    
def test_from_row_pft_bounds(pft_row, pft_sheet):
    """from_row correctly constructs a parameter with PFT-specific bounds.

    Args:
        pft_row (pd.Series): fixture
        pft_sheet (pd.DataFrame): fixture
    """
    param = Parameter.from_row(pft_row, pft_sheet=pft_sheet)
    assert isinstance(param.bounds.min_bound, PFTBound)
    assert isinstance(param.bounds.max_bound, PFTBound)

    
# ===========================================================================
# Parameter.validate
# ===========================================================================
 
 
def test_validate_passes_for_default_parameter(default_row, default_ds):
    """validate() does not raise for a correctly configured DefaultParameter."""
    param = Parameter.from_row(default_row)
    param.validate(default_ds)  # should not raise
 
 
def test_validate_passes_for_sliced_parameter(sliced_row, default_ds):
    """validate() does not raise for a correctly configured SlicedParameter."""
    param = Parameter.from_row(sliced_row)
    param.validate(default_ds)  # should not raise
 
 
def test_validate_passes_for_scale_from_root_parameter(scale_from_root_row, default_ds):
    """validate() does not raise for a correctly configured ScaleFromRootParameter."""
    param = Parameter.from_row(scale_from_root_row)
    param.validate(default_ds)  # should not raise
 
 
def test_validate_passes_for_joint_parameter(joint_param_row, default_ds):
    """validate() does not raise for a correctly configured JointParameter."""
    param = Parameter.from_row(joint_param_row)
    param.validate(default_ds)  # should not raise
 
 
def test_validate_raises_for_missing_variable(default_row, default_ds):
    """validate() raises ValueError when the variable is missing from the dataset."""
    default_row["parameter_name"] = "nonexistent_param"
    param = Parameter.from_row(default_row)
    with pytest.raises(ValueError, match="not found in default dataset"):
        param.validate(default_ds)
 
 
def test_validate_raises_for_wrong_dims(default_row, default_ds):
    """validate() raises ValueError when spec.dims does not match the dataset dims."""
    # Give the spec the wrong dims — variable exists but dims won't match
    default_row["coord"] = "['fates_leafage_class', 'fates_pft']"
    param = Parameter.from_row(default_row)
    with pytest.raises(ValueError, match="Dimensions must match exactly"):
        param.validate(default_ds)
 
 
def test_validate_raises_for_missing_base_param(sliced_row, default_ds):
    """validate() raises ValueError when a base_param variable is missing."""
    sliced_row["base_params"] = "['nonexistent_param']"
    param = Parameter.from_row(sliced_row)
    with pytest.raises(ValueError, match="not found in default dataset"):
        param.validate(default_ds)
 
 
def test_validate_raises_for_missing_root_param(scale_from_root_row, default_ds):
    """validate() raises ValueError when root_param is missing from the dataset."""
    scale_from_root_row["root_param"] = "nonexistent_root"
    param = Parameter.from_row(scale_from_root_row)
    with pytest.raises(ValueError, match="not found in default dataset"):
        param.validate(default_ds)
 
 
def test_validate_called_at_construction_when_ds_supplied(default_row, default_ds):
    """validate() is called immediately when default_ds is passed to from_row."""
    default_row["parameter_name"] = "fates_nonexistent_param"
    with pytest.raises(ValueError, match="not found in default dataset"):
        Parameter.from_row(default_row, default_ds=default_ds)
 


# ===========================================================================
# DefaultParameter
# ===========================================================================


def test_default_get_default_pft(default_row, default_ds):
    """DefaultParameter.get_default returns the full PFT array.

    Args:
        default_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(default_row)
    result = param.get_default(default_ds)
    np.testing.assert_allclose(result, [0.010, 0.020, 0.030])


def test_default_get_default_scalar(scalar_row, default_ds):
    """DefaultParameter.get_default returns a scalar for non-PFT parameters.

    Args:
        scalar_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scalar_row)
    result = param.get_default(default_ds)
    assert float(result) == pytest.approx(0.5)


def test_default_set_value_uniform_no_fixed(default_row, default_ds, working_ds):
    """DefaultParameter.set_value writes value to all PFT positions.

    Args:
        default_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(default_row)
    param.set_value(working_ds, default_ds, value=0.025)
    np.testing.assert_allclose(
        working_ds["fates_leaf_slatop"].values,
        [0.025, 0.025, 0.025],
    )


def test_default_set_value_with_fixed_indices(default_row, default_ds, working_ds):
    """DefaultParameter.set_value leaves fixed PFTs at their default values.

    Args:
        default_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(default_row)
    param.set_value(
        working_ds, default_ds, value=0.025,
        fixed_indices={"fates_pft": [2]},  # fix PFT index 2 (0-based)
    )
    assert working_ds["fates_leaf_slatop"].values[0] == pytest.approx(0.025)
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(0.025)
    assert working_ds["fates_leaf_slatop"].values[2] == pytest.approx(0.030)  # unchanged


def test_default_set_value_with_active_index(default_row, default_ds, working_ds):
    """DefaultParameter.set_value writes only to active_index position.

    Args:
        default_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(default_row)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_ds, default_ds, value=0.099)

    assert working_ds["fates_leaf_slatop"].values[0] == pytest.approx(0.010)  # unchanged
    assert working_ds["fates_leaf_slatop"].values[1] == pytest.approx(0.099)  # written
    assert working_ds["fates_leaf_slatop"].values[2] == pytest.approx(0.030)  # unchanged


def test_default_set_value_scalar_param(scalar_row, default_ds, working_ds):
    """DefaultParameter.set_value correctly writes a scalar parameter.

    Args:
        scalar_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scalar_row)
    param.set_value(working_ds, default_ds, value=0.7)
    assert float(working_ds["fates_canopy_closure_thresh"].values) == pytest.approx(0.7)


# ===========================================================================
# SlicedParameter
# ===========================================================================


def test_sliced_get_default(sliced_row, default_ds):
    """SlicedParameter.get_default returns the slice at slice_index.

    Args:
        sliced_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(sliced_row)
    result = param.get_default(default_ds)
    # slice_index=0: first leafage_class row → [50.0, 60.0, 70.0]
    np.testing.assert_allclose(result, [50.0, 60.0, 70.0])


def test_sliced_set_value_no_fixed(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value writes value across all PFTs at the slice.

    Args:
        sliced_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(sliced_row)
    param.set_value(working_ds, default_ds, value=80.0)

    arr = working_ds["fates_leaf_vcmax25top"].values
    # slice_index=0 (leafage_class dim) should be updated
    np.testing.assert_allclose(arr[0, :], [80.0, 80.0, 80.0])
    # other leafage_class rows unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])


def test_sliced_set_value_with_fixed_indices(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value leaves fixed PFTs at their default values.

    Args:
        sliced_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(sliced_row)
    param.set_value(
        working_ds, default_ds, value=80.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_ds["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # fixed — unchanged
    assert arr[0, 1] == pytest.approx(80.0)  # written
    assert arr[0, 2] == pytest.approx(80.0)  # written


def test_sliced_set_value_with_active_index(sliced_row, default_ds, working_ds):
    """SlicedParameter.set_value writes only to (slice_index, active_index) cell.

    Args:
        sliced_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(sliced_row)
    param.active_index = DimIndex(dim="fates_pft", index=1)
    param.set_value(working_ds, default_ds, value=99.0)

    arr = working_ds["fates_leaf_vcmax25top"].values
    assert arr[0, 0] == pytest.approx(50.0)  # unchanged
    assert arr[0, 1] == pytest.approx(99.0)  # written
    assert arr[0, 2] == pytest.approx(70.0)  # unchanged
    np.testing.assert_allclose(arr[1, :], [40.0, 50.0, 60.0])  # other slice unchanged


# ===========================================================================
# ScaleFromRootParameter
# ===========================================================================


def test_scale_from_root_get_default(scale_from_root_row, default_ds):
    """ScaleFromRootParameter.get_default returns root parameter values.

    Args:
        scale_from_root_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scale_from_root_row)
    result = param.get_default(default_ds)
    np.testing.assert_allclose(result, [-100000.0, -110000.0, -120000.0])


def test_scale_from_root_set_value_no_fixed(scale_from_root_row, default_ds, working_ds):
    """ScaleFromRootParameter.set_value adds delta to root values for all PFTs.

    Args:
        scale_from_root_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scale_from_root_row)
    delta = -10000.0
    param.set_value(working_ds, default_ds, value=delta)

    expected = np.array([-50000.0, -60000.0, -70000.0]) + delta
    np.testing.assert_allclose(
        working_ds["fates_nonhydro_smpsc"].values, expected
    )


def test_scale_from_root_set_value_with_fixed(scale_from_root_row, default_ds, working_ds):
    """ScaleFromRootParameter.set_value leaves fixed PFTs at their default values.

    Args:
        scale_from_root_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scale_from_root_row)
    param.set_value(
        working_ds, default_ds, value=-10000.0,
        fixed_indices={"fates_pft": [0]},
    )
    arr = working_ds["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-100000.0)   # fixed — default value
    assert arr[1] == pytest.approx(-70000.0)    # root + delta
    assert arr[2] == pytest.approx(-80000.0)    # root + delta


def test_scale_from_root_set_value_with_active_index(
    scale_from_root_row, default_ds, working_ds
):
    """ScaleFromRootParameter.set_value writes delta only at active_index.

    Args:
        scale_from_root_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(scale_from_root_row)
    param.active_index = DimIndex(dim="fates_pft", index=0)
    param.set_value(working_ds, default_ds, value=-5000.0)

    arr = working_ds["fates_nonhydro_smpsc"].values
    assert arr[0] == pytest.approx(-55000.0)    # root[0] + delta
    assert arr[1] == pytest.approx(-110000.0)   # unchanged
    assert arr[2] == pytest.approx(-120000.0)   # unchanged


# ===========================================================================
# JointParameter
# ===========================================================================


def test_joint_get_default(joint_param_row, default_ds):
    """JointParameter.get_default returns a list of arrays, one per base_param.

    Args:
        joint_param_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(joint_param_row)
    result = param.get_default(default_ds)
    assert isinstance(result, list)
    assert len(result) == 2
    np.testing.assert_allclose(result[0], [0.012, 0.015, 0.005])
    np.testing.assert_allclose(result[1], [2.1, 2.5, 2.6])


def test_joint_set_value(joint_param_row, default_ds, working_ds):
    """JointParameter.set_value writes each array to its corresponding variable.

    Args:
        joint_param_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(joint_param_row)
    new_values = [
        np.array([0.5, 0.6, 0.7]),
        np.array([5.0, 6.0, 7.0]),
    ]
    param.set_value(working_ds, default_ds, value=new_values)
    np.testing.assert_allclose(working_ds["fates_leafn_vert_scaler_coeff1"].values, [0.5, 0.6, 0.7])
    np.testing.assert_allclose(working_ds["fates_leafn_vert_scaler_coeff2"].values, [5.0, 6.0, 7.0])


def test_joint_set_value_wrong_length_raises(joint_param_row, default_ds, working_ds):
    """JointParameter.set_value raises ValueError if value length mismatches base_params.

    Args:
        joint_param_row (pd.Series): fixture
        default_ds (xr.Dataset): fixture
        working_ds (xr.Dataset): fixture
    """
    param = Parameter.from_row(joint_param_row)
    with pytest.raises(ValueError, match="expected 2 arrays"):
        param.set_value(working_ds, default_ds, value=[np.array([0.5, 0.6, 0.7])])


# ===========================================================================
# _as_scalar helper
# ===========================================================================


def test_as_scalar_from_float():
    """_as_scalar returns a float from a plain float input."""
    assert _as_scalar(0.5, "test_param") == pytest.approx(0.5)


def test_as_scalar_from_single_element_array():
    """_as_scalar returns a float from a single-element numpy array."""
    assert _as_scalar(np.array([0.5]), "test_param") == pytest.approx(0.5)


def test_as_scalar_raises_for_multi_element_array():
    """_as_scalar raises ValueError for a multi-element array."""
    with pytest.raises(ValueError, match="expected a scalar value"):
        _as_scalar(np.array([0.5, 0.6]), "test_param")


# ===========================================================================
# _broadcast_to_array helper
# ===========================================================================


def test_broadcast_scalar_to_all_positions():
    """_broadcast_to_array fills all positions with a scalar value."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _broadcast_to_array(arr, 9.9, fixed=[], name="test")
    np.testing.assert_allclose(result, [9.9, 9.9, 9.9])


def test_broadcast_scalar_skips_fixed_positions():
    """_broadcast_to_array leaves fixed positions unchanged."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _broadcast_to_array(arr, 9.9, fixed=[1], name="test")
    assert result[0] == pytest.approx(9.9)
    assert result[1] == pytest.approx(2.0)  # fixed — unchanged
    assert result[2] == pytest.approx(9.9)


def test_broadcast_array_value():
    """_broadcast_to_array writes an array value to all non-fixed positions."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _broadcast_to_array(arr, np.array([9.0, 8.0, 7.0]), fixed=[], name="test")
    np.testing.assert_allclose(result, [9.0, 8.0, 7.0])


def test_broadcast_array_value_with_fixed():
    """_broadcast_to_array skips fixed positions when writing an array value."""
    arr = np.array([1.0, 2.0, 3.0])
    result = _broadcast_to_array(
        arr, np.array([9.0, 8.0, 7.0]), fixed=[2], name="test"
    )
    assert result[0] == pytest.approx(9.0)
    assert result[1] == pytest.approx(8.0)
    assert result[2] == pytest.approx(3.0)  # fixed — unchanged


def test_broadcast_raises_for_shape_mismatch():
    """_broadcast_to_array raises ValueError when array shapes do not match."""
    arr = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        _broadcast_to_array(arr, np.array([9.0, 8.0]), fixed=[], name="test")


def test_broadcast_scalar_param():
    """_broadcast_to_array handles a 0-d (scalar) array correctly."""
    arr = np.float64(1.0)
    result = _broadcast_to_array(arr, 9.9, fixed=[], name="test")
    assert float(result) == pytest.approx(9.9)