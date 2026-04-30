"""Tests for param_gen.param_spec: ParamSpec and its parsing helpers."""

import pytest

from fates_calibration_library.param_gen.param_spec import (
    ParamSpec,
    _parse_dims,
    _parse_list,
    _parse_optional_int,
    _parse_optional_str,
)
from fates_calibration_library.param_gen.strategy import Strategy

# ===========================================================================
# ParamSpec.from_row: successful construction
# ===========================================================================


def test_from_row_default(default_row):
    """from_row correctly constructs a default parameter.

    Args:
        default_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(default_row)
    assert spec.name == "fates_leaf_slatop"
    assert spec.param_type == "default"
    assert spec.strategy == Strategy.UNIFORM
    assert spec.param_min == "0.005"
    assert spec.param_max == "0.05"
    assert spec.dims == ["fates_pft"]
    assert spec.slice_dim is None
    assert spec.slice_index is None
    assert spec.root_param is None
    assert not spec.base_params


def test_from_row_scalar(scalar_row):
    """from_row correctly constructs a scalar parameter.

    Args:
        scalar_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(scalar_row)
    assert not spec.dims
    assert not spec.free_dims


def test_from_row_sliced(sliced_row):
    """from_row correctly constructs a sliced parameter.

    Args:
        sliced_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(sliced_row)
    assert spec.param_type == "sliced"
    assert spec.slice_dim == "fates_leafage_class"
    assert spec.slice_index == 0
    assert spec.base_params == ["fates_leaf_vcmax25top"]


def test_from_row_scale_from_root(scale_from_root_row):
    """from_row correctly constructs a scale_from_root parameter.

    Args:
        scale_from_root_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(scale_from_root_row)
    assert spec.param_type == "scale_from_root"
    assert spec.root_param == "fates_nonhydro_smpso"
    assert spec.base_params == ["fates_nonhydro_smpsc"]


def test_from_row_joint_param(joint_param_row):
    """from_row correctly constructs a joint_param parameter.

    Args:
        joint_param_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(joint_param_row)
    assert spec.param_type == "joint"
    assert spec.strategy == Strategy.POSTERIOR
    assert spec.base_params == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]


def test_from_row_metadata_fields(default_row):
    """from_row correctly reads metadata fields (category, long_name, units).

    Args:
        default_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(default_row)
    assert (
        spec.long_name
        == "Specific Leaf Area (SLA) at top of canopy, projected area basis"
    )
    assert spec.category == "stomatal"
    assert spec.subcategory == "photosynthesis"
    assert spec.units == "m^2/gC"


# ===========================================================================
# ParamSpec.__post_init__: validation errors
# ===========================================================================


def test_invalid_strategy_raises(default_row):
    """from_row raises ValueError for an unrecognised strategy.

    Args:
        default_row (pd.Series): fixture
    """
    default_row["strategy"] = "bad_strategy"
    with pytest.raises(ValueError, match="Invalid strategy"):
        ParamSpec.from_row(default_row)


def test_missing_strategy_raises(default_row):
    """from_row raises ValueError when strategy cell is empty.

    Args:
        default_row (pd.Series): fixture
    """
    default_row["strategy"] = ""
    with pytest.raises(ValueError, match="Invalid strategy"):
        ParamSpec.from_row(default_row)


def test_sliced_missing_slice_dim_raises(sliced_row):
    """from_row raises ValueError for sliced param_type with no slice_dim.

    Args:
        sliced_row (pd.Series): fixture
    """
    sliced_row["slice_dim"] = None
    with pytest.raises(ValueError, match="slice_dim"):
        ParamSpec.from_row(sliced_row)


def test_sliced_missing_slice_index_raises(sliced_row):
    """from_row raises ValueError for sliced param_type with no slice_index.

    Args:
        sliced_row (pd.Series): fixture
    """
    sliced_row["slice_index"] = None
    with pytest.raises(ValueError, match="slice_dim"):
        ParamSpec.from_row(sliced_row)


def test_sliced_missing_base_params_raises(sliced_row):
    """from_row raises ValueError for sliced param_type with no base_params.

    Args:
        sliced_row (pd.Series): fixture
    """
    sliced_row["base_params"] = ""
    with pytest.raises(ValueError, match="slice_dim"):
        ParamSpec.from_row(sliced_row)


def test_slice_dim_on_non_sliced_raises(default_row):
    """from_row raises ValueError when slice_dim is set on a non-sliced param.

    Args:
        default_row (pd.Series): fixture
    """
    default_row["slice_dim"] = "fates_leafage_class"
    with pytest.raises(ValueError, match="slice_dim set but param_type"):
        ParamSpec.from_row(default_row)


def test_scale_from_root_missing_root_param_raises(scale_from_root_row):
    """from_row raises ValueError for scale_from_root with no root_param.

    Args:
        scale_from_root_row (pd.Series): fixture
    """
    scale_from_root_row["root_param"] = None
    with pytest.raises(ValueError, match="root_param"):
        ParamSpec.from_row(scale_from_root_row)


def test_root_param_on_non_scale_raises(default_row):
    """from_row raises ValueError when root_param is set on a non-scale param.

    Args:
        default_row (pd.Series): fixture
    """
    default_row["root_param"] = "fates_nonhydro_smpso"
    with pytest.raises(ValueError, match="root_param set but param_type"):
        ParamSpec.from_row(default_row)


def test_joint_param_empty_base_params_raises(joint_param_row):
    """from_row raises ValueError for joint param with no base_params.

    Args:
        joint_param_row (pd.Series): fixture
    """
    joint_param_row["base_params"] = ""
    with pytest.raises(ValueError, match="base_params"):
        ParamSpec.from_row(joint_param_row)


# ===========================================================================
# ParamSpec.free_dims property
# ===========================================================================


def test_free_dims_no_slice(default_row):
    """free_dims returns all dims when no slice_dim is set.

    Args:
        default_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(default_row)
    assert spec.free_dims == ["fates_pft"]


def test_free_dims_with_slice(sliced_row):
    """free_dims excludes slice_dim, leaving the remaining free dimensions.

    Args:
        sliced_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(sliced_row)
    assert "fates_leafage_class" not in spec.free_dims
    assert "fates_pft" in spec.free_dims


def test_free_dims_scalar(scalar_row):
    """free_dims returns empty list for scalar parameters.

    Args:
        scalar_row (pd.Series): fixture
    """
    spec = ParamSpec.from_row(scalar_row)
    assert spec.free_dims == []


# ===========================================================================
# _parse_dims
# ===========================================================================


def test_parse_dims_single_dim():
    """_parse_dims parses a single-element list string correctly."""
    assert _parse_dims("['fates_pft']") == ["fates_pft"]


def test_parse_dims_multi_dim():
    """_parse_dims parses a multi-element list string correctly."""
    assert _parse_dims("['fates_leafage_class', 'fates_pft']") == [
        "fates_leafage_class",
        "fates_pft",
    ]


def test_parse_dims_empty_string():
    """_parse_dims returns empty list for empty string."""
    assert _parse_dims("") == []


def test_parse_dims_none():
    """_parse_dims returns empty list for None input."""
    assert _parse_dims(None) == []


def test_parse_dims_nan():
    """_parse_dims returns empty list for NaN (as read from Excel)."""
    assert _parse_dims(float("nan")) == []


def test_parse_dims_plain_string_fallback():
    """_parse_dims falls back to stripping brackets for non-literal strings."""
    assert _parse_dims("fates_pft") == ["fates_pft"]


# ===========================================================================
# _parse_list
# ===========================================================================


def test_parse_list_python_literal():
    """_parse_list parses a Python list literal string."""
    assert _parse_list("['fates_allom_agb1', 'fates_allom_agb2']") == [
        "fates_allom_agb1",
        "fates_allom_agb2",
    ]


def test_parse_list_comma_separated():
    """_parse_list falls back to comma splitting for non-literal strings."""
    assert _parse_list("fates_allom_agb1, fates_allom_agb2") == [
        "fates_allom_agb1",
        "fates_allom_agb2",
    ]


def test_parse_list_single_value():
    """_parse_list returns a single-element list for a plain string."""
    assert _parse_list("fates_nonhydro_smpsc") == ["fates_nonhydro_smpsc"]


def test_parse_list_none():
    """_parse_list returns empty list for None."""
    assert _parse_list(None) == []


def test_parse_list_nan():
    """_parse_list returns empty list for NaN (as read from Excel)."""
    assert _parse_list(float("nan")) == []


def test_parse_list_empty_string():
    """_parse_list returns empty list for empty string."""
    assert _parse_list("") == []


# ===========================================================================
# _parse_optional_int
# ===========================================================================


def test_parse_optional_int_valid_string():
    """_parse_optional_int parses a valid integer string."""
    assert _parse_optional_int("0") == 0


def test_parse_optional_int_float_value():
    """_parse_optional_int parses a float value that represents a whole number."""
    assert _parse_optional_int(1.0) == 1


def test_parse_optional_int_none():
    """_parse_optional_int returns None for None input."""
    assert _parse_optional_int(None) is None


def test_parse_optional_int_nan():
    """_parse_optional_int returns None for NaN (as read from Excel)."""
    assert _parse_optional_int(float("nan")) is None


def test_parse_optional_int_non_numeric():
    """_parse_optional_int returns None for non-numeric string."""
    assert _parse_optional_int("not_a_number") is None


# ===========================================================================
# _parse_optional_str
# ===========================================================================


def test_parse_optional_str_valid():
    """_parse_optional_str returns stripped string for valid input."""
    assert _parse_optional_str("  fates_pft  ") == "fates_pft"


def test_parse_optional_str_none():
    """_parse_optional_str returns None for None input."""
    assert _parse_optional_str(None) is None


def test_parse_optional_str_nan():
    """_parse_optional_str returns None for NaN (as read from Excel)."""
    assert _parse_optional_str(float("nan")) is None


def test_parse_optional_str_empty():
    """_parse_optional_str returns None for empty string."""
    assert _parse_optional_str("") is None


def test_parse_optional_str_whitespace_only():
    """_parse_optional_str returns None for whitespace-only string."""
    assert _parse_optional_str("   ") is None
