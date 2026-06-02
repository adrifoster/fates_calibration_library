"""Tests for param_ens_gen.param_spec: ParamSpec and its parsing helpers."""


import pytest

from fates_calibration_library.param_ens_gen.param_spec import ParamSpec

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
    assert spec.dims == ["fates_pft"]
    assert spec.slice_dim is None
    assert spec.slice_index is None
    assert spec.root_param is None
    assert not spec.base_params


def test_from_row_scalar(scalar_row):
    """from_row correctly constructs a scalar parameter."""
    spec = ParamSpec.from_row(scalar_row)
    assert not spec.dims


def test_from_row_sliced(sliced_row):
    """from_row correctly constructs a sliced parameter."""
    spec = ParamSpec.from_row(sliced_row)
    assert spec.param_type == "sliced"
    assert spec.slice_dim == "fates_leafage_class"
    assert spec.slice_index == 0
    assert spec.base_params == ["fates_leaf_vcmax25top"]


def test_from_row_scale_from_root(scale_from_root_row):
    """from_row correctly constructs a scale_from_root parameter."""
    spec = ParamSpec.from_row(scale_from_root_row)
    assert spec.param_type == "scale_from_root"
    assert spec.root_param == "fates_nonhydro_smpso"
    assert spec.base_params == ["fates_nonhydro_smpsc"]


def test_from_row_joint_param(joint_param_row):
    """from_row correctly constructs a joint_param parameter."""
    spec = ParamSpec.from_row(joint_param_row)
    assert spec.param_type == "joint"
    assert spec.base_params == [
        "fates_leafn_vert_scaler_coeff1",
        "fates_leafn_vert_scaler_coeff2",
    ]


def test_from_row_metadata_fields(default_row):
    """from_row correctly reads metadata fields (category, long_name, units)."""
    spec = ParamSpec.from_row(default_row)
    assert (
        spec.long_name
        == "Specific Leaf Area (SLA) at top of canopy, projected area basis"
    )
    assert spec.category == "stomatal"
    assert spec.subcategory == "photosynthesis"
    assert spec.units == "m^2/gC"


def test_from_row_missing_coord_raises(default_row):
    """from_row raises ValueError when the coord cell is NaN (blank in spreadsheet).
 
    NaN coord is a data error — the spreadsheet author forgot to fill it in.
    An explicit '[]' should be used for scalar parameters instead.
    """
    row = default_row.copy()
    row["coord"] = float("nan")
    with pytest.raises(ValueError, match="coord cell is missing"):
        ParamSpec.from_row(row)
    
def test_from_row_explicit_empty_coord_is_scalar(default_row):
    """from_row produces an empty dims list when coord is explicitly '[]'."""
    row = default_row.copy()
    row["coord"] = "[]"
    spec = ParamSpec.from_row(row)
    assert spec.dims == []
    
def test_from_row_base_params_parsed_from_list_literal(joint_param_row):
    """from_row parses a Python list literal in the base_params cell."""
    spec = ParamSpec.from_row(joint_param_row)
    assert len(spec.base_params) == 2
    
def test_from_row_base_params_parsed_from_plain_string(scale_from_root_row):
    """from_row parses a plain (non-bracketed) string in the base_params cell."""
    spec = ParamSpec.from_row(scale_from_root_row)
    assert spec.base_params == ["fates_nonhydro_smpsc"]
 

# ===========================================================================
# ParamSpec.__post_init__: field-level invariants
#
# These invariants hold regardless of param_type: putting a slice field on a
# non-sliced type, or root_param on a non-scale_from_root type, is always
# wrong at the field level. The Parameter subclass validates type-specific
# required fields (e.g. SlicedParameter checks that slice_dim IS present).
# ===========================================================================

def test_slice_dim_on_non_sliced_raises(default_row):
    """ValueError is raised when slice_dim is set on a non-sliced param."""
    row = default_row.copy()
    row["slice_dim"] = "fates_leafage_class"
    with pytest.raises(ValueError, match="slice_dim set but param_type"):
        ParamSpec.from_row(row)
 
 
def test_slice_index_on_non_sliced_raises(default_row):
    """ValueError is raised when slice_index is set on a non-sliced param.
 
    Uses "0" (string) because the default_row Series is string-typed,
    matching how pandas reads numeric cells from Excel.
    """
    row = default_row.copy()
    row["slice_index"] = "0"
    with pytest.raises(ValueError, match="slice_index set but param_type"):
        ParamSpec.from_row(row)
 
 
def test_root_param_on_non_scale_raises(default_row):
    """ValueError is raised when root_param is set on a non-scale_from_root param."""
    row = default_row.copy()
    row["root_param"] = "fates_nonhydro_smpso"
    with pytest.raises(ValueError, match="root_param set but param_type"):
        ParamSpec.from_row(row)
