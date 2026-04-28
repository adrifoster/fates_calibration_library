"""Tests for param_gen.bounds: Bound types and ParamBounds."""

import numpy as np
import pandas as pd
import pytest
 
from fates_calibration_library.param_gen.bounds import (
    Bound,
    FixedBound,
    NullBound,
    ParamBounds,
    PercentBound,
    PFTBound,
)

# ===========================================================================
# Bound.parse
# ===========================================================================
 
 
def test_parse_returns_fixed_bound_for_number():
    """Bound.parse returns a FixedBound for a plain numeric string.
 
    Args:
        None
    """
    bound = Bound.parse("0.5", bound_side="min")
    assert isinstance(bound, FixedBound)
    assert bound.value == 0.5
 
 
def test_parse_returns_fixed_bound_for_float_input():
    """Bound.parse returns a FixedBound for a float value.
 
    Args:
        None
    """
    bound = Bound.parse(0.9, bound_side="max")
    assert isinstance(bound, FixedBound)
    assert bound.value == 0.9
 
 
def test_parse_returns_percent_bound_for_percent_string():
    """Bound.parse returns a PercentBound for a percent string.
 
    Args:
        None
    """
    bound = Bound.parse("50percent", bound_side="min")
    assert isinstance(bound, PercentBound)
    assert bound.percent == 50.0
    assert bound.bound_side == "min"
 
 
def test_parse_returns_null_bound_for_posterior():
    """Bound.parse returns a NullBound for 'posterior'.
 
    Args:
        None
    """
    bound = Bound.parse("posterior", bound_side="min")
    assert isinstance(bound, NullBound)
 
 
def test_parse_raises_for_pft():
    """Bound.parse raises ValueError for 'pft' — must use PFTBound.from_sheet().
 
    Args:
        None
    """
    with pytest.raises(ValueError, match="PFTBound.from_sheet"):
        Bound.parse("pft", bound_side="min")
 
 
def test_parse_raises_for_empty_cell():
    """Bound.parse raises ValueError for None input.
 
    Args:
        None
    """
    with pytest.raises(ValueError, match="Bound cell is empty"):
        Bound.parse(None, bound_side="min")
 
 
def test_parse_raises_for_nan():
    """Bound.parse raises ValueError for NaN input.
 
    Args:
        None
    """
    with pytest.raises(ValueError, match="Bound cell is empty"):
        Bound.parse(float("nan"), bound_side="min")
 
 
def test_parse_raises_for_invalid_bound_side():
    """Bound.parse raises ValueError for invalid bound_side.
 
    Args:
        None
    """
    with pytest.raises(ValueError, match="bound_side must be"):
        Bound.parse("0.5", bound_side="middle")
        
# ===========================================================================
# FixedBound
# ===========================================================================
 
 
def test_fixed_bound_resolve_returns_value():
    """FixedBound.resolve returns its fixed value regardless of default_value.
 
    Args:
        None
    """
    bound = FixedBound(value=0.5)
    assert bound.resolve() == 0.5
    assert bound.resolve(default_value=99.0) == 0.5
 
 
def test_fixed_bound_resolve_ignores_array_default():
    """FixedBound.resolve ignores a numpy array default_value.
 
    Args:
        None
    """
    bound = FixedBound(value=1.0)
    assert bound.resolve(default_value=np.array([1.0, 2.0, 3.0])) == 1.0
 
 
# ===========================================================================
# PercentBound
# ===========================================================================
 
 
def test_percent_bound_min_scalar():
    """PercentBound resolves min correctly for a scalar default.
 
    Args:
        None
    """
    bound = PercentBound(percent=50.0, bound_side="min")
    result = bound.resolve(default_value=1.0)
    assert result == pytest.approx(0.5)
 
 
def test_percent_bound_max_scalar():
    """PercentBound resolves max correctly for a scalar default.
 
    Args:
        None
    """
    bound = PercentBound(percent=50.0, bound_side="max")
    result = bound.resolve(default_value=1.0)
    assert result == pytest.approx(1.5)
 
 
def test_percent_bound_min_array():
    """PercentBound resolves min correctly for an array default.
 
    Args:
        None
    """
    bound = PercentBound(percent=50.0, bound_side="min")
    default = np.array([1.0, 2.0, 4.0])
    result = bound.resolve(default_value=default)
    np.testing.assert_allclose(result, [0.5, 1.0, 2.0])
 
 
def test_percent_bound_max_array():
    """PercentBound resolves max correctly for an array default.
 
    Args:
        None
    """
    bound = PercentBound(percent=25.0, bound_side="max")
    default = np.array([1.0, 2.0, 4.0])
    result = bound.resolve(default_value=default)
    np.testing.assert_allclose(result, [1.25, 2.5, 5.0])
 
 
def test_percent_bound_uses_abs_for_negative_default():
    """PercentBound uses absolute value so negative defaults shrink toward zero for min.
 
    Args:
        None
    """
    bound = PercentBound(percent=50.0, bound_side="min")
    result = bound.resolve(default_value=-100.0)
    assert result == pytest.approx(-150.0)
 
 
def test_percent_bound_raises_without_default():
    """PercentBound.resolve raises ValueError when default_value is None.
 
    Args:
        None
    """
    bound = PercentBound(percent=50.0, bound_side="min")
    with pytest.raises(ValueError, match="requires a default_value"):
        bound.resolve(default_value=None)
 
 
# ===========================================================================
# NullBound
# ===========================================================================
 
 
def test_null_bound_resolve_returns_none():
    """NullBound.resolve always returns None.
 
    Args:
        None
    """
    bound = NullBound(value=None)
    assert bound.resolve() is None
    assert bound.resolve(default_value=1.0) is None
 
 
# ===========================================================================
# PFTBound
# ===========================================================================
 
 
@pytest.fixture
def pft_bound_sheet() -> pd.DataFrame:
    """A minimal per-parameter PFT sheet with 3 PFTs.
 
    Returns:
        pd.DataFrame: PFT bounds sheet
    """
    return pd.DataFrame({
        "pft_index": [1, 2, 3],
        "pft_name": ["white_spruce", "black_spruce", "deciduous"],
        "param_min": [0.005, 0.004, 0.008],
        "param_max": [0.040, 0.035, 0.060],
    })
 
 
def test_pft_bound_from_sheet_min(pft_bound_sheet):
    """PFTBound.from_sheet correctly reads param_min column.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    bound = PFTBound.from_sheet(pft_bound_sheet, "param_min")
    np.testing.assert_allclose(bound.values, [0.005, 0.004, 0.008])
 
 
def test_pft_bound_from_sheet_max(pft_bound_sheet):
    """PFTBound.from_sheet correctly reads param_max column.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    bound = PFTBound.from_sheet(pft_bound_sheet, "param_max")
    np.testing.assert_allclose(bound.values, [0.040, 0.035, 0.060])
 
 
def test_pft_bound_resolve_returns_array(pft_bound_sheet):
    """PFTBound.resolve returns the full values array.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    bound = PFTBound.from_sheet(pft_bound_sheet, "param_min")
    result = bound.resolve()
    np.testing.assert_allclose(result, [0.005, 0.004, 0.008])
 
 
def test_pft_bound_resolve_ignores_default_value(pft_bound_sheet):
    """PFTBound.resolve ignores default_value — values are always fixed.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    bound = PFTBound.from_sheet(pft_bound_sheet, "param_min")
    result = bound.resolve(default_value=np.array([99.0, 99.0, 99.0]))
    np.testing.assert_allclose(result, [0.005, 0.004, 0.008])
 
 
def test_pft_bound_raises_for_percent_values(pft_bound_sheet):
    """PFTBound.from_sheet raises ValueError if a cell contains a percent string.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    sheet = pd.DataFrame({
        "pft_index": [1, 2, 3],
        "pft_name": ["white_spruce", "black_spruce", "deciduous"],
        "param_min": ["50percent", "0.004", "0.008"],  # string dtype from the start
        "param_max": [0.040, 0.035, 0.060],
    })
    with pytest.raises(ValueError, match="fixed numbers"):
        PFTBound.from_sheet(sheet, "param_min")
 
 
# ===========================================================================
# ParamBounds.resolve
# ===========================================================================
 
 
def test_param_bounds_resolve_fixed():
    """ParamBounds.resolve returns correct (min, max) for fixed bounds.
 
    Args:
        None
    """
    bounds = ParamBounds(
        min_bound=FixedBound(0.1),
        max_bound=FixedBound(0.9),
    )
    lo, hi = bounds.resolve()
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.9)
 
 
def test_param_bounds_resolve_percent():
    """ParamBounds.resolve returns correct (min, max) for percent bounds.
 
    Args:
        None
    """
    bounds = ParamBounds(
        min_bound=PercentBound(50.0, "min"),
        max_bound=PercentBound(50.0, "max"),
    )
    lo, hi = bounds.resolve(default_value=1.0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(1.5)
 
 
def test_param_bounds_resolve_mixed():
    """ParamBounds.resolve handles mixed fixed and percent bounds.
 
    Args:
        None
    """
    bounds = ParamBounds(
        min_bound=PercentBound(50.0, "min"),
        max_bound=FixedBound(2.0),
    )
    lo, hi = bounds.resolve(default_value=1.0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(2.0)
 
 
# ===========================================================================
# ParamBounds.from_row_and_sheet
# ===========================================================================
 
 
def test_from_row_and_sheet_fixed_bounds():
    """from_row_and_sheet constructs FixedBounds from numeric cells.
 
    Args:
        None
    """
    row = pd.Series({"parameter_name": "test_param", "param_min": "0.1", "param_max": "0.9"})
    bounds = ParamBounds.from_row_and_sheet(row)
    assert isinstance(bounds.min_bound, FixedBound)
    assert isinstance(bounds.max_bound, FixedBound)
 
 
def test_from_row_and_sheet_percent_bounds():
    """from_row_and_sheet constructs PercentBounds from percent strings.
 
    Args:
        None
    """
    row = pd.Series({"parameter_name": "test_param", "param_min": "50percent", "param_max": "50percent"})
    bounds = ParamBounds.from_row_and_sheet(row)
    assert isinstance(bounds.min_bound, PercentBound)
    assert isinstance(bounds.max_bound, PercentBound)
 
 
def test_from_row_and_sheet_pft_bounds(pft_bound_sheet):
    """from_row_and_sheet constructs PFTBounds when both min and max are 'pft'.
 
    Args:
        pft_bound_sheet (pd.DataFrame): fixture
    """
    row = pd.Series({"parameter_name": "test_param", "param_min": "pft", "param_max": "pft"})
    bounds = ParamBounds.from_row_and_sheet(row, pft_sheet=pft_bound_sheet)
    assert isinstance(bounds.min_bound, PFTBound)
    assert isinstance(bounds.max_bound, PFTBound)
 
 
def test_from_row_and_sheet_pft_raises_without_sheet():
    """from_row_and_sheet raises ValueError for 'pft' bounds without a pft_sheet.
 
    Args:
        None
    """
    row = pd.Series({"parameter_name": "test_param", "param_min": "pft", "param_max": "pft"})
    with pytest.raises(ValueError, match="no pft_sheet was supplied"):
        ParamBounds.from_row_and_sheet(row, pft_sheet=None)
 
 
def test_from_row_and_sheet_posterior_bounds():
    """from_row_and_sheet constructs NullBounds for posterior strategy params.
 
    Args:
        None
    """
    row = pd.Series({"parameter_name": "test_param", "param_min": "posterior", "param_max": "posterior"})
    bounds = ParamBounds.from_row_and_sheet(row)
    assert isinstance(bounds.min_bound, NullBound)
    assert isinstance(bounds.max_bound, NullBound)