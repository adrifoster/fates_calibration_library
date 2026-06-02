"""Tests for param_ens_gen.distribution_stat.
 
Scope
-----
- DistributionStat.parse: all accepted formats, all error cases
- DistributionStat.from_row: fixed, percent, pft, and error cases
- FixedStat.resolve
- PercentStat.resolve: min, max, sd, scalar and array default values
- PFTStat.resolve and PFTStat.from_sheet
"""
 
import numpy as np
import pandas as pd
import pytest
 
from fates_calibration_library.param_ens_gen.distribution_stat import (
    DistributionStat,
    FixedStat,
    PercentStat,
    PFTStat,
)
 
 
# ===========================================================================
# DistributionStat.parse: successful construction
# ===========================================================================
 
 
def test_parse_plain_number_returns_fixed_stat():
    """parse() returns a FixedStat for a plain numeric string."""
    stat = DistributionStat.parse("0.9", stat_type="max")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(0.9)
 
 
def test_parse_negative_number_returns_fixed_stat():
    """parse() correctly handles negative numeric values."""
    stat = DistributionStat.parse("-50000", stat_type="min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(-50000.0)
 
 
def test_parse_integer_string_returns_fixed_stat():
    """parse() accepts an integer string."""
    stat = DistributionStat.parse("1", stat_type="min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(1.0)
 
 
def test_parse_percent_word_returns_percent_stat():
    """parse() returns a PercentStat for '50percent' syntax."""
    stat = DistributionStat.parse("50percent", stat_type="min")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)
 
 
def test_parse_percent_symbol_returns_percent_stat():
    """parse() returns a PercentStat for '50%' syntax."""
    stat = DistributionStat.parse("50%", stat_type="max")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)
 
 
def test_parse_percent_with_spaces_returns_percent_stat():
    """parse() returns a PercentStat for '50 percent' syntax."""
    stat = DistributionStat.parse("50 percent", stat_type="min")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)
 
 
def test_parse_percent_stat_stores_stat_type():
    """parse() stores the stat_type on the returned PercentStat."""
    stat = DistributionStat.parse("30percent", stat_type="sd")
    assert stat.stat_type == "sd"
 
 
def test_parse_float_input_returns_fixed_stat():
    """parse() accepts a float (not just strings) as cell_value."""
    stat = DistributionStat.parse(0.5, stat_type="min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(0.5)
 
 
# ===========================================================================
# DistributionStat.parse: error cases
# ===========================================================================
 
 
def test_parse_unknown_stat_type_raises():
    """parse() raises ValueError for an unrecognised stat_type."""
    with pytest.raises(ValueError, match="Unknown stat_type"):
        DistributionStat.parse("0.5", stat_type="unknown")
 
 
def test_parse_none_raises():
    """parse() raises ValueError for None cell value."""
    with pytest.raises(ValueError, match="empty"):
        DistributionStat.parse(None, stat_type="min")
 
 
def test_parse_nan_raises():
    """parse() raises ValueError for NaN cell value."""
    with pytest.raises(ValueError, match="empty"):
        DistributionStat.parse(float("nan"), stat_type="min")
 
 
def test_parse_pft_string_raises():
    """parse() raises ValueError for 'pft' — must use from_row() instead."""
    with pytest.raises(ValueError, match="pft"):
        DistributionStat.parse("pft", stat_type="min")
 
 
def test_parse_percent_for_mean_raises():
    """parse() raises ValueError for percent syntax with stat_type='mean'."""
    with pytest.raises(ValueError, match="mean"):
        DistributionStat.parse("50percent", stat_type="mean")
 
 
def test_parse_zero_percent_raises():
    """parse() raises ValueError for 0 percent (would make min == max == default)."""
    with pytest.raises(ValueError, match="0"):
        DistributionStat.parse("0percent", stat_type="min")


def test_parse_negative_percent_raises():
    """parse() raises ValueError for negative percentages."""
    with pytest.raises(ValueError, match="-50"):
        DistributionStat.parse("-50percent", stat_type="min")
 
 
def test_parse_non_numeric_string_raises():
    """parse() raises ValueError for an unrecognisable string."""
    with pytest.raises(ValueError, match="Could not parse"):
        DistributionStat.parse("not_a_number", stat_type="min")
 
 
def test_parse_malformed_percent_raises():
    """parse() raises ValueError when percent prefix is not a number."""
    with pytest.raises(ValueError, match="Could not parse percent"):
        DistributionStat.parse("abcpercent", stat_type="min")
 
 
# ===========================================================================
# DistributionStat.from_row
# ===========================================================================
 
 
def test_from_row_fixed_min(default_row):
    """from_row() returns a FixedStat for a plain numeric param_min."""
    stat = DistributionStat.from_row(default_row, "min")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(0.005)
 
 
def test_from_row_fixed_max(default_row):
    """from_row() returns a FixedStat for a plain numeric param_max."""
    stat = DistributionStat.from_row(default_row, "max")
    assert isinstance(stat, FixedStat)
    assert stat.value == pytest.approx(0.05)
 
 
def test_from_row_percent(percent_row):
    """from_row() returns a PercentStat for a percent param_min."""
    stat = DistributionStat.from_row(percent_row, "min")
    assert isinstance(stat, PercentStat)
    assert stat.percent == pytest.approx(50.0)
 
 
def test_from_row_pft(pft_row, pft_sheet):
    """from_row() returns a PFTStat when param_min is 'pft'."""
    stat = DistributionStat.from_row(pft_row, "min", pft_sheet=pft_sheet)
    assert isinstance(stat, PFTStat)
 
 
def test_from_row_pft_without_sheet_raises(pft_row):
    """from_row() raises ValueError when param is 'pft' but no sheet is supplied."""
    with pytest.raises(ValueError, match="pft_sheet"):
        DistributionStat.from_row(pft_row, "min", pft_sheet=None)
 
 
# ===========================================================================
# FixedStat.resolve
# ===========================================================================
 
 
def test_fixed_stat_resolve_returns_value():
    """FixedStat.resolve() returns its stored value regardless of default_value."""
    stat = FixedStat(value=0.9)
    assert stat.resolve() == pytest.approx(0.9)
 
 
def test_fixed_stat_resolve_ignores_default_value():
    """FixedStat.resolve() ignores any default_value passed to it."""
    stat = FixedStat(value=0.9)
    assert stat.resolve(default_value=np.array([1.0, 2.0, 3.0])) == pytest.approx(0.9)
 
 
# ===========================================================================
# PercentStat.resolve
# ===========================================================================
 
 
def test_percent_stat_resolve_min_scalar():
    """PercentStat.resolve() for min subtracts the delta from the default."""
    stat = PercentStat(percent=50.0, stat_type="min")
    result = stat.resolve(default_value=1.0)
    assert result == pytest.approx(0.5)
 
 
def test_percent_stat_resolve_max_scalar():
    """PercentStat.resolve() for max adds the delta to the default."""
    stat = PercentStat(percent=50.0, stat_type="max")
    result = stat.resolve(default_value=1.0)
    assert result == pytest.approx(1.5)
 
 
def test_percent_stat_resolve_sd_scalar():
    """PercentStat.resolve() for sd adds the delta to the default."""
    stat = PercentStat(percent=25.0, stat_type="sd")
    result = stat.resolve(default_value=2.0)
    assert result == pytest.approx(2.5)
 
 
def test_percent_stat_resolve_min_array():
    """PercentStat.resolve() broadcasts correctly over a numpy array."""
    stat = PercentStat(percent=50.0, stat_type="min")
    default = np.array([1.0, 2.0, 4.0])
    result = stat.resolve(default_value=default)
    np.testing.assert_allclose(result, [0.5, 1.0, 2.0])
 
 
def test_percent_stat_resolve_uses_abs_for_negative_default():
    """PercentStat.resolve() uses abs(default) so the delta is always positive."""
    stat = PercentStat(percent=50.0, stat_type="min")
    # min of -100 should be -100 - 50 = -150, not -100 + 50 = -50
    result = stat.resolve(default_value=-100.0)
    assert result == pytest.approx(-150.0)
 
 
def test_percent_stat_resolve_none_default_raises():
    """PercentStat.resolve() raises ValueError when default_value is None."""
    stat = PercentStat(percent=50.0, stat_type="min")
    with pytest.raises(ValueError, match="default_value"):
        stat.resolve(default_value=None)
 
 
# ===========================================================================
# PFTStat.from_sheet and resolve
# ===========================================================================
 
 
def test_pft_stat_from_sheet_returns_correct_values(pft_sheet):
    """PFTStat.from_sheet() loads the correct min values from the sheet."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    np.testing.assert_allclose(stat.values, [0.005, 0.004, 0.008])
 
 
def test_pft_stat_from_sheet_max_returns_correct_values(pft_sheet):
    """PFTStat.from_sheet() loads the correct max values from the sheet."""
    stat = PFTStat.from_sheet(pft_sheet, "param_max")
    np.testing.assert_allclose(stat.values, [0.040, 0.035, 0.060])
 
 
def test_pft_stat_from_sheet_values_are_float(pft_sheet):
    """PFTStat.from_sheet() stores values as float dtype."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    assert stat.values.dtype == float
 
 
def test_pft_stat_from_sheet_non_numeric_raises():
    """PFTStat.from_sheet() raises ValueError for non-numeric cell values."""
    bad_sheet = pd.DataFrame({
        "pft_index": [1, 2, 3],
        "pft_name": ["white_spruce", "black_spruce", "deciduous"],
        "param_min": ["50percent", 0.004, 0.008],
        "param_max": [0.040, 0.035, 0.060],
    })
    with pytest.raises(ValueError, match="fixed numbers"):
        PFTStat.from_sheet(bad_sheet, "param_min")
 
 
def test_pft_stat_resolve_returns_values(pft_sheet):
    """PFTStat.resolve() returns the stored values array."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    result = stat.resolve()
    np.testing.assert_allclose(result, stat.values)
 
 
def test_pft_stat_resolve_ignores_default_value(pft_sheet):
    """PFTStat.resolve() ignores any default_value passed to it."""
    stat = PFTStat.from_sheet(pft_sheet, "param_min")
    result = stat.resolve(default_value=np.array([99.0, 99.0, 99.0]))
    np.testing.assert_allclose(result, stat.values)