"""Tests for param_gen.strategy: Strategy enum."""
 
import pytest
 
from fates_calibration_library.param_ens_gen.strategy import Strategy
 
 
# ===========================================================================
# Strategy.parse
# ===========================================================================
 
 
def test_parse_uniform():
    """Strategy.parse returns UNIFORM for 'uniform'."""
    assert Strategy.parse("uniform") is Strategy.UNIFORM
 
 
def test_parse_posterior():
    """Strategy.parse returns POSTERIOR for 'posterior'."""
    assert Strategy.parse("posterior") is Strategy.POSTERIOR
 
 
def test_parse_is_case_insensitive():
    """Strategy.parse accepts uppercase and mixed-case input."""
    assert Strategy.parse("UNIFORM") is Strategy.UNIFORM
    assert Strategy.parse("Posterior") is Strategy.POSTERIOR
 
 
def test_parse_strips_whitespace():
    """Strategy.parse strips leading and trailing whitespace."""
    assert Strategy.parse("  uniform  ") is Strategy.UNIFORM
    assert Strategy.parse("  posterior  ") is Strategy.POSTERIOR
 
 
def test_parse_raises_for_invalid_string():
    """Strategy.parse raises ValueError for an unrecognised strategy string."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        Strategy.parse("bad_strategy")
 
 
def test_parse_raises_for_empty_string():
    """Strategy.parse raises ValueError for an empty string."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        Strategy.parse("")
 
 
def test_parse_error_message_lists_valid_options():
    """Strategy.parse error message includes valid strategy options."""
    with pytest.raises(ValueError, match="uniform"):
        Strategy.parse("bad_strategy")
 
 
# ===========================================================================
# Strategy.requires_bounds
# ===========================================================================
 
 
def test_uniform_requires_bounds():
    """UNIFORM requires bounds."""
    assert Strategy.UNIFORM.requires_bounds() is True
 
 
def test_posterior_does_not_require_bounds():
    """POSTERIOR does not require bounds."""
    assert Strategy.POSTERIOR.requires_bounds() is False
 
 
# ===========================================================================
# Strategy.requires_posterior
# ===========================================================================
 
 
def test_posterior_requires_posterior():
    """POSTERIOR requires a posterior source."""
    assert Strategy.POSTERIOR.requires_posterior() is True
 
 
def test_uniform_does_not_require_posterior():
    """UNIFORM does not require a posterior source."""
    assert Strategy.UNIFORM.requires_posterior() is False
 
 
# ===========================================================================
# Inverse relationship
# ===========================================================================
 
 
def test_requires_bounds_and_requires_posterior_are_inverses():
    """requires_bounds() and requires_posterior() are inverses for all current members.
 
    This test exists to catch any future Strategy variant that breaks the
    assumption. If a new strategy needs neither or both, this will fail
    with a clear message rather than causing a silent logic error downstream.
    """
    for member in Strategy:
        assert member.requires_bounds() != member.requires_posterior(), (
            f"Strategy.{member.name}: requires_bounds() and requires_posterior() "
            "returned the same value. If this is intentional for a new strategy, "
            "update this test and review all callsites that assume they are inverses."
        )