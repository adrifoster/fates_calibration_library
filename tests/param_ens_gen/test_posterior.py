"""Tests for param_ens_gen.posterior: PosteriorSource."""
 
import pytest
import numpy as np
import pandas as pd
 
from fates_calibration_library.param_ens_gen.posterior import PosteriorSource
 
 
# ===========================================================================
# PosteriorSource.__post_init__: validation
# ===========================================================================

def test_post_init_path_converted_to_path_object(posterior_file):
    """__post_init__ converts a string path to a Path object."""
    source = PosteriorSource(
        path=str(posterior_file),
        array_indices="all",
        parameters=["param_a", "param_b"],
    )
    assert hasattr(source.path, "exists")


def test_post_init_array_indices_all_valid(posterior_file):
    """__post_init__ accepts 'all' as array_indices."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["param_a"],
    )
    assert source.array_indices == "all"


def test_post_init_array_indices_list_of_ints_valid(posterior_file):
    """__post_init__ accepts a list of ints as array_indices."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices=[0, 1, 2],
        parameters=["param_a"],
    )
    assert source.array_indices == [0, 1, 2]
    

def test_post_init_array_indices_float_list_converted(posterior_file):
    """__post_init__ converts a list of floats to ints (e.g. from YAML parsing)."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices=[0.0, 1.0, 2.0],
        parameters=["param_a"],
    )
    assert source.array_indices == [0, 1, 2]
    assert all(isinstance(i, int) for i in source.array_indices)


def test_post_init_invalid_string_raises(posterior_file):
    """__post_init__ raises ValueError for an unrecognised string."""
    with pytest.raises(ValueError, match="'all'"):
        PosteriorSource(
            path=posterior_file,
            array_indices="some_other_string",
            parameters=["param_a"],
        )
        
def test_post_init_non_list_converts(posterior_file):
    """__post_init__ converts a scalar input for array_indices into a list."""
    source = PosteriorSource(
            path=posterior_file,
            array_indices=42,
            parameters=["param_a"],
        )
    assert source.array_indices == [42]
        
def test_post_init_non_numeric_list_raises(posterior_file):
    """__post_init__ raises ValueError when list entries cannot be converted to int."""
    with pytest.raises(ValueError, match="could not be converted"):
        PosteriorSource(
            path=posterior_file,
            array_indices=["a", "b"],
            parameters=["param_a"],
        )
        
# ===========================================================================
# PosteriorSource.is_broadcast property
# ===========================================================================
 
 
def test_is_broadcast_true_for_all(posterior_file):
    """is_broadcast returns True when array_indices is 'all'."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["param_a"],
    )
    assert source.is_broadcast is True
 
 
def test_is_broadcast_false_for_list(posterior_file):
    """is_broadcast returns False when array_indices is a list."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices=[0, 1],
        parameters=["param_a"],
    )
    assert source.is_broadcast is False
    
# ===========================================================================
# PosteriorSource._load: loading and validation
# ===========================================================================
 
 
def test_load_populates_draws(posterior_source):
    """_load populates _draws with the correct columns."""
    posterior_source._load()
    assert posterior_source._draws is not None
    assert list(posterior_source._draws.columns) == ["param_a", "param_b"]
 
 
def test_load_sorts_by_sort_param_index(posterior_source):
    """_load sorts draws by the column at sort_param_index."""
    posterior_source._load()
    col = posterior_source._draws["param_a"]
    assert list(col) == sorted(col)
 
 
def test_load_sorts_by_second_column_when_sort_param_index_is_1(posterior_file):
    """_load sorts by the correct column when sort_param_index=1."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["param_a", "param_b"],
        sort_param_index=1,
    )
    source._load()
    col = source._draws["param_b"]
    assert list(col) == sorted(col)
 
 
def test_load_missing_file_raises(tmp_path):
    """_load raises FileNotFoundError for a non-existent file."""
    source = PosteriorSource(
        path=tmp_path / "does_not_exist.txt",
        array_indices="all",
        parameters=["param_a"],
    )
    with pytest.raises(FileNotFoundError, match="cannot find input file"):
        source._load()
 
 
def test_load_missing_column_raises(posterior_file):
    """_load raises ValueError when a requested parameter is not in the file."""
    source = PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["param_a", "nonexistent_param"],
    )
    with pytest.raises(ValueError, match="nonexistent_param"):
        source._load()
 
 
# ===========================================================================
# PosteriorSource.draw_row: sampling
# ===========================================================================
 
 
def test_draw_row_triggers_lazy_load(posterior_source):
    """draw_row loads the file automatically on first call."""
    assert posterior_source._draws is None
    posterior_source.draw_row(0.5)
    assert posterior_source._draws is not None
 
 
def test_draw_row_returns_series(posterior_source):
    """draw_row returns a pd.Series indexed by parameter name."""
    row = posterior_source.draw_row(0.5)
    assert isinstance(row, pd.Series)
    assert "param_a" in row.index
    assert "param_b" in row.index
 
 
def test_draw_row_zero_returns_lowest(posterior_source):
    """draw_row with value=0.0 returns the row with the lowest sort column value."""
    posterior_source._load()
    min_val = posterior_source._draws["param_a"].iloc[0]
    row = posterior_source.draw_row(0.0)
    assert row["param_a"] == min_val
 
 
def test_draw_row_one_returns_highest(posterior_source):
    """draw_row with value=1.0 returns the row with the highest sort column value."""
    posterior_source._load()
    max_val = posterior_source._draws["param_a"].iloc[-1]
    row = posterior_source.draw_row(1.0)
    assert row["param_a"] == max_val
 
 
def test_draw_row_monotonic_with_increasing_value(posterior_source):
    """draw_row returns monotonically non-decreasing sort column values
    as input increases from 0 to 1."""
    values = np.linspace(0, 1, 20)
    drawn = [posterior_source.draw_row(v)["param_a"] for v in values]
    assert all(drawn[i] <= drawn[i + 1] for i in range(len(drawn) - 1))
 
 
def test_draw_row_does_not_reload_on_second_call(posterior_source, mocker):
    """draw_row does not reload the file on subsequent calls."""
    spy = mocker.spy(posterior_source, "_load")
    posterior_source.draw_row(0.5)
    posterior_source.draw_row(0.5)
    assert spy.call_count == 1