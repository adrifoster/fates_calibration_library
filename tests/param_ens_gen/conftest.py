"""Fixtures shared across param_ens_gen tests."""

import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pathlib import Path

from fates_calibration_library.param_ens_gen.posterior_source import PosteriorSource

N_PFTS = 3
N_LEAFAGE = 2
N_ORGANS = 4

# ========================================================================================
# ParamSpec fixtures
# ========================================================================================

def _base_row(**kwargs) -> pd.Series:
    """Build a minimal valid main-sheet row, with overrides."""
    defaults = {
        "parameter_name": "fates_leaf_slatop",
        "long_name": "Specific Leaf Area (SLA) at top of canopy, projected area basis",
        "category": "stomatal",
        "subcategory": "photosynthesis",
        "units": "m^2/gC",
        "coord": "['fates_pft']",
        "param_type": "default",
        "strategy": "uniform",
        "param_min": "0.005",
        "param_max": "0.05",
        "slice_dim": None,
        "slice_index": None,
        "root_param": None,
        "base_params": "",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)

@pytest.fixture
def default_row() -> pd.Series:
    """A valid row for a default PFT parameter.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row()


@pytest.fixture
def scalar_row() -> pd.Series:
    """A valid row for a scalar (non-PFT) default parameter.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_canopy_closure_thresh",
        long_name="tree canopy coverage at which crown area allometry changes from savanna to forest value",
        category="biogeochemistry",
        subcategory="vegetation dynamics",
        units="1/yr",
        coord="[]",
        param_min="0.7",
        param_max="0.9",
    )


@pytest.fixture
def sliced_row() -> pd.Series:
    """A valid row for a sliced parameter (fates_leafage_class x fates_pft).
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_vcmax25top",
        long_name="maximum carboxylation rate of Rub. at 25C, canopy top",
        category="stomatal",
        subcategory="photosynthesis",
        units="umol CO2/m^2/s",
        coord="['fates_leafage_class', 'fates_pft']",
        param_type="sliced",
        param_min="20.0",
        param_max="90.0",
        slice_dim="fates_leafage_class",
        slice_index=0,
        base_params="fates_leaf_vcmax25top",
    )

@pytest.fixture
def scale_from_root_row() -> pd.Series:
    """A valid row for a scale_from_root parameter.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="smpsc_delta",
        long_name="Soil water potential at full stomatal closing (delta)",
        category="stomatal",
        subcategory="vegetation water",
        units="mm",
        coord="['fates_pft']",
        param_type="scale_from_root",
        param_min="-600000",
        param_max="-20000",
        root_param="fates_nonhydro_smpso",
        base_params="fates_nonhydro_smpsc",
    )


@pytest.fixture
def joint_param_row() -> pd.Series:
    """A valid row for a joint parameter.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leafn_vert_scaler",
        long_name="Coefficient one for decrease in leaf nitrogen through the canopy, from Lloyd et al. 2010",
        category="stomatal",
        subcategory="photosynthesis",
        units="unitless",
        coord="['fates_pft']",
        param_type="joint",
        strategy="posterior",
        param_min="",
        param_max="",
        base_params="['fates_leafn_vert_scaler_coeff1', 'fates_leafn_vert_scaler_coeff2']",
    )

    
# ========================================================================================
# PosteriorSource fixtures
# ========================================================================================

@pytest.fixture
def posterior_file(tmp_path) -> Path:
    """A minimal posterior text file with two parameters and 20 rows.
 
    Columns: param_a, param_b
    Values are shuffled so that _load's sort behaviour is actually tested.
 
    Returns:
        Path: path to the temporary file
    """
    rng = np.random.default_rng(42)
    n = 20
    data = pd.DataFrame({
        "param_a": rng.permutation(np.linspace(0.0, 0.95, n)),
        "param_b": rng.permutation(np.linspace(10.0, 19.5, n)),
    })
    path = tmp_path / "posterior.txt"
    data.to_csv(path, sep=" ", index=False)
    return path

@pytest.fixture
def posterior_file_2(tmp_path) -> Path:
    """A second minimal posterior file for multi-source tests."""
    rng = np.random.default_rng(99)
    n = 20
    data = pd.DataFrame({
        "param_a": rng.permutation(np.linspace(1.0, 1.95, n)),
        "param_b": rng.permutation(np.linspace(20.0, 29.5, n)),
    })
    path = tmp_path / "posterior_2.txt"
    data.to_csv(path, sep=" ", index=False)
    return path


@pytest.fixture
def empty_posterior_file(tmp_path) -> Path:
    """An empty posterior text file with two parameters.
 
    Columns: param_a, param_b
 
    Returns:
        Path: path to the temporary file
    """
    data = pd.DataFrame({
        "param_a": [],
        "param_b": [],
    })
    path = tmp_path / "empty_posterior.txt"
    data.to_csv(path, sep=" ", index=False)
    return path
 
@pytest.fixture
def posterior_source(posterior_file) -> PosteriorSource:
    """A PosteriorSource pointing at posterior_file, not yet loaded.
 
    Returns:
        PosteriorSource: unloaded source
    """
    return PosteriorSource(
        path=posterior_file,
        array_indices="all",
        parameters=["param_a", "param_b"],
    )

@pytest.fixture
def empty_posterior_source(empty_posterior_file) -> PosteriorSource:
    """A PosteriorSource pointing at empty_posterior_file, not yet loaded.
 
    Returns:
        PosteriorSource: unloaded source
    """
    return PosteriorSource(
        path=empty_posterior_file,
        array_indices="all",
        parameters=["param_a", "param_b"],
    )

# ===========================================================================
# DistributionStat fixtures
# ===========================================================================

@pytest.fixture
def percent_row() -> pd.Series:
    """A valid row for a parameter with percent-based bounds.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_vcmax25top",
        param_min="50percent",
        param_max="50percent",
    )

@pytest.fixture
def pft_sheet() -> pd.DataFrame:
    """A minimal per-parameter PFT bounds sheet with 3 PFTs.
 
    Returns:
        pd.DataFrame: PFT bounds sheet
    """
    return pd.DataFrame({
        "pft_index": [1, 2, 3],
        "pft_name": ["white_spruce", "black_spruce", "deciduous"],
        "param_min": [0.005, 0.004, 0.008],
        "param_max": [0.040, 0.035, 0.060],
    })


@pytest.fixture
def pft_row(pft_sheet) -> pd.Series:
    """A valid row for a parameter with PFT-specific bounds.
 
    Returns:
        pd.Series: spreadsheet row
    """
    return _base_row(
        parameter_name="fates_leaf_slatop",
        param_min="pft",
        param_max="pft",
    )

# ===========================================================================
# Sampler fixtures
# ===========================================================================
 
 
@pytest.fixture
def posterior_config(posterior_file) -> dict:
    """A minimal posterior_config dict pointing at posterior_file.
 
    Covers a single broadcast source (array_indices='all') for two parameters.
 
    Returns:
        dict: posterior config
    """
    return {
        "parameters": ["param_a", "param_b"],
        "files": [
            {
                "path": str(posterior_file),
                "array_indices": "all",
            }
        ],
    }

@pytest.fixture
def multi_source_posterior_config(posterior_file, posterior_file_2) -> dict:
    """A posterior_config dict pointing at two non-broadcast sources covering different indices.
 
    Returns:
        dict: posterior config
    """
    return {
        "parameters": ["param_a", "param_b"],
        "files": [
            {
                "path": str(posterior_file),
                "array_indices": [0, 1],
            },
            {
                "path": str(posterior_file_2),
                "array_indices": [2]
            }
        ],
    }
 
 
@pytest.fixture
def pft_uniform_row(pft_sheet) -> pd.Series:
    """A valid row for a uniform parameter with PFT-specific bounds."""
    return _base_row(
        parameter_name="fates_leaf_slatop",
        strategy="uniform",
        param_min="pft",
        param_max="pft",
    )

# ===========================================================================
# Parameter fixtures
# ===========================================================================


@pytest.fixture
def default_ds() -> xr.Dataset:
    """A minimal default FATES parameter dataset covering all param types.

    Returns:
        xr.Dataset: default dataset
    """
    return xr.Dataset(
        {
            # default PFT parameter
            "fates_leaf_slatop": (
                ["fates_pft"],
                np.array([0.010, 0.020, 0.030]),
            ),
            # scalar parameter
            "fates_canopy_closure_thresh": (
                [],
                np.float64(0.5),
            ),
            # sliced parameter (leafage_class x pft)
            "fates_leaf_vcmax25top": (
                ["fates_leafage_class", "fates_pft"],
                np.array([[50.0, 60.0, 70.0],
                          [40.0, 50.0, 60.0]]),
            ),
            # scale_from_root — root and dependent
            "fates_nonhydro_smpso": (
                ["fates_pft"],
                np.array([-50000.0, -60000.0, -70000.0]),
            ),
            "fates_nonhydro_smpsc": (
                ["fates_pft"],
                np.array([-100000.0, -110000.0, -120000.0]),
            ),
            # joint targets
            "fates_leafn_vert_scaler_coeff1": (
                ["fates_pft"],
                np.array([0.012, 0.015, 0.005]),
            ),
            "fates_leafn_vert_scaler_coeff2": (
                ["fates_pft"],
                np.array([2.1, 2.5, 2.6]),
            ),
        }
    )


@pytest.fixture
def working_ds(default_ds) -> xr.Dataset:
    """A deep copy of default_ds to use as the working dataset.

    Args:
        default_ds (xr.Dataset): fixture

    Returns:
        xr.Dataset: working copy
    """
    return default_ds.copy(deep=True)
