"""Methods for processing and analyzing ILAMB data"""

import os
import logging
from datetime import date
from typing import Dict
from functools import reduce
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe

from fates_calibration_library.analysis_functions import (
    calculate_monthly_mean,
    calculate_annual_mean,
    calculate_month_of_max,
    area_mean,
)
from fates_calibration_library.utils import (
    evaluate_conversion_factor,
    should_skip_file,
    validate_dict_keys,
)
from fates_calibration_library.surface_data_functions import extract_biome

logger = logging.getLogger(__name__)

BOUNDS_VARS = ["lat_bounds", "lon_bounds", "lat_bnds", "lon_bnds"]


def regrid_all_ilamb_data(config_dict: Dict, ilamb_dict: Dict, target_grid: xr.Dataset):
    """Regrids all ILAMB datasets

    Args:
        config_dict (dict): Configuration containing regrid_dir, out_dir, etc.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        target_grid (xr.Dataset): Target grid for regridding.
    """

    REQUIRED_CONFIG_KEYS = {"top_dir", "regrid_dir", "regrid_tag", "user", "clobber"}
    REQUIRED_ILAMB_KEYS = {"sub_dir", "model", "in_var"}

    # make sure all our keys are here
    validate_dict_keys(config_dict, REQUIRED_CONFIG_KEYS, "config_dict")

    # create output directory if it doesn't exist
    os.makedirs(config_dict["regrid_dir"], exist_ok=True)

    # regrid each dataset
    for dataset, attributes in ilamb_dict.items():

        # make sure all our keys are here
        validate_dict_keys(attributes, REQUIRED_ILAMB_KEYS, f"ilamb_dict['{dataset}']")

        # we skip regridding ef and albedo because these will be calculated from mean values
        if attributes["in_var"] in {"ef", "albedo"}:
            continue
        out_file = construct_output_filename(config_dict, attributes, True)

        # skip if file exists and clobber is False
        if should_skip_file(out_file, config_dict["clobber"]):
            logger.info(f"File {out_file} for {dataset} exists, skipping")
            continue

        ds_out = regrid_dataset(config_dict, attributes, target_grid)
        ds_out.to_netcdf(out_file, mode="w")


def construct_output_filename(config: Dict, attributes: Dict, regridded: bool) -> str:
    """Creates an output file name from set of config information

    Args:
        config (dict): Configuration containing regrid_dir, out_dir
        attributes (dict): Dictionary with ILAMB dataset information.
        regridded (bool): whether or not this is the regridded raw file

    Returns:
        str: output file name
    """
    if regridded:
        return os.path.join(
            config["regrid_dir"],
            f"{attributes['model']}_{attributes['out_var'].upper()}_{config['regrid_tag']}.nc",
        )
    else:
        return os.path.join(
            config["out_dir"],
            f"{attributes['model']}_{attributes['out_var'].upper()}.nc",
        )


def add_metadata(ds: xr.Dataset, original_file: str, user: str) -> xr.Dataset:
    """Adds metadata to a dataset

    Args:
        ds (xr.Dataset): input dataset
        original_file (str): original file location
        user (str): user

    Returns:
        xr.Dataset: output dataset
    """
    ds.attrs["Original"] = original_file
    ds.attrs["Date"] = str(date.today())
    ds.attrs["Author"] = user
    return ds


def regrid_dataset(
    config_dict: Dict, attributes: Dict, target_grid: xr.DataArray
) -> xr.Dataset:
    """Handles regridding for a single dataset

    Args:
        config (dict): Configuration containing top_dir, clobber, out_dir
        attributes (dict): dictionary with attributes about this ILAMB dataset
        target_grid (xr.DataArray): target grid for regridding
    """

    # read or ILAMB data
    ilamb_dat, original_file = read_ilamb_data(config_dict["top_dir"], attributes)

    # drop bounds if they exist in the dataset
    to_drop = [var for var in BOUNDS_VARS if var in ilamb_dat.variables]
    ilamb_dat = ilamb_dat.drop_vars(to_drop, errors="ignore")

    # regrid the dataset
    ds = regrid_data(ilamb_dat, target_grid, attributes["in_var"])
    ds = add_metadata(ds, original_file, config_dict["user"])

    return ds


def get_all_ilamb_data(config_dict: Dict, ilamb_dict: Dict):

    # create output directory if it doesn't exist
    os.makedirs(config_dict["out_dir"], exist_ok=True)

    # process each dataset
    for dataset, attributes in ilamb_dict.items():
        out_file = construct_output_filename(config_dict, attributes, False)

        # skip if file exists and clobber is False
        if should_skip_file(out_file, config_dict["clobber"]):
            logger.info(f"File {out_file} for {dataset} exists, skipping")
            continue

        ds_out = process_dataset(config_dict, ilamb_dict, attributes)
        ds_out.to_netcdf(out_file, mode="w")


def get_annual_ef(le, le_dict, sh, sh_dict, metadata, out_var):

    # calculate annual LE and SH
    annual_le = get_annual_ds(
        le,
        le_dict["in_var"],
        le_dict["in_var"],
        evaluate_conversion_factor(le_dict["time_conversion_factor"]),
        metadata,
    )

    annual_sh = get_annual_ds(
        sh,
        sh_dict["in_var"],
        sh_dict["in_var"],
        evaluate_conversion_factor(sh_dict["time_conversion_factor"]),
        metadata,
    )

    annual_ds = evapfrac(
        annual_sh[sh_dict["in_var"]], annual_le[le_dict["in_var"]], 0.0
    ).to_dataset(name=out_var)
    annual_ds[out_var].attrs["units"] = ""
    annual_ds[out_var].attrs["long_name"] = "evaporative fraction"

    return annual_ds


def get_annual_albedo(rsds, rsds_dict, rsus, rsus_dict, metadata, out_var):

    # calculate annual rsds and rsus
    annual_rsds = get_annual_ds(
        rsds,
        rsds_dict["in_var"],
        rsds_dict["in_var"],
        evaluate_conversion_factor(rsds_dict["time_conversion_factor"]),
        metadata,
    )

    annual_rsus = get_annual_ds(
        rsus,
        rsus_dict["in_var"],
        rsus_dict["in_var"],
        evaluate_conversion_factor(rsus_dict["time_conversion_factor"]),
        metadata,
    )

    annual_ds = albedo(
        annual_rsds[rsds_dict["in_var"]], annual_rsus[rsus_dict["in_var"]], 0.0
    ).to_dataset(name=out_var)
    annual_ds[out_var].attrs["units"] = ""
    annual_ds[out_var].attrs["long_name"] = "albedo"

    return annual_ds

def filter_incomplete_years(ds, var):
    
    monthly_counts = ds[var].groupby('time.year').count()
    complete_years = monthly_counts.where(monthly_counts == 12, drop=True).year
    return ds.where(ds['time.year'].isin(complete_years), drop=True)

def process_dataset(
    config_dict: dict,
    ilamb_dict: dict,
    attributes: dict,
):
    """Handles reading, conversion, and regridding for a single dataset

    Args:
        config (dict): Configuration containing top_dir, clobber, out_dir, start_date, end_date.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        dataset (str): ILAMB dataset name
        attributes (dict): dictionary with attributes about this ILAMB dataset
    """

    # read in regions
    region = xr.open_dataset(
        os.path.join(config_dict["top_dir"], "regions/GlobalLandNoAnt.nc")
    )
    # swap out 0.0 (land) for 1.0
    region["ids"] = xr.where(region.ids == 0.0, 1.0, 0.0)

    if attributes["in_var"] == "ef":

        le_dict, sh_dict = (
            ilamb_dict[f"{attributes['model']}_LE"],
            ilamb_dict[f"{attributes['model']}_SH"],
        )

        # prepare metadata
        original_file = build_file_paths(
            config_dict["top_dir"],
            [le_dict["sub_dir"], sh_dict["sub_dir"]],
            attributes["model"],
            [le_dict["filename"], sh_dict["filename"]],
        )

        metadata = {
            "units": attributes["units"],
            "longname": attributes["longname"],
            "original_file": original_file,
            "user": config_dict["user"],
        }

        le, sh = get_ef_data(
            config_dict["regrid_dir"], le_dict, sh_dict, config_dict["regrid_tag"]
        )
        le = filter_incomplete_years(le, le_dict['in_var'])
        sh = filter_incomplete_years(sh, sh_dict['in_var'])
        annual_ds = get_annual_ef(
            le, le_dict, sh, sh_dict, metadata, attributes["out_var"]
        )

        le_raw, sh_raw = get_ef_data(config_dict["top_dir"], le_dict, sh_dict)
        le_raw = filter_incomplete_years(le_raw, le_dict['in_var'])
        sh_raw = filter_incomplete_years(sh_raw, sh_dict['in_var'])
        annual_original = get_annual_ef(
            le_raw, le_dict, sh_raw, sh_dict, metadata, attributes["out_var"]
        )

        # calculate sub-annual for climatology
        ilamb_dat = evapfrac(
            sh_raw[sh_dict["in_var"]], le_raw[le_dict["in_var"]], 20.0
        ).to_dataset(name=attributes["out_var"])
        regridded_dat = evapfrac(
            sh[sh_dict["in_var"]], le[le_dict["in_var"]], 20.0
        ).to_dataset(name=attributes["out_var"])

    elif attributes["in_var"] == "albedo":

        rsds_dict, rsus_dict = (
            ilamb_dict[f"{attributes['model']}_RSDS"],
            ilamb_dict[f"{attributes['model']}_FSR"],
        )

        # prepare metadata
        original_file = build_file_paths(
            config_dict["top_dir"],
            [rsds_dict["sub_dir"], rsus_dict["sub_dir"]],
            attributes["model"],
            [rsds_dict["filename"], rsus_dict["filename"]],
        )

        metadata = {
            "units": attributes["units"],
            "longname": attributes["longname"],
            "original_file": original_file,
            "user": config_dict["user"],
        }

        rsds, rsus = get_albedo_data(
            config_dict["regrid_dir"], rsds_dict, rsus_dict, config_dict["regrid_tag"]
        )
        rsds = filter_incomplete_years(rsds, rsds_dict['in_var'])
        rsus = filter_incomplete_years(rsus, rsus_dict['in_var'])
        annual_ds = get_annual_albedo(
            rsds, rsds_dict, rsus, rsus_dict, metadata, attributes["out_var"]
        )

        rsds_raw, rsus_raw = get_albedo_data(
            config_dict["top_dir"], rsds_dict, rsus_dict
        )
        rsds_raw = filter_incomplete_years(rsds_raw, rsds_dict['in_var'])
        rsus_raw = filter_incomplete_years(rsus_raw, rsus_dict['in_var'])
        annual_original = get_annual_albedo(
            rsds_raw, rsds_dict, rsus_raw, rsus_dict, metadata, attributes["out_var"]
        )

        ilamb_dat = albedo(
            rsds_raw[rsds_dict["in_var"]], rsus_raw[rsus_dict["in_var"]], 10.0
        ).to_dataset(name=attributes["out_var"])
        regridded_dat = albedo(
            rsds[rsds_dict["in_var"]], rsus[rsus_dict["in_var"]], 20.0
        ).to_dataset(name=attributes["out_var"])

    else:

        # read in regridded dataset
        file_name = os.path.join(
            config_dict["regrid_dir"],
            f"{attributes['model']}_{attributes['out_var'].upper()}_{config_dict['regrid_tag']}.nc",
        )
        regridded_dat = xr.open_dataset(file_name)
        if attributes['in_var'] != 'biomass':
            regridded_dat = filter_incomplete_years(regridded_dat, attributes['in_var'])

        # prepare metadata
        metadata = {
            "units": attributes["units"],
            "longname": attributes["longname"],
            "original_file": file_name,
            "user": config_dict["user"],
        }
        
        # calculate annual data
        annual_ds = get_annual_ds(
            regridded_dat,
            attributes["in_var"],
            attributes["out_var"],
            evaluate_conversion_factor(attributes["time_conversion_factor"]),
            metadata,
        )
        

        # read in raw data
        ilamb_dat, original_file = read_ilamb_data(config_dict["top_dir"], attributes)
        if attributes['in_var'] != 'biomass':
            ilamb_dat = filter_incomplete_years(ilamb_dat, attributes['in_var'])
        annual_original = get_annual_ds(
            ilamb_dat,
            attributes["in_var"],
            attributes["out_var"],
            evaluate_conversion_factor(attributes["time_conversion_factor"]),
            metadata,
        )

    # calculate cell areas and land area
    cell_area = get_ilamb_land_area(ilamb_dat)
    region_regridder = xe.Regridder(region["ids"], ilamb_dat, "bilinear")
    landfrac = region_regridder(region["ids"])
    land_area = landfrac * cell_area

    global_mean = area_mean(
        annual_original[attributes["out_var"]],
        evaluate_conversion_factor(attributes["area_conversion_factor"]),
        land_area,
    ).to_dataset(name=f"{attributes['out_var']}_global")

    time_len = len(ilamb_dat.time)
    num_years = len(np.unique(ilamb_dat["time.year"]))

    if time_len > 1 and time_len > num_years:

        monthly_mean = get_monthly_ds(
            ilamb_dat[attributes["in_var"]],
            f"{attributes['out_var']}_monthly",
            evaluate_conversion_factor(attributes["time_conversion_factor"]),
            metadata,
        )
        
        regridded_monthly_mean = get_monthly_ds(
            regridded_dat[attributes["in_var"]],
            f"{attributes['out_var']}_monthly",
            evaluate_conversion_factor(attributes["time_conversion_factor"]),
            metadata,
        )

        # calculate month of maximum value
        month_of_max = calculate_month_of_max(
            regridded_monthly_mean[f"{attributes['out_var']}_monthly"]
        ).to_dataset(name=f"{attributes['out_var']}_month_of_max")
        
        climatology_ds = get_ilamb_climatology(monthly_mean, 
            land_area, attributes['out_var'], 
            evaluate_conversion_factor(attributes["area_conversion_factor"]))

        # return all files combined
        return xr.merge([annual_ds, global_mean, month_of_max, climatology_ds])
    else:
        return xr.merge([annual_ds, global_mean])

def get_ilamb_land_area(ds):

    lats = ds.lat.values
    lons = ds.lon.values

    areas = cell_areas(lats, lons)
    area_da = xr.DataArray(
        areas * 1e-6,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons},
        name="cell_area",
        attrs={"units": "km2", "description": "grid cell area"},
    )

    return area_da


def get_ilamb_climatology(
    data_array: xr.DataArray, land_area: xr.DataArray, out_var: str, area_cf
) -> xr.Dataset:
    """Returns dataset of climatology of the monthly input data

    Args:
        monthly_mean (xr.DataArray): monthly mean values
        land_area (xr.DataArray): land area [km2]
        out_var (str): name of output variable
        area_cf (float): area conversion factor

    Returns:
        xr.Dataset: climatology dataset
    """

    # sum up to get areas where there is no data
    # sum_monthly = monthly_mean[f"{out_var}_monthly"].sum(dim="month")
    all_nan = data_array[f"{out_var}_monthly"].isnull().all(dim="month")
    
    land_area = xr.where(~all_nan, land_area, 0.0)

    if area_cf is None:
        area_cf = 1 / land_area.sum(dim=["lat", "lon"]).values

    # weight by landarea
    area_weighted = land_area * data_array[f"{out_var}_monthly"]

    # calculate area mean
    climatology = area_cf * area_weighted.sum(dim=["lat", "lon"]).to_dataset(
        name=f"{out_var}_cycle"
    )

    # calculate anomaly
    climatology_mean = climatology[f"{out_var}_cycle"].mean(dim="month")
    monthly_anomaly = (climatology[f"{out_var}_cycle"] - climatology_mean).to_dataset(
        name=f"{out_var}_anomaly"
    )

    climatology_ds = xr.merge([climatology, monthly_anomaly])

    return climatology_ds


def build_file_paths(top_dir, sub_dirs, model, filenames):
    """
    Constructs file paths from top directory, subdirectories, model name, and filenames.

    Args:
        top_dir (str): The top-level directory.
        sub_dirs (list of str): List of subdirectories.
        model (str): The model name.
        filenames (list of str): List of filenames.

    Returns:
        str: A formatted string joining the paths with " and ".
    """
    paths = [
        os.path.join(top_dir, sub_dir, model, filename)
        for sub_dir, filename in zip(sub_dirs, filenames)
    ]
    return " and ".join(paths)


def read_ilamb_data(
    top_dir: str,
    attributes: dict,
) -> tuple[xr.Dataset, str]:
    """Reads in a raw dataset from the ILAMB data repository and potentially filters
    on min/max value

    Args:
        top_dir (str): top directory with raw ILAMB data
        attributes (dict): dictionary with attributes about this ILAMB dataset such as
                'sub_dir', 'model', 'filename', 'in_var', and optional 'max_val'/'min_val'.

    Returns:
        tuple[xr.Dataset, str]: A tuple containing:
            - the ILAMB dataset as an xarray.Dataset
            - the constructed path string of the original file
    Raises:
        KeyError: if the required keys are missing in attributes
        IOError: if the file reading fails
        ValueError: if the input variable isn't found on dataset
    """

    # validate dictionary keys
    validate_dict_keys(
        attributes, {"sub_dir", "model", "filename", "in_var"}, "attributes_dict"
    )

    # read dataset
    file_path = build_ilamb_file_path(
        top_dir, attributes["sub_dir"], attributes["model"], attributes["filename"]
    )
    logger.info(f"Reading ILAMB dataset from {file_path}")

    try:
        raw_ds = xr.open_dataset(file_path)
    except FileNotFoundError:
        raise IOError(f"File not found: {file_path}")
    except OSError as e:
        raise IOError(f"Error opening dataset at {file_path}: {e}")

    # check to make sure input  variable is on dataset
    if attributes["in_var"] not in raw_ds:
        raise ValueError(f"Variable {attributes['in_var']} not found in raw dataset.")

    # filter on min/max values if present
    filtered_ds = filter_dataset(
        raw_ds,
        attributes["in_var"],
        attributes.get("min_val", None),
        attributes.get("max_val", None),
    )
    if attributes["in_var"] not in filtered_ds:
        raise ValueError(
            f"Variable {attributes['in_var']} not found in dataset after filtering."
        )

    return filtered_ds, file_path


def get_monthly_ds(
    ds: xr.Dataset, out_var: str, conversion_factor: float, metadata: Dict[str, str]
) -> xr.Dataset:
    """Calculates a monthly dataset from an input dataset for variable in_var

    Args:
        ds (xr.Dataset): input dataset
        in_var (str): variable to calculate annual values
        out_var (str): name of output monthly variable
        conversion_factor (float): conversion factor to go to monthly values
        metadata (dict): Dictionary containing:
            - 'units' (str): Units of the output variable.
            - 'longname' (str): Long name of the variable.
            - 'original_file' (str): Path to the original file.
            - 'user' (str): User's name.

    Returns:
        xr.Dataset: output dataset
    """

    monthly_ds = calculate_monthly_mean(ds, conversion_factor).to_dataset(name=out_var)
    monthly_ds[out_var].attrs["units"] = metadata["units"]
    monthly_ds[out_var].attrs["long_name"] = metadata["longname"]

    return monthly_ds


def build_ilamb_file_path(top_dir: str, sub_dir: str, model: str, filename: str) -> str:
    """Constructs a file path to a raw ILAMB dataset

    Args:
        top_dir (str): top ILAMB DATA directory
        sub_dir (str): subdirectory for the variable
        model (str): model name (also directory)
        filename (str): file name

    Returns:
        str: full path to raw ILAMB dataset
    """

    return os.path.join(top_dir, sub_dir, model, filename)


def filter_dataset(
    ds: xr.Dataset, var: str, min_val: float = None, max_val: float = None
) -> xr.Dataset:
    """Filters a dataset based on value constraints.

    Args:
        ds (xr.Dataset): Input dataset.
        var (str): Variable to filter on.
        min_val (float, optional): minimum value constraint. Defaults to None.
        max_val (float, optional): maximum value constraint. Defaults to None.

    Returns:
        xr.Dataset: filtered dataset
    Raises:
        KeyError: var not found on dataset
    """

    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    
    # filter by min/max
    if max_val is not None:
        ds = ds.where(ds[var] <= max_val)
    if min_val is not None:
        ds = ds.where(ds[var] >= min_val)

    return ds


def get_ef_data(
    top_dir: str,
    le_dict: dict,
    sh_dict: dict,
    regrid_tag: str = None,
) -> xr.Dataset:
    """Calculate evaporative fraction for a given ILAMB dataset (LE and SH)

    Args:
        top_dir (str): top ILAMB directory
        in_var (str): variable name to set to
        le_dict (dict): dictionary with information about latent heat data
        sh_dict (dict): dictionary with information about sensible heat data

    Returns:
        xr.Dataset: evaporative fraction dataset
    """

    # read in datasets
    if regrid_tag is not None:
        le_path = os.path.join(
            top_dir, f"{le_dict['model']}_{le_dict['out_var'].upper()}_{regrid_tag}.nc"
        )
        sh_path = os.path.join(
            top_dir, f"{sh_dict['model']}_{sh_dict['out_var'].upper()}_{regrid_tag}.nc"
        )
    else:
        le_path = os.path.join(
            top_dir, le_dict["sub_dir"], le_dict["model"], le_dict["filename"]
        )
        sh_path = os.path.join(
            top_dir, sh_dict["sub_dir"], sh_dict["model"], sh_dict["filename"]
        )
    le = xr.open_dataset(le_path)
    sh = xr.open_dataset(sh_path)

    # filter by date and min/max
    le = filter_dataset(le, le_dict["in_var"], le_dict.get("min_val", None), 
                        le_dict.get("max_val", None))
    sh = filter_dataset(sh, sh_dict["in_var"], sh_dict.get("min_val", None),
                        sh_dict.get("max_val", None))

    return le, sh


def evapfrac(
    sh: xr.DataArray, le: xr.DataArray, energy_threshold: float
) -> xr.DataArray:
    """Calculates evaporative fraction as le/(le + sh)

    Args:
        sh (xr.DataArray): sensible heat flux
        le (xr.DataArray): latent heat flux
        energy_threshold (float): energy threshold to prevent div/0s

    Returns:
        xr.DataArray: evaporative fraction [0-1]
    """
    sh = sh.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    le = le.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    ef = le / (le + sh)

    return ef


def get_albedo_data(
    top_dir: str,
    rsds_dict: dict,
    rsus_dict: dict,
    regrid_tag: str = None,
) -> xr.Dataset:
    """Calculate albedo for a given ILAMB dataset (RSDS and RSUS)

    Args:
        top_dir (str): top ILAMB directory
        in_var (str): variable name to set to
        le_dict (dict): dictionary with information about rsds data
        sh_dict (dict): dictionary with information about rsus data
    Returns:
        xr.Dataset: evaporative fraction dataset
    """

    # read in datasets
    if regrid_tag is not None:
        rsds_path = os.path.join(
            top_dir,
            f"{rsds_dict['model']}_{rsds_dict['out_var'].upper()}_{regrid_tag}.nc",
        )
        rsus_path = os.path.join(
            top_dir,
            f"{rsus_dict['model']}_{rsus_dict['out_var'].upper()}_{regrid_tag}.nc",
        )
    else:
        rsds_path = os.path.join(
            top_dir, rsds_dict["sub_dir"], rsds_dict["model"], rsds_dict["filename"]
        )
        rsus_path = os.path.join(
            top_dir, rsus_dict["sub_dir"], rsus_dict["model"], rsus_dict["filename"]
        )

    rsds = xr.open_dataset(rsds_path)
    rsus = xr.open_dataset(rsus_path)


    # filter by date and min/max
    rsds = filter_dataset(rsds, rsds_dict["in_var"], rsds_dict.get("min_val", None),
                          rsds_dict.get("max_val", None))
    rsus = filter_dataset(rsus, rsus_dict["in_var"], rsus_dict.get("min_val", None),
                          rsus_dict.get("max_val", None))

    return rsds, rsus


def albedo(
    rsds: xr.DataArray, rsus: xr.DataArray, energy_threshold: float
) -> xr.DataArray:
    """Calculates albedo as rsus/rsds

    Args:
        rsds (xr.DataArray): downward shortwave radiation
        rsus (xr.DataArray): upward shortwave radiation
        energy_threshold (float): energy threshold to prevent div/0s

    Returns:
        xr.DataArray: albedo [0-1]
    """
    rsds = rsds.where(rsds > energy_threshold)
    rsus = rsus.where(rsds > energy_threshold)
    alb = rsus / rsds

    return alb


def cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Given arrays of latitude and longitude, return cell areas in square meters

    Args:
        lat (np.ndarray): a 1D array of latitudes (cell centroids)
        lon (np.ndarray): a 1D array of longitudes (cell centroids)

    Returns:
        np.ndarray: a 2D array of cell areas [m2]
    """

    earth_radius = 6.371e6

    lons = np.zeros(lon.size + 1)
    lons[1:-1] = 0.5 * (lon[1:] + lon[:-1])
    lons[0] = lon[0] - 0.5 * (lon[1] - lon[0])
    lons[-1] = lon[-1] + 0.5 * (lon[-1] - lon[-2])

    if lons.max() > 181:
        lons -= 180
    lons = lons.clip(-180, 180)
    lons *= np.pi / 180.0

    lats = np.zeros(lat.size + 1)
    lats[1:-1] = 0.5 * (lat[1:] + lat[:-1])
    lats[0] = lat[0] - 0.5 * (lat[1] - lat[0])
    lats[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])

    lats = lats.clip(-90, 90)
    lats *= np.pi / 180.0

    dlons = np.abs(earth_radius * (lons[1:] - lons[:-1]))
    dlats = np.abs(earth_radius * (np.sin(lats[1:]) - np.sin(lats[:-1])))
    areas = np.outer(dlons, dlats).T

    return areas


def get_annual_ds(
    ds: xr.Dataset,
    in_var: str,
    out_var: str,
    conversion_factor: float,
    metadata: Dict[str, str],
) -> xr.Dataset:
    """Calculates an annual dataset from an input dataset for variable in_var

    Args:
        ds (xr.Dataset): input dataset
        in_var (str): variable to calculate annual values
        out_var (str): name of output annual variable
        conversion_factor (float): conversion factor to go to annual values
        metadata (dict): Dictionary containing:
            - 'units' (str): Units of the output variable.
            - 'longname' (str): Long name of the variable.
            - 'original_file' (str): Path to the original file.
            - 'user' (str): User's name.

    Returns:
        xr.Dataset: output dataset
    """

    # calculate annual values
    if len(ds.time) > 1:
        annual = calculate_annual_mean(ds[in_var], conversion_factor)
        annual_ds = annual.to_dataset(name=out_var)
    else:
        year = ds["time.year"].values[0]
        annual_ds = ds[in_var].isel(time=0).to_dataset(name=out_var).drop_vars("time")
        annual_ds = annual_ds.expand_dims(year=[year]).assign_coords(
            year=("year", [year])
        )

    annual_ds[out_var].attrs["units"] = metadata["units"]
    annual_ds[out_var].attrs["long_name"] = metadata["longname"]

    # add some global attributes
    annual_ds.attrs["Original"] = metadata["original_file"]
    annual_ds.attrs["Date"] = str(date.today())
    annual_ds.attrs["Author"] = metadata["user"]

    return annual_ds


def regrid_data(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    var: str,
    method: str = "conservative",
) -> xr.Dataset:
    """Regrids an ILAMB dataset based on an input target grid

    Args:
        ds (xr.Dataset): input dataset
        target_grid (xr.Dataset): target grid
        var (str): variable to regrid
        method (str, optional): regridding method to use. Defaults to "bilinear".
        interpolate (bool, optional): interpolate lats/lons to get rid of
                                     ocean/land mismatch. Defaults to True.

    Returns:
        xr.Dataset: output dataset
    """

    # create mask
    ds[var] = ds[var].fillna(0)

    regridder = xe.Regridder(ds[var], target_grid, method)
    ds_regrid = regridder(ds[var]).to_dataset(name=var)
    ds_regrid[var] = ds_regrid[var] * target_grid.landmask
    ds_regrid[var].attrs = ds[var].attrs
    ds_regrid.attrs = ds.attrs

    return ds_regrid

def compile_ilamb_datasets(
    out_dir: str, ilamb_dict: dict, land_area: xr.DataArray
) -> xr.Dataset:
    """Compiles all regridded ILAMB datasets, computes average and variance over years,
    and merges into a single dataset

    Args:
        out_dir (str): output directory where files are located
        ilamb_dict (dict): dictionary with information about ILAMB data
        land_area (xr.DataArray): land area

    Returns:
        xr.Dataset: compiled dataset
    """

    # group datasets by variable
    var_to_models = {}
    for _, attributes in ilamb_dict.items():
        out_var = attributes["out_var"]
        var_to_models.setdefault(out_var, []).append(attributes["model"])

    # process each variable separately
    compiled_data = [
        compile_variable(out_var, models, out_dir)
        for out_var, models in var_to_models.items()
    ]

    ds_out = xr.merge(compiled_data)
    ds_out["land_area"] = land_area

    return ds_out


def compile_variable(var: str, models: list[str], out_dir: str) -> xr.Dataset:
    """Compiles a specific ILAMB variable across multiple models.

    Args:
        var (str): Variable name.
        models (list[str]): List of models corresponding to this variable.
        out_dir (str): Directory where files are stored.

    Returns:
        xr.Dataset: Combined dataset for the variable.
    """
    datasets = []
    for model in models:
        file_name = os.path.join(out_dir, f"{model}_{var.upper()}.nc")
        datasets.append(xr.open_dataset(file_name))
    var_ds = xr.concat(datasets, dim="model", data_vars="all")
    return var_ds.assign_coords(model=("model", models))


def get_average_and_iav(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Gives average and variance of values of 'var' across years for a dataset

    Args:
        ds (xr.Dataset): input dataset
        var (str): variable to average

    Returns:
        xr.Dataset: output dataset
    """

    # average
    annual_mean = ds[var].mean(dim="year").to_dataset(name=var)
    annual_mean[var].attrs["units"] = ds[var].attrs["units"]
    annual_mean[var].attrs["long_name"] = ds[var].attrs["long_name"]

    # variance
    annual_var = ds[var].var(dim="year").to_dataset(name=f"{var}_iav")
    annual_var[f"{var}_iav"].attrs["units"] = ds[var].attrs["units"]
    annual_var[f"{var}_iav"].attrs[
        "long_name"
    ] = f"interannual variation of {ds[var].attrs['long_name']}"

    annual_ds = xr.merge([annual_mean, annual_var])

    return annual_ds


def get_model_da(ds: xr.Dataset, var: str, models: list[str]) -> xr.DataArray:
    """Returns just the ILAMB DataArray for the specified variable and models

    Args:
        ds (xr.Dataset): input dataset
        var (str): variable
        models (list[str]): list of models

    Returns:
        xr.DataArray: output data array
    """
    return ds[var].where(ds.model.isin(models), drop=True)


def average_obs_across_models(
    obs_ds: xr.Dataset, models: list[str], var_name: str
) -> xr.Dataset:
    """Gets average of a variable across a set of ILAMB models

    Args:
        obs_ds (xr.Dataset): input dataset
        models (list[str]): list of models
        var_name (str): variable to average

    Returns:
        xr.Dataset: output data
    """

    obs = get_model_da(obs_ds, var_name, models)

    obs_mean = obs.mean(dim="model").to_dataset(name=f"{var_name}")
    obs_var = obs.var(dim="model").to_dataset(name=f"{var_name}_var")

    obs_sd = obs.std(dim="model")
    obs_rel_sd = (obs_sd / obs_sd.mean()).to_dataset(name=f"{var_name}_rel_sd")

    obs_data = xr.merge([obs_mean, obs_var, obs_rel_sd])

    return obs_data


def extract_obs(
    obs_ds: xr.Dataset,
    var: str,
    models: list[str],
    lats: np.ndarray,
    lons: np.ndarray,
) -> pd.DataFrame:
    """Extracts observations for a variable and for an input lat/lon

    Args:
        obs_ds (xr.Dataset): input dataset
        var (str): variable in question
        models (list[str]): list of models
        lats (np.ndarray): list of latitudes
        lons (np.ndarray): list of longitudes

    Returns:
        pd.DataFrame: output dataframe
    """

    # average observations by model
    obs = average_obs_across_models(obs_ds, models, var)

    # extract observations at the chosen gridcells
    var_mean = np.zeros(len(lats))
    var_var = np.zeros(len(lats))
    var_rel_sd = np.zeros(len(lats))

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        nearest_index_lat = np.abs(obs["lat"] - lat).argmin()
        nearest_index_lon = np.abs(obs["lon"] - lon).argmin()

        # grab data at correct lat/lon
        var_mean[i] = obs[var][nearest_index_lat, nearest_index_lon]
        if len(models) > 1:
            var_var[i] = obs[f"{var}_var"][nearest_index_lat, nearest_index_lon]
            var_rel_sd[i] = obs[f"{var}_rel_sd"][nearest_index_lat, nearest_index_lon]

    obs_df = pd.DataFrame(
        {
            "lat": lats,
            "lon": lons,
            f"{var}": var_mean,
            f"{var}_var": var_var,
            f"{var}_rel_sd": var_rel_sd,
        }
    )

    return obs_df


def filter_df(df: pd.DataFrame, filter_vars: list[str], tol: float) -> pd.DataFrame:
    """Filters a dataframe for a set of variables on an input tolerance

    Args:
        df (pd.DataFrame): input dataframe
        filter_vars (list[str]): variables to filter
        tol (float): tolerance level

    Returns:
        pd.DataFrame: filtered dataframe
    """

    for var in filter_vars:
        df = df.where(df[f"{var}_rel_sd"] < tol)

    df = df.dropna()

    return df


def extract_ilamb_obs(
    obs_ds: xr.Dataset,
    grids: pd.DataFrame,
    biome: xr.DataArray,
    ilamb_config: dict,
    threshold_dict: dict,
):
    """Extract ILAMB and Whittaker biomes for a set of lat/lons

    Args:
        obs_ds (xr.Dataset): ILAMB observations
        grids (pd.DataFrame): dataframe with lat and lons
        biome (xr.DataArray): whittaker biomes
        ilamb_config (dict): config dictionary with information about ILAMB data
        threshold_dict (dict): config dicationary with information about thresholding:
            - 'threshold_vars': variables to threshold uncertainty on
            - 'tol':            tolerance value

    Returns:
        _type_: _description_
    """

    all_df = []
    for _, attributes in ilamb_config.items():
        # extract all ILAMB data
        all_df.append(
            extract_obs(
                obs_ds,
                attributes["var"],
                attributes["models"],
                grids.lats.values,
                grids.lons.values,
            )
        )

    # also add in whittaker biome
    all_df.append(
        extract_biome(biome, grids.lats.values, grids.lons.values, grids.pft.values)
    )
    out_df = reduce(lambda x, y: pd.merge(x, y, on=["lat", "lon"]), all_df)

    # return filtered df
    return filter_df(out_df, threshold_dict["filter_vars"], threshold_dict["tol"])
