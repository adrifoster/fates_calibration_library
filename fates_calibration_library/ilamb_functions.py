"""Methods for processing and analyzing ILAMB data"""

import os
from datetime import date
from typing import Dict, Optional
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe


from fates_calibration_library.analysis_functions import (
    calculate_monthly_mean,
    calculate_annual_mean,
    get_monthly_max,
    get_conversion_factor,
)
from fates_calibration_library.utils import config_to_dict, str_to_bool

def get_all_ilamb_data(config_dict: Dict, ilamb_dict: Dict, target_grid: xr.Dataset):
    """Processes ILAMB datasets: reads, converts to annual values, regrids, and saves.

    Args:
        config_dict (dict): Configuration containing top_dir, clobber, out_dir, start_date,
                        end_date, and user.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        target_grid (xr.Dataset): Target grid for regridding.
        clobber (bool): whether to overwrite files.
    """

    # create output directory if it doesn't exist
    os.makedirs(config_dict["out_dir"], exist_ok=True)

    # process each dataset
    for dataset, attributes in ilamb_dict.items():
        out_file = os.path.join(
            config_dict["out_dir"],
            f"{attributes['model']}_{attributes['out_var'].upper()}.nc",
        )

        # skip if file exists and clobber is False
        if os.path.isfile(out_file) and not config_dict["clobber"]:
            print(f"File {out_file} for {dataset} exists, skipping")
            return
        print(out_file)

        ds_out = process_dataset(config_dict, ilamb_dict, attributes, target_grid)
        ds_out.to_netcdf(out_file, mode="w")


def process_dataset(
    config_dict: dict,
    ilamb_dict: dict,
    attributes: dict,
    target_grid: xr.DataArray,
):
    """Handles reading, conversion, and regridding for a single dataset

    Args:
        config (dict): Configuration containing top_dir, clobber, out_dir, start_date, end_date.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        dataset (str): ILAMB dataset name
        attributes (dict): dictionary with attributes about this ILAMB dataset
        target_grid (xr.DataArray): target grid for regridding
    """

    # read or compute ILAMB data
    ilamb_dat, original_file = read_ilamb_data(
        config_dict, ilamb_dict, attributes
    )

    # prepare metadata
    metadata = {
        "units": attributes["units"],
        "longname": attributes["longname"],
        "original_file": original_file,
        "user": config_dict["user"],
    }

    # calculate annual dataset
    annual_ds = get_annual_ds(
        ilamb_dat,
        attributes["in_var"],
        attributes["out_var"],
        get_conversion_factor(attributes["conversion_factor"]),
        metadata,
    )

    # regrid annual
    regridded_annual = regrid_ilamb_ds(annual_ds, target_grid, attributes["out_var"])

    # monthly mean
    if attributes["conversion_factor"] in ['intrinsic', 'mrro']:
        conversion_factor = 1.0
    else:
        conversion_factor = float(attributes["conversion_factor"])
        
    monthly_mean = calculate_monthly_mean(
        ilamb_dat[attributes["in_var"]],
        conversion_factor,
    )
        
    # regrided monthly mean
    regridded_monthly = regrid_ilamb_ds(
        monthly_mean.to_dataset(name=f"{attributes['out_var']}_monthly"),
        target_grid,
        f"{attributes['out_var']}_monthly")
    
    sum_monthly = regridded_monthly[f"{attributes['out_var']}_monthly"].sum(dim='month')
    regridded_monthly = regridded_monthly.where(np.abs(sum_monthly) > 0.0)
        

    # calculate month of maximum value
    month_of_max = get_monthly_max(regridded_monthly[f"{attributes['out_var']}_monthly"]).to_dataset(
        name=f"{attributes['out_var']}_month_of_max"
    )

    # get climatology
    climatology_ds = get_ilamb_climatology(
        regridded_monthly,
        attributes['out_var'],
        get_conversion_factor(attributes["area_conversion_factor"],
                                      convert_intrinsic=False)
    )

    # return all files combined
    return xr.merge([month_of_max, regridded_annual, climatology_ds])


def get_ilamb_climatology(
    monthly_mean: xr.DataArray, out_var: str, area_cf) -> xr.Dataset:
    """Returns dataset of climatology of the monthly input data

    Args:
        monthly_mean (xr.DataArray): monthly mean values
        out_var (str): name of output variable
        area_cf (float): area conversion factor

    Returns:
        xr.Dataset: climatology dataset
    """
    
    # sum up to get areas where there is no data
    sum_monthly = monthly_mean[f"{out_var}_monthly"].sum(dim='month')
    
    land_area = monthly_mean.land_area
    land_area = xr.where(np.abs(sum_monthly) > 0.0, land_area, 0.0)
    
    if area_cf == 'intrinsic':
        area_cf = 1 / land_area.sum(dim=['lat', 'lon']).values

    # weight by landarea
    area_weighted = land_area * monthly_mean[f"{out_var}_monthly"]

    # calculate area mean
    climatology = area_cf * area_weighted.sum(dim=['lat', 'lon']).to_dataset(name=f"{out_var}_cycle")

    # calculate anomaly
    climatology_mean = climatology[f"{out_var}_cycle"].mean(dim="month")
    monthly_anomaly = (
        climatology[f"{out_var}_cycle"] - climatology_mean
    ).to_dataset(name=f"{out_var}_anomaly")

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
    config: dict, ilamb_dict: dict, attributes: dict, 
) -> tuple[xr.Dataset, str]:
    """Handles reading or computing different types of ILAMB datasets

    Args:
        config (dict): Configuration containing top_dir, clobber, out_dir,
            start_date, end_date.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        attributes (dict): dictionary with attributes about this ILAMB dataset


    Returns:
        tuple[xr.Dataset, str]: output dataset and string for original file
    """

    # create the filter_options dictionary
    filter_options = {
        "tstart": config.get("start_date"),
        "tstop": config.get("end_date"),
        "max_val": get_filter_values(attributes.get("max_val")),
        "min_val": get_filter_values(attributes.get("min_val")),
    }

    if attributes["out_var"] == "ef":
        le_dict, sh_dict = (
            ilamb_dict[f"{attributes['model']}_LE"],
            ilamb_dict[f"{attributes['model']}_SH"],
        )
        ilamb_dat = get_ef_ds(
            config['top_dir'],
            attributes["in_var"],
            le_dict,
            sh_dict,
            filter_options,
        )
        original_file = build_file_paths(
            config['top_dir'],
            [le_dict["sub_dir"], sh_dict["sub_dir"]],
            attributes["model"],
            [le_dict["filename"], sh_dict["filename"]],
        )

    elif attributes["out_var"] == "albedo":
        rsds_dict, rsus_dict = (
            ilamb_dict[f"{attributes['model']}_RSDS"],
            ilamb_dict[f"{attributes['model']}_FSR"],
        )
        ilamb_dat = get_albedo_ds(
            config['top_dir'],
            attributes["in_var"],
            rsds_dict,
            rsus_dict,
            filter_options,
        )
        original_file = build_file_paths(
            config['top_dir'],
            [rsds_dict["sub_dir"], rsus_dict["sub_dir"]],
            attributes["model"],
            [rsds_dict["filename"], rsus_dict["filename"]],
        )

    else:
        # create the file_info dictionary
        file_info = {
            "top_dir": config['top_dir'],
            "sub_dir": attributes["sub_dir"],
            "model": attributes["model"],
            "filename": attributes["filename"],
        }

        # read dataset
        ilamb_dat = get_ilamb_ds(file_info, attributes["in_var"], filter_options)

        # construct original file name
        original_file = build_file_paths(
            config['top_dir'], attributes["sub_dir"], attributes["model"], attributes["filename"]
        )

    return ilamb_dat, original_file

def get_filter_values(input_string: str) -> float:
    """Return values to filter to based on an input string

    Args:
        input_string (str): input string

    Returns:
        float: value to filter to
    """

    if input_string != "None":
        return float(input_string)
    return None


def get_ilamb_ds(
    file_info: Dict[str, str], in_var: str, filter_options: Dict[str, Optional[float]]
) -> xr.Dataset:
    """Reads in a raw dataset from the ILAMB data repository and potentially filters
    on date and min/max value

    Args:
        file_info (dict): Dictionary containing 'top_dir', 'sub_dir', 'model', and 'filename'.
        in_var (str): Name of the variable to extract.
        model (str): model
        filter_options (dict): Dictionary containing optional filtering parameters:
            - 'tstart' (str, optional): Start time to filter to.
            - 'tstop' (str, optional): End time to filter to.
            - 'max_val' (float, optional): Maximum value filter.
            - 'min_val' (float, optional): Minimum value filter.

    Returns:
        xr.Dataset: ILAMB dataset
    """

    # construct file path
    file_path = os.path.join(
        file_info["top_dir"],
        file_info["sub_dir"],
        file_info["model"],
        file_info["filename"],
    )
    # read dataset
    raw_ds = xr.open_dataset(file_path)

    # apply filtering
    filtered_ds = filter_dataset(raw_ds, in_var, filter_options)

    return filtered_ds


def filter_dataset(
    ds: xr.Dataset, var: str, filter_options: Dict[str, Optional[float]]
) -> xr.Dataset:
    """Filters a dataset based on time range and value constraints.

    Args:
        ds (xr.Dataset): Input dataset.
        var (str): Variable to filter on.
        filter_options (dict): Dictionary containing optional filtering parameters:
            - 'tstart' (str, optional): Start time for filtering.
            - 'tstop' (str, optional): End time for filtering.
            - 'max_val' (float, optional): Maximum value constraint.
            - 'min_val' (float, optional): Minimum value constraint.

    Returns:
        xr.Dataset: filtered dataset
    """

    # filter by min/max
    if filter_options.get("max_val") is not None:
        ds = ds.where(ds[var] <= filter_options["max_val"])
    if filter_options.get("min_val") is not None:
        ds = ds.where(ds[var] >= filter_options["min_val"])

    # subset by time
    if (
        filter_options.get("tstart") is not None
        and filter_options.get("tstop") is not None
    ):
        ds = ds.sel(time=slice(filter_options["tstart"], filter_options["tstop"]))

    return ds


def get_ef_ds(
    top_dir: str,
    in_var: str,
    le_dict: dict,
    sh_dict: dict,
    filter_options: Dict[str, Optional[float]],
) -> xr.Dataset:
    """Calculate evaporative fraction for a given ILAMB dataset (LE and SH)

    Args:
        top_dir (str): top ILAMB directory
        in_var (str): variable name to set to
        le_dict (dict): dictionary with information about latent heat data
        sh_dict (dict): dictionary with information about sensible heat data
        filter_options (dict): Dictionary containing optional filtering parameters:
            - 'tstart' (str, optional): Start time for filtering.
            - 'tstop' (str, optional): End time for filtering.
            - 'max_val' (float, optional): Maximum value constraint.
            - 'min_val' (float, optional): Minimum value constraint.

    Returns:
        xr.Dataset: evaporative fraction dataset
    """

    # read in datasets
    le_path = os.path.join(
        top_dir, le_dict["sub_dir"], le_dict["model"], le_dict["filename"]
    )
    sh_path = os.path.join(
        top_dir, sh_dict["sub_dir"], sh_dict["model"], sh_dict["filename"]
    )
    le = xr.open_dataset(le_path)
    sh = xr.open_dataset(sh_path)

    # filter by date and min/max
    le = filter_dataset(le, le_dict["in_var"], filter_options)
    sh = filter_dataset(sh, sh_dict["in_var"], filter_options)

    # calculate evaporative fraction
    ef = evapfrac(sh[sh_dict["in_var"]], le[le_dict["in_var"]], 20).to_dataset(
        name=in_var
    )

    return ef


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


def get_albedo_ds(
    top_dir: str,
    in_var: str,
    rsds_dict: dict,
    rsus_dict: dict,
    filter_options: Dict[str, Optional[float]],
) -> xr.Dataset:
    """Calculate albedo for a given ILAMB dataset (RSDS and RSUS)

    Args:
        top_dir (str): top ILAMB directory
        in_var (str): variable name to set to
        le_dict (dict): dictionary with information about rsds data
        sh_dict (dict): dictionary with information about rsus data
        filter_options (dict): Dictionary containing optional filtering parameters:
            - 'tstart' (str, optional): Start time for filtering.
            - 'tstop' (str, optional): End time for filtering.
            - 'max_val' (float, optional): Maximum value constraint.
            - 'min_val' (float, optional): Minimum value constraint.

    Returns:
        xr.Dataset: evaporative fraction dataset
    """

    # read in datasets
    rsds_path = os.path.join(
        top_dir, rsds_dict["sub_dir"], rsds_dict["model"], rsds_dict["filename"]
    )
    rsus_path = os.path.join(
        top_dir, rsus_dict["sub_dir"], rsus_dict["model"], rsus_dict["filename"]
    )

    rsds = xr.open_dataset(rsds_path)
    rsus = xr.open_dataset(rsus_path)

    # filter by date and min/max
    rsds = filter_dataset(rsds, rsds_dict["in_var"], filter_options)
    rsus = filter_dataset(rsus, rsus_dict["in_var"], filter_options)

    # calculate albedo
    alb = albedo(rsds[rsds_dict["in_var"]], rsus[rsus_dict["in_var"]], 10).to_dataset(
        name=in_var
    )

    return alb


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
    rsus = rsus.where(rsus > energy_threshold)
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

    dlons = earth_radius * (lons[1:] - lons[:-1])
    dlats = earth_radius * (np.sin(lats[1:]) - np.sin(lats[:-1]))
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
    else:
        annual = ds[in_var].isel(time=0).drop_vars(["time"])

    annual_ds = annual.to_dataset(name=out_var)
    annual_ds[out_var].attrs["units"] = metadata["units"]
    annual_ds[out_var].attrs["long_name"] = metadata["longname"]

    # add some global attributes
    annual_ds.attrs["Original"] = metadata["original_file"]
    annual_ds.attrs["Date"] = str(date.today())
    annual_ds.attrs["Author"] = metadata["user"]

    return annual_ds


def regrid_ilamb_ds(
    ds: xr.Dataset,
    target_grid: xr.Dataset,
    var: str,
    method: str = "bilinear",
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

    
    # fill NAs with 0
    ds[var] = ds[var].fillna(0)
    
    # regrid
    regridder = xe.Regridder(ds, target_grid, method)
    ds_regrid = regridder(ds)
    ds_regrid[var] = ds_regrid[var] * target_grid.landmask
    ds_regrid[var].attrs = ds[var].attrs

    ds_regrid.attrs = ds.attrs

    return ds_regrid


def extract_land_area(land_area_file: str) -> xr.Dataset:
    """Extracts land area information from a CLM history file.

    Args:
        land_area_file (str): Path to the file containing land area data.

    Returns:
        xr.Dataset: Dataset containing land area information.
    """
    ds_grid = xr.open_dataset(land_area_file)
    land_area = (ds_grid.landfrac * ds_grid.area).to_dataset(name="land_area")
    land_area["land_area"].attrs["units"] = "km2"
    return land_area


def compile_ilamb_datasets(out_dir: str, ilamb_dict: dict, land_area: xr.DataArray) -> xr.Dataset:
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
    ds_out['land_area'] = land_area

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
        ds = xr.open_dataset(file_name)
        processed_ds = get_average_and_iav(ds, var) if var != "biomass" else ds[var].to_dataset(name=var)
        processed_ds[f"{var}_cycle"] = ds[f"{var}_cycle"]
        processed_ds[f"{var}_month_of_max"] = ds[f"{var}_month_of_max"]
        processed_ds[f"{var}_anomaly"] = ds[f"{var}_anomaly"]
        datasets.append(processed_ds)

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

def get_conf_dict(config_file: str) -> dict:
    """Returns a conversion dictionary from a config file path

    Args:
        config_file (str): path to config file

    Returns:
        dict: dictionary with information about plotting ILAMB data
    """

    conversion_dict = config_to_dict(config_file)
    for _, attributes in conversion_dict.items():
        attributes["models"] = [
            model.strip() for model in attributes["models"].split(",")
        ]
        if attributes["conversion_factor"] == "None":
            attributes["conversion_factor"] = None
        else:
            attributes["conversion_factor"] = float(attributes["conversion_factor"])
        attributes['diverging_cmap'] = str_to_bool(attributes['diverging_cmap'])
            
    return conversion_dict

