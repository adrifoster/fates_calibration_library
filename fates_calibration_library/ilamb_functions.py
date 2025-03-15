"""Methods for processing and analyzing ILAMB data"""

import os
import math
from datetime import date
from typing import Dict, Optional
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
import matplotlib.pyplot as plt

from fates_calibration_library.analysis_functions import month_wts, get_zonal
from fates_calibration_library.plotting_functions import generate_subplots, map_function
from fates_calibration_library.plotting_functions import (
    round_down,
    round_up,
    get_blank_plot,
)


def get_all_ilamb_data(config: Dict, ilamb_dict: Dict, target_grid: xr.Dataset):
    """Processes ILAMB datasets: reads, converts to annual values, regrids, and saves.

    Args:
        config (dict): Configuration containing top_dir, clobber, out_dir, start_date,
                        end_date, and user.
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        target_grid (xr.Dataset): Target grid for regridding.
        clobber (bool): whether to overwrite files.
    """

    # create output directory if it doesn't exist
    os.makedirs(config["out_dir"], exist_ok=True)

    # process each dataset
    for dataset, attributes in ilamb_dict.items():
        process_dataset(config, ilamb_dict, dataset, attributes, target_grid)


def process_dataset(
    config: dict,
    ilamb_dict: dict,
    dataset: str,
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

    model = attributes["model"]
    out_var = attributes["out_var"]
    file_name = f"{model}_{out_var.upper()}.nc"
    out_file = os.path.join(config["out_dir"], file_name)

    # skip if file exists and clobber is False
    if os.path.isfile(out_file) and not config["clobber"]:
        print(f"File {out_file} for {dataset} exists, skipping")
        return

    # read or compute ILAMB data
    ilamb_dat, original_file = read_ilamb_data(
        config["top_dir"], ilamb_dict, model, attributes, config
    )

    # prepare metadata for get_annual_ds
    metadata = {
        "units": attributes["units"],
        "longname": attributes["longname"],
        "original_file": original_file,
        "user": config["user"],
    }
    # convert to annual dataset
    annual_ds = get_annual_ds(
        ilamb_dat,
        attributes["in_var"],
        out_var,
        get_conversion_factor(attributes["conversion_factor"]),
        metadata,
    )

    # Regrid and write to file
    regridded_ds = regrid_ilamb_ds(annual_ds, target_grid, out_var)
    regridded_ds.to_netcdf(out_file, mode="w")


def read_ilamb_data(
    top_dir: str, ilamb_dict: dict, model: str, attributes: dict, config: dict
) -> tuple[xr.Dataset, str]:
    """Handles reading or computing different types of ILAMB datasets

    Args:
        top_dir (str): top ILAMB directory
        ilamb_dict (dict): Dictionary with ILAMB dataset information.
        model (str): model name
        attributes (dict): dictionary with attributes about this ILAMB dataset
        config (dict): Configuration containing top_dir, clobber, out_dir,
        start_date, end_date.

    Returns:
        tuple[xr.Dataset, str]: output dataset and string for original file
    """

    out_var = attributes["out_var"]

    # create the filter_options dictionary
    filter_options = {
        "tstart": config.get("start_date"),
        "tstop": config.get("end_date"),
        "max_val": get_filter_values(attributes.get("max_val")),
        "min_val": get_filter_values(attributes.get("min_val")),
    }

    if out_var == "ef":
        le_dict, sh_dict = ilamb_dict[f"{model}_LE"], ilamb_dict[f"{model}_SH"]
        ilamb_dat = get_ef_ds(
            top_dir,
            attributes["in_var"],
            le_dict,
            sh_dict,
            filter_options,
        )
        original_file = (
            f"{os.path.join(top_dir, le_dict['sub_dir'], model, le_dict['filename'])} and "
            f"{os.path.join(top_dir, sh_dict['sub_dir'], model, sh_dict['filename'])}"
        )

    elif out_var == "albedo":
        rsds_dict, rsus_dict = ilamb_dict[f"{model}_RSDS"], ilamb_dict[f"{model}_FSR"]
        ilamb_dat = get_albedo_ds(
            top_dir,
            attributes["in_var"],
            rsds_dict,
            rsus_dict,
            filter_options,
        )
        original_file = (
            f"{os.path.join(top_dir, rsds_dict['sub_dir'], model, rsds_dict['filename'])} and "
            f"{os.path.join(top_dir, rsus_dict['sub_dir'], model, rsus_dict['filename'])}"
        )

    else:
        # create the file_info dictionary
        file_info = {
            "top_dir": top_dir,
            "sub_dir": attributes["sub_dir"],
            "model": model,
            "filename": attributes["filename"],
        }

        # read dataset
        ilamb_dat = get_ilamb_ds(file_info, attributes["in_var"], filter_options)

        # construct original file name
        original_file = os.path.join(
            top_dir, attributes["sub_dir"], model, attributes["filename"]
        )

    ilamb_dat[attributes['in_var']] = ilamb_dat[attributes['in_var']].fillna(0)
    
    return ilamb_dat, original_file


def get_conversion_factor(input_string: str) -> float:
    """Gets a conversion factor for ILAMB data based on an input string

    Args:
        input_string (str): input string

    Returns:
        float: conversion factor
    """

    if input_string == "intrinsic":
        return 1 / 365.0
    if input_string == "mrro":
        return 24 * 60 * 60 / 365.0
    return float(input_string)


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

    # calculate cell areas
    lons = ds.lon.values
    lats = np.array(sorted(ds.lat.values))
    areas = cell_areas(lats, lons)
    land_area = xr.DataArray(areas, coords={"lat": lats, "lon": lons})

    # calculate annual values
    if len(ds.time) > 1:
        annual = get_annual_obs(ds, in_var, conversion_factor)
    else:
        annual = ds[in_var].isel(time=0).drop_vars(["time"])

    annual_ds = annual.to_dataset(name=out_var)
    annual_ds[out_var].attrs["units"] = metadata["units"]
    annual_ds[out_var].attrs["long_name"] = metadata["longname"]

    # add in land area
    annual_ds["land_area"] = land_area
    annual_ds["land_area"].attrs["units"] = "m2"
    annual_ds["land_area"].attrs["long_name"] = "land area"

    # add some global attributes
    annual_ds.attrs["Original"] = metadata["original_file"]
    annual_ds.attrs["Date"] = str(date.today())
    annual_ds.attrs["Author"] = metadata["user"]

    return annual_ds


def get_annual_obs(ds: xr.Dataset, var: str, conversion_factor: float) -> xr.Dataset:
    """Sums the annual values for a variable, using a conversion factor

    Args:
        ds (xr.Dataset): Dataset
        var (str): variable to sum
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: output annual sum
    """

    # calculate annual values
    nyears = len(np.unique(ds["time.year"]))
    annual = (
        conversion_factor * (ds[var] * month_wts(nyears)).groupby("time.year").sum()
    )

    return annual


def create_target_grid(file: str, var: str) -> xr.Dataset:
    """Creates a target grid to resample to

    Args:
        file (str): path to dataset to regrid to
        var (str): variable to create the grid off of

    Returns:
        xr.Dataset: output dataset
    """

    ds = xr.open_dataset(file)
    target_grid = ds[var].mean(dim="time")
    target_grid['landmask'] = ds['landmask'].fillna(0)
    target_grid['landfrac'] = ds['landfrac'].fillna(0)

    return target_grid


def regrid_ilamb_ds(
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

    # regrid
    regridder = xe.Regridder(ds, target_grid, method)
    ds_regrid = regridder(ds)
    ds_regrid = ds_regrid * target_grid.landmask
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


def compile_ilamb_datasets(
    out_dir: str, ilamb_dict: dict, land_area_file: str
) -> xr.Dataset:
    """Compiles all regridded ILAMB datasets, computes average and variance over years,
    and merges into a single dataset

    Args:
        out_dir (str): output directory where files are located
        ilamb_dict (dict): dictionary with information about ILAMB data
        land_area_file (str): path to CLM history file with land area on it

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

    return xr.merge(compiled_data)


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
        processed_ds = get_average_and_iav(ds, var) if var != "biomass" else ds
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


def plot_global(
    da: xr.DataArray,
    varname: str,
    units: str,
    cmap: str,
    diverging_cmap: bool = False,
):
    """Plots a global data array of ILAMB models, one subplot per model

    Args:
        da (xr.DataArray): data array
        varname (str): variable name for legend
        units (str): units for legend
        cmap (str): colormap to use
        diverging_cmap (bool, optional): whether the colormap is a diverging scale.
                                    Defaults to False.
    """

    vmin = da.min().values
    vmax = da.max().values
    models = da.model.values
    num_plots = len(models)

    # get the emtpy subplots
    figure, axes = generate_subplots(num_plots)

    if num_plots > 1:
        axes = axes.flatten(order="F")
        for idx, ax in enumerate(axes):
            pcm = map_function(
                ax,
                da.sel(model=models[idx]),
                models[idx],
                cmap,
                vmax,
                vmin,
                diverging_cmap=diverging_cmap,
            )
        cbar = figure.colorbar(
            pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation="horizontal"
        )
    else:
        pcm = map_function(
            axes[0],
            da.sel(model=models[0]),
            models[0],
            cmap,
            vmax,
            vmin,
            diverging_cmap=diverging_cmap,
        )
        cbar = figure.colorbar(pcm, ax=axes[0], shrink=0.5, orientation="horizontal")
    cbar.set_label(f"{varname} ({units})", size=10, fontweight="bold")


def plot_by_lat(
    da: xr.DataArray,
    units: str,
    var: str,
    varname: str,
    conversion_config: dict = None,
):
    """Plots zonal (by latitude) ILAMB data for each model

    Args:
        da (xr.DataArray): data array
        units (str): units description for by_latitude data
        var (str): variable plotting
        varname (str): variable name for axes
        conversion_config (dict, optional): configuration dictionary with 'land_area' and
                'conversion_factor'. Defaults to None.
    """

    conversion_config = conversion_config or {
        "land_area": None,
        "conversion_factor": None,
    }

    # get by latitude
    by_lat = get_zonal(
        da,
        units,
        conversion_config["land_area"],
        conversion_config["conversion_factor"],
    )

    # turn into pandas data frame for easier plotting
    df = pd.DataFrame(
        {
            "lat": np.tile(by_lat.lat, len(by_lat.model)),
            "model": np.repeat(by_lat.model, len(by_lat.lat)),
            var: by_lat.values.flatten(),
        }
    )

    # get min/max values
    minval = df[var].min()
    minvar = round_up(np.abs(minval)) * -1.0 if minval < 0 else round_down(minval)
    maxvar = round_up(df[var].max())

    # get a blank plot
    get_blank_plot()

    # add latitude-specific ticks/lines
    plt.xlim(minvar, maxvar)
    plt.ylim(-90, 90)

    plt.yticks(
        range(-90, 91, 15), [str(x) + "º" for x in range(-90, 91, 15)], fontsize=10
    )
    plt.xticks(fontsize=10)

    for lat in range(-90, 91, 15):
        plt.plot(
            range(math.floor(minvar), math.ceil(maxvar) + 1),
            [lat] * len(range(math.floor(minvar), math.ceil(maxvar) + 1)),
            "--",
            lw=0.5,
            color="black",
            alpha=0.3,
        )

    plt.tick_params(bottom=False, top=False, left=False, right=False)

    cols = [
        "#e60049",
        "#0bb4ff",
        "#50e991",
        "#e6d800",
        "#9b19f5",
        "#ffa300",
        "#dc0ab4",
        "#b3d4ff",
        "#00bfa0",
    ]

    # plot models
    for rank, model in enumerate(np.unique(df.model.values)):
        data = df[df.model == model]
        plt.plot(data[var].values, data.lat.values, lw=2, color=cols[rank], label=model)

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {varname} ({units})", fontsize=11)
    plt.title(
        f"Observed Annual {varname}" + " by latitude for different data products",
        fontsize=11,
    )
    plt.legend(loc="upper right")


def plot_ilamb_var(
    ilamb_dat: xr.Dataset,
    var: str,
    plot_config: dict,
):
    """Plots ILAMB data, globally and by latitude, for a variable for all models

    Args:
        ilamb_dat (xr.Dataset): input ILAMB compiled dataset
        var (str): variable to plot
        plot_config (dict): configuration dictionary with keys:
            - models (list[str]): list of ILAMB models
            - conversion_factor (float): conversion factor for going to latitude sums
            - varname (str): variable name for plotting
            - global_units (str): global units for axes
            - lat_units (str): latitude units for axes
            - cmap (str): color map for global plot
            - diverging_cmap (bool): whether the cmap is diverging or not
    """

    # get the data for just this variable
    da = get_model_da(ilamb_dat, var, plot_config["models"])

    # plot globally
    plot_global(
        da,
        plot_config["varname"],
        plot_config["global_units"],
        plot_config["cmap"],
        diverging_cmap=plot_config["diverging_cmap"],
    )

    # get conversion factor
    conversion_factor = plot_config.get("conversion_factor", None)

    # plot by latitude
    if conversion_factor is not None:
        conversion_dict = {
            "land_area": ilamb_dat.land_area,
            "conversion_factor": conversion_factor,
        }
        plot_by_lat(
            da, plot_config["lat_units"], var, plot_config["varname"], conversion_dict
        )
    else:
        plot_by_lat(da, plot_config["lat_units"], var, plot_config["varname"])


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
