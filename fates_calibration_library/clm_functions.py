"""Functions to assist with processing of CLM model outputs"""

import glob
import functools
import os
from datetime import date
import pandas as pd
import numpy as np
import xarray as xr

from fates_calibration_library.analysis_functions import (
    preprocess,
    calculate_annual_mean,
    calculate_zonal_mean,
    calculate_monthly_mean,
)


def get_files(hist_dir: str, hstream="h0") -> list[str]:
    """Returns all CLM history files in a directory given an input hstream

    Args:
        hist_dir (str): directory
        hstream (str, optional): history level. Defaults to 'h0'.

    Returns:
        list[str]: list of files
    """
    return sorted(glob.glob(f"{hist_dir}/*clm2.{hstream}*.nc"))


def get_clm_ds(
    files: list[str],
    data_vars: list[str],
    start_year: int,
    run_dict: dict = None,
) -> xr.Dataset:
    """Reads in a CLM dataset and does some initial post-processing

    Args:
        files (list[str]): list of files
        data_vars (list[str]): data variables to read in
        start_year (int): start year
        run_dict (dict, optional): Dictionary describing aspects of the run:
            fates (bool, optional): is it a FATES run? defaults to True.
            sparse (bool, optional): is it a sparse run? Defaults to True.
            ensemble (int, optional): ensemble member. Defaults to None

    Returns:
        xr.Dataset: output dataset
    """

    # create an empty dictionary if not supplied
    if run_dict is None:
        run_dict = {}

    # read in dataset
    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=functools.partial(preprocess, data_vars=data_vars),
        parallel=True,
        autoclose=True,
    )

    # update time
    ds["time"] = xr.cftime_range(str(start_year), periods=len(ds.time), freq="MS")

    if run_dict.get("fates", True):
        ds["GPP"] = ds["FATES_GPP"] * ds["FATES_FRACTION"]  # kg m-2 s-1
        ds["GPP"].attrs["units"] = ds["FATES_GPP"].attrs["units"]
        ds["GPP"].attrs["long_name"] = ds["FATES_GPP"].attrs["long_name"]

        ds["LAI"] = ds["FATES_LAI"] * ds["FATES_FRACTION"]  # m m-2
        ds["LAI"].attrs["units"] = ds["FATES_LAI"].attrs["units"]
        ds["LAI"].attrs["long_name"] = ds["FATES_LAI"].attrs["long_name"]

    else:
        ds["GPP"] = ds["FPSN"] * 1e-6 * 12.011 / 1000.0  # kg m-2 s-1
        ds["GPP"].attrs["units"] = "kg m-2 s-1"
        ds["GPP"].attrs["long_name"] = ds["FPSN"].attrs["long_name"]

        ds["LAI"] = ds["TLAI"]  # m m-2
        ds["LAI"].attrs["units"] = ds["TLAI"].attrs["units"]
        ds["LAI"].attrs["long_name"] = ds["TLAI"].attrs["long_name"]

    sh = ds.FSH
    le = ds.EFLX_LH_TOT
    energy_threshold = 20

    sh = sh.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    le = le.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
    ds["EF"] = le / (le + sh)
    ds["EF"].attrs["units"] = "unitless"
    ds["EF"].attrs["long_name"] = "Evaporative fraction"

    rsds = ds.FSDS.where(ds.FSDS >= 10)
    rsus = ds.FSR.where(ds.FSDS >= 10)
    ds["ASA"] = rsus / rsds
    ds["ASA"].attrs["units"] = "unitless"
    ds["ASA"].attrs["long_name"] = "All sky albedo"

    ds["RLNS"] = ds.FLDS - ds.FIRE
    ds["RLNS"].attrs["units"] = "W m-2"
    ds["RLNS"].attrs["long_name"] = "surface net longwave radiation"

    ds["RN"] = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
    ds["RN"].attrs["units"] = "W m-2"
    ds["RN"].attrs["long_name"] = "surface net radiation"

    ds["Temp"] = ds.TSA - 273.15
    ds["Temp"].attrs["units"] = "degrees C"
    ds["Temp"].attrs["long_name"] = ds["TSA"].attrs["long_name"]

    ds["Precip"] = ds.SNOW + ds.RAIN
    ds["Precip"].attrs["units"] = "mm s-1"
    ds["Precip"].attrs["long_name"] = "total precipitation"

    ds["ET"] = ds.QVEGE + ds.QVEGT + ds.QSOIL
    ds["ET"].attrs["units"] = ds["QVEGE"].attrs["units"]
    ds["ET"].attrs["long_name"] = "evapotranspiration"

    ds["DTR"] = ds.TREFMXAV - ds.TREFMNAV
    ds["DTR"].attrs["units"] = ds["TREFMXAV"].attrs["units"]
    ds["DTR"].attrs["long_name"] = "diurnal temperature range"

    if run_dict.get("sparse", True):
        ds0 = xr.open_dataset(files[0])
        extras = ["grid1d_lat", "grid1d_lon"]
        for extra in extras:
            ds[extra] = ds0[extra]

    if run_dict.get("ensemble", None) is not None:
        ds["ensemble"] = run_dict["ensemble"]

    ds.attrs["Date"] = str(date.today())
    ds.attrs["Original"] = files[0]

    return ds


def check_ensembles_run(key_df: pd.DataFrame, keys_finished: list[str]) -> list[int]:
    """Checks a list of ensemble keys run against a list of ensemble keys that were
    supposed to run and reports any missing ensemble members

    Args:
        key_df (pd.DataFrame): dataframe with ensemble keys to run
        keys_finished (list[str]): list of ensemble keys finished

    Returns:
        list[int]: list of missing keys
    """

    # get set of keys to run in ensemble
    expected = set(np.unique(key_df.key))

    # get set of keys actually run
    ran = set([int(k) for k in keys_finished])

    # check for missing keys
    missing = expected - ran
    if not missing:
        print("All ensemble members were run.")
    else:
        print("The following ensemble members were not run:")
        for m in sorted(missing):
            print(m)
        return list(missing)


def post_process_ds(
    hist_dir: str,
    data_vars: list[str],
    whittaker_ds: xr.Dataset,
    years: list[int],
    run_dict: dict = None,
) -> xr.Dataset:
    """Post-processes a CLM dataset

    Args:
        hist_dir (str): history directory
        data_vars (list[str]): history variables to read in
        whittaker_ds (xr.Dataset): Whittaker biome dataset
        years (list[int]): start and end year of simulation
        run_dict (dict, optional): Dictionary describing aspects of the run:
            fates (bool, optional): is it a FATES run? defaults to True.
            sparse (bool, optional): is it a sparse run? Defaults to True.
            ensemble (int, optional): ensemble member. Default to None.
            filter_nyears (int, optional): How many years to filter at end of simulation.
                Defaults to None.

    Returns:
        xr.Dataset: output dataset
    """

    # assign default values if run_dict is None
    if run_dict is None:
        run_dict = {}
    sparse = run_dict.get("sparse", True)
    filter_nyears = run_dict.get("filter_nyears", None)

    # read in dataset and calculate/convert units on some variables
    ds = get_clm_ds(
        get_files(hist_dir),
        data_vars,
        years[0],
        run_dict,
    )

    # add Whittaker biomes if we are doing a sparse run
    if sparse:
        ds["biome"] = whittaker_ds.biome
        ds["biome_name"] = whittaker_ds.biome_name

    # filter on years
    if filter_nyears is not None:
        mod_years = np.unique(ds.time.dt.year)
        last_n = mod_years[-filter_nyears:]
        ds = ds.sel(time=slice(f"{last_n[0]}-01-01", f"{last_n[-1]}-12-31"))
        ds["time"] = xr.cftime_range(str(years[0]), periods=len(ds.time), freq="MS")

    ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))

    return ds


def area_mean_from_sparse(
    da: xr.DataArray, biome: xr.DataArray, domain: str, cf, land_area: xr.DataArray
) -> xr.DataArray:
    """Calculates an area mean of a sparse grid dataset, either by biome or globally

    Args:
        da (xr.DataArray): input data array
        biome (xr.DataArray): biome data
        domain (str): either "global" or "biome"
        cf (_type_): conversion factor
        land_area (xr.DataArray): land area data array

    Returns:
        xr.DataArray: output data array
    """

    ## update conversion factor if need be
    if cf is None:
        if domain == "global":
            cf = 1 / land_area.sum()
        else:
            cf = 1 / land_area.groupby(biome).sum()

    # weight by landarea
    area_weighted = land_area * da

    # sort out domain groupings
    area_weighted["biome"] = biome
    area_weighted = area_weighted.swap_dims({"gridcell": "biome"})

    if domain == "global":
        grid = 1 + 0 * area_weighted.biome  # every gridcell is in biome 1
    else:
        grid = area_weighted.biome

    # calculate area mean
    weighted_mean = cf * area_weighted.groupby(grid).sum()

    if domain == "global":
        weighted_mean = weighted_mean.mean(dim="biome")  # get rid of gridcell dimension

    return weighted_mean


def post_process_ensemble(
    run_dict: dict, data_vars: list[str], biome: xr.DataArray
) -> list[str]:
    """Create single history files for each set of history files in an ensemble.

    Args:
        run_dict (dict): Dictionary describing aspects of the run:
            top_dir (str): path to top directory with archived ensemble history output
            postp_dir (str): directory where post-processed files will be placed
            years (list[int]): start and end year of simulation
            clobber (bool): whether or not to overwrite files. Defaults to False.
            fates (bool, optional): is it a FATES run? defaults to True.
            sparse (bool, optional): is it a sparse run? Defaults to True.
            ensemble (int, optional): ensemble member. Default to None.
            filter_nyears (int, optional): How many years to filter at end of simulation.
                Defaults to None.
        data_vars (list[str]): list of variables to read in
        biome (xr.DataArray): Whittaker biome dataset

    Returns:
        list[str]: list of ensemble keys successfully post-processed and written out
    """

    # create output directory if it doesn't exist
    os.makedirs(run_dict["postp_dir"], exist_ok=True)

    keys_finished = []
    dirs = sorted(os.listdir(run_dict["top_dir"]))

    for hist_dir in dirs:
        ensemble = int(hist_dir.replace(run_dict["tag"], ""))
        run_dict["ensemble"] = ensemble
        out_file = os.path.join(run_dict["postp_dir"], f"{hist_dir}.nc")

        # skip if file exists and clobber is False
        if os.path.isfile(out_file) and not run_dict.get("clobber", False):
            print(f"File {out_file} for ensemble member {ensemble} exists, skipping")
            keys_finished.append(ensemble)
            continue

        # create history file for this ensemble
        ds_out = post_process_ds(
            os.path.join(run_dict["top_dir"], hist_dir, "lnd", "hist"),
            data_vars,
            biome,
            run_dict["years"],
            run_dict=run_dict,
        )
        # write to file
        if ds_out is not None:
            if (
                len(ds_out.time)
                == (run_dict["years"][1] - run_dict["years"][0] + 1) * 12
            ):
                ds_out.to_netcdf(out_file)
                keys_finished.append(ensemble)

    # also write out default simulation
    tag = "_".join(dirs[0].split("_")[:-1])
    out_file = os.path.join(run_dict["postp_dir"], f"{tag}_000.nc")
    if os.path.isfile(out_file) and not run_dict.get("clobber", False):
        print(f"File {out_file} for default simulation exists, skipping")
    else:
        ds_default = post_process_ds(
            os.path.join(run_dict["default_dir"], "lnd", "hist"),
            data_vars,
            biome,
            run_dict["years"],
            run_dict=run_dict,
        )
        ds_default["ensemble"] = 0
        ds_default.to_netcdf(out_file)
    return keys_finished


def compile_global_ensemble(
    run_dict, out_vars, var_dict, sparse_grid, sparse_land_area, global_land_area
):

    # read in ensemble and re-chunk for faster analysis
    files = sorted(
        [
            os.path.join(run_dict["postp_dir"], f)
            for f in os.listdir(run_dict["postp_dir"])
        ]
    )
    ensemble_ds = xr.open_mfdataset(
        files, combine="nested", concat_dim=["ensemble"], parallel=True
    )
    ensemble_ds = ensemble_ds.chunk({"gridcell": 20, "ensemble": 20, "time": 20})

    biome = ensemble_ds.isel(ensemble=0).biome.drop_vars("ensemble")

    # calculate annual and monthly means
    annual_means = apply_to_vars(
        ensemble_ds,
        out_vars,
        func=calculate_annual_mean,
        add_sparse=True,
        conversion_factor={
            var: var_dict[var]["time_conversion_factor"] for var in out_vars
        },
        new_units={var: var_dict[var]["annual_units"] for var in out_vars},
    )
    annual_means["ASA"] = annual_means["FSR"] / annual_means["FSDS"]
    annual_means["EF"] = annual_means["EFLX_LH_TOT"] / (
        annual_means["EFLX_LH_TOT"] + annual_means["FSH"]
    )

    monthly_means = apply_to_vars(
        ensemble_ds,
        out_vars,
        func=calculate_monthly_mean,
        add_sparse=True,
        conversion_factor={
            var: var_dict[var]["time_conversion_factor"] for var in out_vars
        },
    )

    # remap annual means to whole globe
    annual_maps_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_annual_maps.nc'
    )
    if os.path.isfile(annual_maps_filename) and not run_dict.get("clobber", False):
        print(f"File {annual_maps_filename} exists, skipping")
    else:
        annual_maps = get_sparse_maps(
            annual_means.mean(dim="year"), sparse_grid, out_vars, ensemble=True
        )
        annual_maps.to_netcdf(annual_maps_filename)

    # calculate zonal means (i.e. by latitude)
    zonal_means_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_zonal_means.nc'
    )
    if os.path.isfile(zonal_means_filename) and not run_dict.get("clobber", False):
        print(f"File {zonal_means_filename} exists, skipping")
    else:
        zonal_means = apply_to_vars(
            annual_maps,
            out_vars,
            func=calculate_zonal_mean,
            add_sparse=False,
            land_area=global_land_area,
            conversion_factor={
                var: var_dict[var]["area_conversion_factor"] for var in out_vars
            },
        )
        zonal_means.to_netcdf(zonal_means_filename)

    # get climatology
    climatology_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_climatology.nc'
    )
    if os.path.isfile(climatology_filename) and not run_dict.get("clobber", False):
        print(f"File {climatology_filename} exists, skipping")
    else:
        climatology = get_sparse_area_means(
            monthly_means, "global", out_vars, var_dict, sparse_land_area, biome
        )
        climatology.to_netcdf(climatology_filename)
        
    biome_climatology_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_biome_climatology.nc'
    )
    if os.path.isfile(biome_climatology_filename) and not run_dict.get("clobber", False):
        print(f"File {biome_climatology_filename} exists, skipping")
    else:
        biome_climatology = get_sparse_area_means(
            monthly_means, "biome", out_vars, var_dict, sparse_land_area, biome
        )
        biome_climatology.to_netcdf(biome_climatology_filename)


    # get area means
    area_means_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_area_means.nc'
    )
    if os.path.isfile(area_means_filename) and not run_dict.get("clobber", False):
        print(f"File {area_means_filename} exists, skipping")
    else:
        area_means = get_sparse_area_means(
            annual_means, "global", out_vars, var_dict, sparse_land_area, biome
        )

        # get mean and iav of area means and concat
        area_means_mean = area_means.mean(dim="year")
        area_means_iav = area_means.var(dim="year")

        area_means_out = xr.concat(
            [area_means_mean, area_means_iav], dim="summation_var", data_vars="all"
        )
        area_means_out = area_means_out.assign_coords(
            summation_var=("summation_var", ["mean", "iav"])
        )
        area_means_out.to_netcdf(area_means_filename)
        
    biome_area_means_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_biome_area_means.nc'
    )
    if os.path.isfile(biome_area_means_filename) and not run_dict.get("clobber", False):
        print(f"File {biome_area_means_filename} exists, skipping")
    else:
        biome_area_means = get_sparse_area_means(
            annual_means, "biome", out_vars, var_dict, sparse_land_area, biome
        )

        # get mean and iav of area means and concat
        biome_area_means_mean = biome_area_means.mean(dim="year")
        biome_area_means_iav = biome_area_means.var(dim="year")

        biome_area_means_out = xr.concat(
            [biome_area_means_mean, biome_area_means_iav], dim="summation_var", data_vars="all"
        )
        biome_area_means_out = biome_area_means_out.assign_coords(
            summation_var=("summation_var", ["mean", "iav"])
        )
        biome_area_means_out.to_netcdf(biome_area_means_filename)


def global_from_sparse(
    sparse_grid: xr.Dataset, da: xr.DataArray, ds: xr.Dataset, ensemble: bool = False
) -> xr.DataArray:
    """Creates a global map from an input sparse grid in a "paint by numbers" method

    Args:
        sparse_grid (xr.Dataset): input sparse grid cluster file
        da (xr.DataArray): input data array to change to global
        ds (xr.Dataset): sparse grid dataset
        ensemble (bool): is the dataset an ensemble. Defaults to False.

    Returns:
        xr.DataArray: output global data array
    """

    # grab only one ensemble member to remap
    if ensemble:
        ds = ds.isel(ensemble=0)

    # create empty array
    out = np.zeros(sparse_grid.cclass.shape) + np.nan

    # number of clusters
    num_clusters = len(sparse_grid.numclust)

    # fill empty array with cluster class
    for gridcell, (lon, lat) in enumerate(sparse_grid.rcent_coords):
        i = np.arange(num_clusters)[
            (abs(ds.grid1d_lat - lat) < 0.1) & (abs(ds.grid1d_lon - lon) < 0.1)
        ]
        out[sparse_grid.cclass == gridcell + 1] = i

    # set cluster class
    cluster_class = out.copy()
    cluster_class[np.isnan(out)] = 0

    # create a sparse grid map
    sparse_grid_map = xr.Dataset()
    sparse_grid_map["cluster_class"] = xr.DataArray(
        cluster_class.astype(int), dims=["lat", "lon"]
    )
    sparse_grid_map["notnan"] = xr.DataArray(~np.isnan(out), dims=["lat", "lon"])
    sparse_grid_map["lat"] = sparse_grid.lat
    sparse_grid_map["lon"] = sparse_grid.lon

    # get output map
    out_map = (
        da.sel(gridcell=sparse_grid_map.cluster_class)
        .where(sparse_grid_map.notnan)
        .compute()
    )

    return out_map


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
    target_grid["area"] = ds["area"].fillna(0)
    target_grid["landmask"] = ds["landmask"].fillna(0)
    target_grid["landfrac"] = ds["landfrac"].fillna(0)
    target_grid["land_area"] = target_grid.area * target_grid.landfrac
    target_grid["land_area"] = target_grid["land_area"].where(
        target_grid.lat > -60.0, 0.0
    )

    return target_grid


def apply_to_vars(
    ds: xr.Dataset, varlist: list[str], func, add_sparse: bool, *args, **kwargs
) -> xr.Dataset:
    """Applies a function to each variable in varlist and merges results.

    Args:
        ds (xr.Dataset): Input dataset.
        varlist (list[str]): List of variables to process.
        func (callable): Function to apply to each variable.
        add_sparse (bool): whether or not to add sparse grid
        *args: Positional arguments for the function
        **kwargs: Additional keyword arguments for the function.

    Returns:
        xr.Dataset: Merged dataset with processed variables.
    """

    ds_out = xr.Dataset()
    for var in varlist:

        var_kwargs = {
            key: (val[var] if isinstance(val, dict) and var in val else val)
            for key, val in kwargs.items()
        }
        ds_out[var] = func(ds[var], *args, **var_kwargs)

    if add_sparse:
        ds_out["grid1d_lat"] = ds.grid1d_lat
        ds_out["grid1d_lon"] = ds.grid1d_lon

    return ds_out


def get_sparse_maps(
    ds: xr.Dataset,
    sparse_grid: xr.Dataset,
    varlist: list[str],
    ensemble=False,
) -> xr.Dataset:
    """Gets a dataset of global maps of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        sparse_grid (xr.Dataset): sparse grid file
        varlist (list[str]): list of variables
        ensemble (optional, bool): whether it is an ensemble. defaults to False.

    Returns:
        xr.Dataset: output dataset
    """

    # loop through each variable and map from sparse to global
    ds_list = []
    for var in varlist:
        var_ds = global_from_sparse(
            sparse_grid, ds[var], ds, ensemble=ensemble
        ).to_dataset(name=var)
        var_ds[var] = var_ds[var]
        ds_list.append(var_ds)

    return xr.merge(ds_list)


def get_sparse_area_means(
    ds: xr.Dataset,
    domain: str,
    varlist: list[str],
    var_dict: dict,
    land_area: xr.DataArray,
    biome: xr.DataArray,
) -> xr.Dataset:
    """Gets a dataset of sparse area means of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        domain (str): 'global' or 'biome'
        varlist (list[str]): list of variables
        var_dict (dict): dictionary with information about variables
        land_area (xr.DataArray): land area for sparse grid
        biome (xr.DataArray): whittaker biome dataset

    Returns:
        xr.Dataset: output dataset
    """
    ds_list = []
    for var in varlist:
        ds_list.append(
            area_mean_from_sparse(
                ds[var],
                biome,
                domain,
                var_dict[var]["area_conversion_factor"],
                land_area,
            ).to_dataset(name=var)
        )

    return xr.merge(ds_list)
