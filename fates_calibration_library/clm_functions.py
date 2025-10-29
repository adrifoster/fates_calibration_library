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
    years: list[int],
    run_dict: dict={},
    whittaker_ds: xr.Dataset=None,
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

    # assign default values if not there
    sparse = run_dict.get("sparse", True)
    filter_years = run_dict.get("filter_years", None)

    # read in dataset and calculate/convert units on some variables
    ds = get_clm_ds(
        get_files(hist_dir),
        data_vars,
        years[0],
        run_dict,
    )

    # add Whittaker biomes if we are doing a "sparse" run
    if sparse and whittaker_ds is not None:
        ds["biome"] = whittaker_ds.biome
        ds["biome_name"] = whittaker_ds.biome_name

    # filter on years
    if filter_years is not None:
        ds = ds.sel(time=slice(f"{filter_years[0]}-01-01", f"{filter_years[-1]}-12-31"))
        ds["time"] = xr.cftime_range(str(years[0]), periods=len(ds.time), freq="MS")

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
    run_dict: dict, data_vars: list[str], biome: xr.DataArray=None
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
        biome (xr.DataArray, optional): Whittaker biome dataset. Defaults to None.

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
            run_dict["years"],
            run_dict=run_dict,
            whittaker_ds=biome,
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
    out_file = os.path.join(run_dict["postp_dir"], f"{run_dict['tag']}000.nc")
    if os.path.isfile(out_file) and not run_dict.get("clobber", False):
        print(f"File {out_file} for default simulation exists, skipping")
        keys_finished.append(0)
    else:
        ds_default = post_process_ds(
            os.path.join(run_dict["default_dir"], "lnd", "hist"),
            data_vars,
            run_dict["years"],
            run_dict=run_dict,
            whittaker_ds=biome
        )
        ds_default["ensemble"] = 0
        ds_default.to_netcdf(out_file)
        keys_finished.append(0)
    return keys_finished

def aggregate_ensembles(run_dict, out_vars, var_dict, sparse_grid, sparse_land_area,
                       global_land_area):
    
    out_dir = os.path.join(run_dict["postp_dir"], 'aggregated')
    os.makedirs(out_dir, exist_ok=True)
    
    files = sorted(
        [
            os.path.join(run_dict["postp_dir"], f)
            for f in os.listdir(run_dict["postp_dir"]) if f.endswith('.nc')
        ]
    )
    for file in files:
        
        tag = os.path.basename(file).replace('.nc', '')
        ds = xr.open_dataset(file)
        
        # calculate monthly and annual means
        annual_means = get_annual_means(ds, out_vars, var_dict)
        monthly_means = get_monthly_means(ds, out_vars, var_dict)
        
        # remap annual means to whole globe
        annual_maps_filename = os.path.join(
            out_dir, f'{tag}_annual_maps.nc'
        )
        if os.path.isfile(annual_maps_filename) and not run_dict.get("clobber", False):
            print(f"File {annual_maps_filename} exists, skipping")
        else:
            annual_maps = get_sparse_maps(
                annual_means.mean(dim="year"), sparse_grid, out_vars, ensemble=False
            )
            annual_maps['ensemble'] = ds.ensemble
            annual_maps.to_netcdf(annual_maps_filename)
            
        # calculate zonal means (i.e. by latitude)
        zonal_means_filename = os.path.join(
            out_dir, f'{tag}_zonal_means.nc'
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
            zonal_means['ensemble'] = ds.ensemble
            zonal_means.to_netcdf(zonal_means_filename)
            
        # get climatology
        climatology_filename = os.path.join(
            out_dir, f'{tag}_climatology.nc'
        )
        if os.path.isfile(climatology_filename) and not run_dict.get("clobber", False):
            print(f"File {climatology_filename} exists, skipping")
        else:
            climatology = get_sparse_area_means(
                monthly_means, "global", out_vars, var_dict, sparse_land_area, ds.biome,
            )
            climatology['ensemble'] = ds.ensemble
            climatology.to_netcdf(climatology_filename)
            
        # get area means
        area_means_filename = os.path.join(
            out_dir, f'{tag}_area_means.nc'
        )
        if os.path.isfile(area_means_filename) and not run_dict.get("clobber", False):
            print(f"File {area_means_filename} exists, skipping")
        else:
            area_means = get_sparse_area_means(
                annual_means, "global", out_vars, var_dict, sparse_land_area, ds.biome
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
            area_means_out['ensemble'] = ds.ensemble
            area_means_out.to_netcdf(area_means_filename)
            
        biome_area_means_filename = os.path.join(
            out_dir, f'{tag}_biome_area_means.nc'
        )
        if os.path.isfile(biome_area_means_filename) and not run_dict.get("clobber", False):
            print(f"File {biome_area_means_filename} exists, skipping")
        else:
            biome_area_means = get_sparse_area_means(
                annual_means, "biome", out_vars, var_dict, sparse_land_area, ds.biome
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
            biome_area_means_out['ensemble'] = ds.ensemble
            biome_area_means_out.to_netcdf(biome_area_means_filename)
        
def get_annual_means(ds, out_vars, var_dict):
    
    # calculate annual means
    annual_means = apply_to_vars(
        ds,
        out_vars,
        func=calculate_annual_mean,
        add_sparse=True,
        conversion_factor={
            var: var_dict[var]["time_conversion_factor"] for var in out_vars
        },
        new_units={var: var_dict[var]["annual_units"] for var in out_vars},
    )
    
    sh = annual_means['FSH']
    le = annual_means['EFLX_LH_TOT']
    sh = sh.where((sh > 0.0) & (le > 0.0) & ((le + sh) > 0.0))
    le = le.where((sh > 0.0) & (le > 0.0) & ((le + sh) > 0.0))
    fsr = annual_means["FSR"].where(annual_means["FSDS"] > 0.0)
    fsds = annual_means["FSDS"].where(annual_means["FSDS"] > 0.0)
    
    annual_means["ASA"] = fsr/fsds
    annual_means["EF"] = le/(sh + le)
    
    return annual_means

def get_monthly_means(ds, out_vars, var_dict):
    
    # calculate monthly means
    monthly_means = apply_to_vars(
        ds,
        out_vars,
        func=calculate_monthly_mean,
        add_sparse=True,
        conversion_factor={
            var: var_dict[var]["time_conversion_factor"] for var in out_vars
        },
    )
    return monthly_means

def compile_by_tag(dir, tag, remove_biome=False):
    
    files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(tag)])
    if remove_biome:
        files = [f for f in files if not f.endswith('biome_area_means.nc')]
    ensemble_ds = xr.open_mfdataset(
        files, combine="nested", concat_dim=["ensemble"], parallel=True
    )
    return ensemble_ds
    
def compile_global_ensemble(run_dict):

    aggregated_dir = os.path.join(run_dict['postp_dir'], 'aggregated')
    
    # annual maps
    file_name = os.path.join(run_dict['out_dir'], 
                             f"{run_dict['ensemble_name']}_annual_maps.nc")
    if os.path.isfile(file_name) and not run_dict.get("clobber", False):
        print(f"File {file_name} exists, skipping")
    else:
        annual_maps = compile_by_tag(aggregated_dir, 'annual_maps.nc')
        annual_maps.to_netcdf(file_name)
    
    # area means
    file_name = os.path.join(run_dict['out_dir'], 
                             f"{run_dict['ensemble_name']}_area_means.nc")
    if os.path.isfile(file_name) and not run_dict.get("clobber", False):
        print(f"File {file_name} exists, skipping")
    else:
        annual_maps = compile_by_tag(aggregated_dir, 'area_means.nc', remove_biome=True)
        annual_maps.to_netcdf(file_name)
    
    # biome area means
    file_name = os.path.join(run_dict['out_dir'], 
                             f"{run_dict['ensemble_name']}_biome_area_means.nc")
    if os.path.isfile(file_name) and not run_dict.get("clobber", False):
        print(f"File {file_name} exists, skipping")
    else:
        annual_maps = compile_by_tag(aggregated_dir, 'biome_area_means.nc')
        annual_maps.to_netcdf(file_name)
    
    # climatology
    file_name = os.path.join(run_dict['out_dir'], 
                             f"{run_dict['ensemble_name']}_climatology.nc")
    if os.path.isfile(file_name) and not run_dict.get("clobber", False):
        print(f"File {file_name} exists, skipping")
    else:
        annual_maps = compile_by_tag(aggregated_dir, 'climatology.nc')
        annual_maps.to_netcdf(file_name)
    
    # zonal means
    file_name = os.path.join(run_dict['out_dir'], 
                             f"{run_dict['ensemble_name']}_zonal_means.nc")
    if os.path.isfile(file_name) and not run_dict.get("clobber", False):
        print(f"File {file_name} exists, skipping")
    else:
        annual_maps = compile_by_tag(aggregated_dir, 'zonal_means.nc')
        annual_maps.to_netcdf(file_name)
    
def compile_pft_ensemble(
    run_dict, out_vars, var_dict
):

    # read in ensemble
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
    
    annual_means_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_annual_means.nc'
    )
    if os.path.isfile(annual_means_filename) and not run_dict.get("clobber", False):
        print(f"File {annual_means_filename} exists, skipping")
    else:

        # calculate annual means
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
        
        sh = annual_means['FSH']
        le = annual_means['EFLX_LH_TOT']
        sh = sh.where((sh > 0.0) & (le > 0.0) & ((le + sh) > 0.0))
        le = le.where((sh > 0.0) & (le > 0.0) & ((le + sh) > 0.0))
        fsr = annual_means["FSR"].where(annual_means["FSDS"] > 0.0)
        fsds = annual_means["FSDS"].where(annual_means["FSDS"] > 0.0)
        
        annual_means["ASA"] = fsr/fsds
        annual_means["EF"] = le/(sh + le)
        
        # average by year
        mean_dat = []
        for var in out_vars:
            mean_dat.append(annual_means[var].mean(dim='year'))
        mean_dat.append(annual_means['grid1d_lat'])
        mean_dat.append(annual_means['grid1d_lon'])
        annual_means_mean = xr.merge(mean_dat)
        
        annual_means_mean.to_netcdf(annual_means_filename)
        
    monthly_means_filename = os.path.join(
        run_dict["out_dir"], f'{run_dict["ensemble_name"]}_monthly_means.nc'
    )
    if os.path.isfile(monthly_means_filename) and not run_dict.get("clobber", False):
        print(f"File {monthly_means_filename} exists, skipping")
    else:
    
        # calculate monthly means
        monthly_means = apply_to_vars(
            ensemble_ds,
            out_vars,
            func=calculate_monthly_mean,
            add_sparse=True,
            conversion_factor={
                var: var_dict[var]["time_conversion_factor"] for var in out_vars
            },
        )
        monthly_means.to_netcdf(monthly_means_filename)
    

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

def get_pft_grids(land_mask_file, mesh_file, pft):
        
    mesh = xr.open_dataset(mesh_file)
    mesh = mesh.where(mesh.elementMask == 1, drop=True)
    
    centerCoords = mesh.centerCoords.values
    grids = mesh.elementCount.values

    mesh_lats = [coord[1] for coord in centerCoords]
    mesh_lons = [coord[0] for coord in centerCoords]

    land_mask = xr.open_dataset(land_mask_file)

    land_mask_pft = land_mask.where(land_mask.pft == pft)
    all_lats = land_mask_pft.lat.values
    all_lons = land_mask_pft.lon.values
    indices = np.argwhere(np.array(~np.isnan(land_mask_pft['landmask'])))
    
    nc_lats = []
    nc_lons = []
    for coord in indices:
        nc_lats.append(all_lats[coord[0]])
        nc_lons.append(all_lons[coord[1]])

    pft_grids = []
    for i in range(len(nc_lats)):
        pft_grids.append(grids[np.argwhere((mesh_lats == nc_lats[i])*(mesh_lons == nc_lons[i]))[0][0]])

    return pft_grids

def attach_land_area(ensemble, pft_grids, target_grid_file, surdat):

    ds_grid = xr.open_dataset(target_grid_file)
    ds0 = ds_grid.isel(time=0)

    land_area = (ds0.landfrac*ds0.area).values
    land_frac = ds0.landfrac.values
    lats = ds0.lat
    lons = ds0.lon

    default = ensemble.isel(ensemble=0)
    grid_lats = default.grid1d_lat
    grid_lons = default.grid1d_lon

    # extract land area at the chosen gridcells
    frac = np.zeros(len(grid_lats))
    area = np.zeros(len(grid_lats))
    for i in range(len(grid_lats)):
        nearest_index_lat = np.abs(lats - grid_lats[i]).argmin()
        nearest_index_lon = np.abs(lons - grid_lons[i]).argmin()
        
        # grab data at correct lat/lon
        area[i] = land_area[nearest_index_lat, nearest_index_lon]
        frac[i] = land_frac[nearest_index_lat, nearest_index_lon]
        
    lats = surdat.lat
    lons = surdat.lon
    pct_lake = surdat.PCT_LAKE
    lake = np.zeros(len(grid_lats))
    for i in range(len(grid_lats)):
        nearest_index_lat = np.abs(lats - grid_lats[i]).argmin()
        nearest_index_lon = np.abs(lons - grid_lons[i]).argmin()
        lake[i] = pct_lake[nearest_index_lat, nearest_index_lon]

    ensemble['land_area'] = xr.DataArray(area, coords={"gridcell": pft_grids})
    ensemble['land_area'].attrs = {'units': ds0.area.attrs['units']}
    
    ensemble['land_frac'] = xr.DataArray(frac, coords={"gridcell": pft_grids})
    ensemble['land_frac'].attrs = {'units': '0-1'}
    
    ensemble['pct_lake'] = xr.DataArray(lake, coords={"gridcell": pft_grids})
    ensemble['pct_lake'].attrs = {'units': '0-1'}

    return ensemble

def get_pft_ensemble(ensemble_file, pft_grids, target_grid_file, surdat):
    
    ensemble = xr.open_dataset(ensemble_file)
    
    ensemble_pft = ensemble.where(ensemble.gridcell.isin(pft_grids), drop=True)

    ensemble_pft = attach_land_area(ensemble_pft, pft_grids, target_grid_file, surdat)

    return ensemble_pft

def weighted_mean(ds: xr.Dataset, var: str):
    """Takes the land area-weighted mean of a variable in a dataset
       Assumes dataset has a 'land_area' variable

    Args:
        ds (xr.Dataset): dataset
        var (str): variable to take average

    Returns:
        xr.DataArray: weighted mean
    """
    
    corrected_var = ds[var]*ds.land_frac
    return ((corrected_var*ds.land_area).sum(dim='gridcell'))/ds.land_area.sum(dim='gridcell')