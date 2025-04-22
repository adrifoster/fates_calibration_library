"""Functions to assist with global analysis of observations and model outputs"""

import functools
import glob
import os
from datetime import date
import xarray as xr
import numpy as np
import pandas as pd


def month_difference(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """Calculate the minimum cyclic difference between two xarray DataArrays of months.

    Args:
        da1 (xr.DataArray): first data array
        da2 (xr.DataArray): second data array

    Returns:
        xr.DataArray: output data array
    """

    diff = da2 - da1
    return xr.where(diff > 6, diff - 12, xr.where(diff < -6, diff + 12, diff))


def adjust_lon(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """Adjusts the longitude values of a dataset to be from 0-360 to -180 to 180

    Args:
        ds (xr.Dataset): Dataset
        lon_name (str): name of the longitude variable

    Returns:
        xr.Dataset: Dataset with the longitude values changes
    """

    # adjust lon values to make sure they are within (-180, 180)
    ds["longitude_adjusted"] = xr.where(
        ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name]
    )

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds.swap_dims({lon_name: "longitude_adjusted"})
        .sel(**{"longitude_adjusted": sorted(ds.longitude_adjusted)})
        .drop_vars(lon_name)
    )

    ds = ds.rename({"longitude_adjusted": lon_name})

    return ds


def get_files(hist_dir: str, hstream="h0") -> list[str]:
    """Returns all clm history files in a directory given an input hstream

    Args:
        hist_dir (str): directory
        hstream (str, optional): history level. Defaults to 'h0'.

    Returns:
        list[str]: list of files
    """
    return sorted(glob.glob(f"{hist_dir}/*clm2.{hstream}*.nc"))


def calculate_zonal_mean(
    da: xr.DataArray,
    land_area,
    conversion_factor=None,
) -> xr.DataArray:
    """Calculates zonal (by longitude) mean for an input global data array

    Args:
        da (xr.DataArray): input data array
        land_area (xr.DataArray): land area array
        conversion_factor (float): conversion factor
    Returns:
        xr.DataArray: output data array
    """

    # land area by latitude
    land_area_by_lat = land_area.sum(dim="lon")

    # compute area-weighted sum
    area_weighted = (da * land_area).sum(dim="lon")

    # normalize by land area per latitude
    if conversion_factor is None:
        conversion_factor = 1 / land_area_by_lat

    # convert units
    by_lat = conversion_factor * area_weighted

    return by_lat


def get_sparse_climatology(
    da: xr.DataArray,
    cf_time: float,
    cf_area: float,
    biome: xr.Dataset,
    land_area: xr.DataArray,
) -> xr.DataArray:
    """Calculates mean monthly values for an input dataset and variable for a sparse grid dataset

    Args:
        da (xr.Dataset): input data array
        cf_time (float): conversion factor for time averaging
        cf_area (float): conversion factor for area averaging
        biome (xr.Dataset): Whittaker biome dataset
        land_area (xr.DataArray): land area [km2]

    Returns:
        xr.DataArray: output data array
    """

    months = da["time.daysinmonth"]
    monthly_mean = cf_time * (months * da).groupby("time.month").mean()
    monthly_global = area_mean_from_sparse(
        monthly_mean, biome, "global", cf_area, land_area
    )

    return monthly_global


def calculate_month_of_max(monthly_mean: xr.DataArray) -> xr.DataArray:
    """Calculates the month of the maximum value of a monthly data array

    Args:
        monthly_mean (xr.DataArray): monthly data array

    Returns:
        xr.DataArray: output maximum month
    """

    # calculate the month of the maximum value
    month_of_max = monthly_mean.idxmax(dim="month")

    return month_of_max


def calculate_annual_mean(
    data_array: xr.DataArray, conversion_factor: float = None, new_units: str = ""
) -> xr.DataArray:
    """Calculates annual mean of an input DataArray, applies a conversion factor if supplied

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float): Conversion factor.
        new_units (str, optional): new units, defaults to empty

    Returns:
        xr.DataArray: output DataArray
    """

    months = data_array["time.daysinmonth"]

    if conversion_factor is None:
        conversion_factor = 1 / 365

    annual_mean = (
        conversion_factor * (months * data_array).groupby("time.year").sum()
    )
    annual_mean.name = data_array.name
    if new_units != "":
        annual_mean.attrs["units"] = new_units
    return annual_mean


def calculate_monthly_mean(
    data_array: xr.DataArray, conversion_factor: float = None
) -> xr.DataArray:
    """Calculates monthly mean of an input DataArray, applies a conversion factor

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: output DataArray
    """

    months = data_array["time.daysinmonth"]

    if conversion_factor is None:
        conversion_factor = 1.0

    monthly_mean = (conversion_factor * data_array * months).groupby("time.month").sum(
        dim="time"
    ) / months.groupby("time.month").sum(dim="time")
    monthly_mean.name = data_array.name
    return monthly_mean


def preprocess(data_set: xr.Dataset, data_vars: list[str]) -> xr.Dataset:
    """Preprocesses and xarray Dataset by subsetting to specific variables - to be used on read-in

    Args:
        ds (xr.Dataset): input Dataset

    Returns:
        xr.Dataset: output Dataset
    """

    return data_set[data_vars]


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
        ensemble = int(hist_dir.replace(run_dict['tag'], ''))
        run_dict['ensemble'] = ensemble
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
    tag = '_'.join(dirs[0].split('_')[:-1])
    out_file = os.path.join(run_dict["postp_dir"], f"{tag}_000.nc")
    if os.path.isfile(out_file) and not run_dict.get("clobber", False):
        print(f"File {out_file} for default simulation exists, skipping")
    else:
        ds_default = post_process_ds(os.path.join(run_dict['default_dir'], "lnd", "hist"),
                                        data_vars, biome,run_dict["years"], run_dict=run_dict)
        ds_default["ensemble"] = 0
        ds_default.to_netcdf(out_file)
    return keys_finished


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

    ds["ASA"] = ds.FSR / ds.FSDS.where(ds.FSDS > 0)
    ds["ASA"].attrs["units"] = "unitless"
    ds["ASA"].attrs["long_name"] = "All sky albedo"

    ds["RLNS"] = ds.FLDS - ds.FIRE
    ds["RLNS"].attrs["units"] = "W m-2"
    ds["RLNS"].attrs["long_name"] = "surface net longwave radiation"

    ds["RN"] = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
    ds["RN"].attrs["units"] = "W m-2"
    ds["RN"].attrs["long_name"] = "surface net radiation"

    ds["TempAir"] = ds.TBOT - 273.15
    ds["TempAir"].attrs["units"] = "degrees C"
    ds["TempAir"].attrs["long_name"] = ds["TBOT"].attrs["long_name"]

    ds["Temp"] = ds.TSA - 273.15
    ds["Temp"].attrs["units"] = "degrees C"
    ds["Temp"].attrs["long_name"] = ds["TSA"].attrs["long_name"]

    if run_dict.get("sparse", True):
        ds0 = xr.open_dataset(files[0])
        extras = ["grid1d_lat", "grid1d_lon"]
        for extra in extras:
            ds[extra] = ds0[extra]

    if run_dict.get("ensemble", None) is not None:
        ds["ensemble"] = run_dict['ensemble']

    ds.attrs["Date"] = str(date.today())
    ds.attrs["Original"] = files[0]

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


def area_mean(da: xr.DataArray, cf, land_area: xr.DataArray) -> xr.DataArray:
    """Calculates a global area-weighted mean of a global dataset

    Args:
        da (xr.DataArray): input data array
        cf (_type_): conversion factor
        land_area (xr.DataArray): land area data array

    Returns:
        xr.DataArray: output data array
    """

    # update conversion factor if need be
    if cf is None:
        cf = 1 / land_area.sum(dim=["lat", "lon"]).values

    # weight by landarea
    area_weighted = land_area * da

    # calculate area mean
    weighted_mean = cf * area_weighted.sum(dim=["lat", "lon"])

    return weighted_mean


def compile_global_ensemble(run_dict, out_vars, var_dict, sparse_grid, sparse_land_area,
                            global_land_area):

    # read in ensemble and re-chunk for faster analysis
    files = sorted([os.path.join(run_dict['postp_dir'], f) for f in os.listdir(run_dict['postp_dir'])])
    ensemble_ds = xr.open_mfdataset(
        files, combine="nested", concat_dim=["ensemble"], parallel=True
    )
    ensemble_ds = ensemble_ds.chunk({"gridcell": 20, "ensemble": 20, "time": 20})

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
    annual_maps = get_sparse_maps(annual_means.mean(dim='year'), 
                                  sparse_grid, out_vars, ensemble=True)

    # calculate zonal means (i.e. by latitude)
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
    
    biome = ensemble_ds.isel(ensemble=0).biome.drop_vars('ensemble')
    
    # get climatology
    climatology = get_sparse_area_means(
        monthly_means, "global", out_vars, var_dict, sparse_land_area, biome
    )

    # get area means
    area_means = get_sparse_area_means(
        annual_means, "global", out_vars, var_dict, sparse_land_area, biome
    )
    
    # get mean and iav of area means and concat
    area_means_mean = area_means.mean(dim='year')
    area_means_iav = area_means.var(dim='year')
    
    area_means_out = xr.concat([area_means_mean, area_means_iav], dim="summation_var", data_vars="all")
    area_means_out = area_means_out.assign_coords(summation_var=("summation_var", ['mean', 'iav']))
    
    # write to files
    annual_maps.to_netcdf(os.path.join(run_dict['out_dir'], 
                                       f'{run_dict["ensemble_name"]}_annual_maps.nc'))
    zonal_means.to_netcdf(os.path.join(run_dict['out_dir'], 
                                       f'{run_dict["ensemble_name"]}_zonal_means.nc'))
    climatology.to_netcdf(os.path.join(run_dict['out_dir'], 
                                       f'{run_dict["ensemble_name"]}_climatology.nc'))
    area_means_out.to_netcdf(os.path.join(run_dict['out_dir'], 
                                       f'{run_dict["ensemble_name"]}_area_means.nc'))


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
    ds: xr.Dataset, sparse_grid: xr.Dataset, varlist: list[str], ensemble=False,
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
        var_ds = global_from_sparse(sparse_grid, ds[var], ds, ensemble=ensemble).to_dataset(name=var)
        var_ds[var] = var_ds[var]
        ds_list.append(var_ds)

    return xr.merge(ds_list)


def get_sparse_area_means(ds: xr.Dataset, domain: str, varlist: list[str], var_dict: dict, 
                          land_area: xr.DataArray, biome: xr.DataArray) -> xr.Dataset:
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

def fix_infl_months(inflection_months):

    # check number of months
    if len(inflection_months) == 3:
        return inflection_months  # return as is

    elif len(inflection_months) == 2:
        first, second = inflection_months
        if first <= 6:
            # assume we're missing the last month
            return np.sort(np.append(inflection_months, 12))
        else:
            # assume we're missing the first month
            return np.sort(np.append(1, inflection_months))
    elif len(inflection_months) == 1:
        # assume only have peak
        return np.sort(np.append(inflection_months, (12, 1)))
    else:
        print("ERROR")
        return None

def compute_infl(da, dim='month'):
    
    # compute the first and second derivatives
    da_vals = da.values
    first_diff = np.diff(da_vals, axis=-1) >= 0.0
    second_diff = np.diff(first_diff.astype(int), axis=-1) != 0

    # pad with two False values at the beginning to match original length
    pad_shape = list(second_diff.shape)
    pad_shape[-1] = 2
    pad = np.zeros(pad_shape, dtype=bool)
    padded = np.concatenate([pad, second_diff], axis=-1)

    # construct output DataArray with correct coords
    new_coords = dict(da.coords)
    new_coords[dim] = da[dim]

    infl = xr.DataArray(padded, coords=new_coords, dims=da.dims)
    
    # get months where inflection points occur
    infl_months = da['month'].where(infl).values
    # remove NaNs
    non_nan_months = infl_months[~np.isnan(infl_months)]

    return fix_infl_months(non_nan_months)
        
def get_start_end_slopes(da, infl_months):
    
    start = da.sel(month=slice(infl_months[0].item(), infl_months[1].item()))
    end = da.sel(month=slice(infl_months[1].item(), infl_months[2].item()))
    
    x_start = start['month'].values
    y_start = start.values
    x_end = end['month'].values
    y_end = end.values
    
    # skip NaNs or too-short periods
    if len(x_start) >= 2 and len(x_end) >= 2:
        slope_start = np.polyfit(x_start, y_start, 1)[0]
        slope_end = np.polyfit(x_end, y_end, 1)[0]
    else:
        slope_start = np.nan
        slope_end = np.nan

    return slope_start, slope_end