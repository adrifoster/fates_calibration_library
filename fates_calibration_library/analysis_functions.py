"""Functions to assist with global analysis of observations and model outputs"""

import functools
import glob
import os
from datetime import date
import xarray as xr
import numpy as np

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

    return by_lat.transpose("model", "lat")


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


def get_monthly_max(monthly_mean: xr.DataArray) -> xr.DataArray:
    """Calculates the month of the maximum value of a monthly data array

    Args:
        monthly_mean (xr.DataArray): monthly data array

    Returns:
        xr.DataArray: output maximum month
    """

    # get sum so that we can get rid of anything less than 0.0
    month_sum = monthly_mean.sum(dim="month")

    # calculate the month of the maximum value
    month_of_max = monthly_mean.idxmax(dim="month")
    month_of_max = month_of_max.where(np.abs(month_sum) > 0.0)

    return month_of_max


def calculate_annual_mean(
    data_array: xr.DataArray, conversion_factor: float=None, new_units: str = ""
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
        conversion_factor = 1 / months.groupby("time.year").sum()
    
    annual_mean = conversion_factor * (months * data_array).groupby("time.year").sum()
    annual_mean.name = data_array.name
    if new_units != "":
        annual_mean.attrs["units"] = new_units
    return annual_mean


def calculate_monthly_mean(
    data_array: xr.DataArray, conversion_factor: float=None
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
            ensemble (bool, optional): is it an ensemble run? Defaults to False
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
        years = np.unique(ds.time.dt.year)
        last_n = years[-filter_nyears:]
        ds = ds.sel(time=slice(f"{last_n[0]}-01-01", f"{last_n[-1]}-12-31"))
        ds["time"] = xr.cftime_range(str(years[0]), periods=len(ds.time), freq="MS")

    ds = ds.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))

    return ds


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
            ensemble (bool, optional): is it an ensemble run? Defaults to False

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

    if run_dict.get("enemble", False):
        ds["ensemble"] = os.path.basename(files[0]).split("_")[-1]

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
    if cf == "intrinsic":
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
    if cf == "intrinsic":
        cf = 1 / land_area.sum(dim=["lat", "lon"]).values

    # weight by landarea
    area_weighted = land_area * da

    # calculate area mean
    weighted_mean = cf * area_weighted.sum(dim=["lat", "lon"])

    return weighted_mean


def global_from_sparse(
    sparse_grid: xr.Dataset, da: xr.DataArray, ds: xr.Dataset
) -> xr.DataArray:
    """Creates a global map from an input sparse grid in a "paint by numbers" method

    Args:
        sparse_grid (xr.Dataset): input sparse grid cluster file
        da (xr.DataArray): input data array to change to global
        ds (xr.Dataset): sparse grid dataset

    Returns:
        xr.DataArray: output global data array
    """

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

def apply_to_vars(ds: xr.Dataset, varlist: list[str], func, *args, **kwargs) -> xr.Dataset:
    """Applies a function to each variable in varlist and merges results.

    Args:
        ds (xr.Dataset): Input dataset.
        varlist (list[str]): List of variables to process.
        func (callable): Function to apply to each variable.
        *args: Additional positional arguments for the function.
        **kwargs: Additional keyword arguments for the function.

    Returns:
        xr.Dataset: Merged dataset with processed variables.
    """
    ds_list = [func(ds[var], *args, **kwargs).to_dataset(name=var) for var in varlist]
    return xr.merge(ds_list)

def get_zonal_means(
    ds: xr.Dataset, varlist: list[str], var_dict: dict, land_area: xr.DataArray
) -> xr.Dataset:
    """Calculates zonal weighted area means (by latitude) for a range of input variables

    Args:
        ds (xr.Dataset): input dataset
        varlist (list[str]): list of variables to compute
        var_dict (dict): dictionary with information about the variables
        land_area (xr.DataArray): land area [km2]

    Returns:
        xr.Dataset: output dataset
    """
    cf_area_dict = {var: var_dict[var]["cf_area"] if var_dict[var]["cf_area"] is not None else 1.0 for var in varlist}

    return apply_to_vars(ds, varlist, calculate_zonal_mean, land_area, cf_area_dict)

def get_sparse_maps(
    ds: xr.Dataset, sparse_grid: xr.Dataset, varlist: list[str]
) -> xr.Dataset:
    """Gets a dataset of global maps of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        sparse_grid (xr.Dataset): sparse grid file
        varlist (list[str]): list of variables

    Returns:
        xr.Dataset: output dataset
    """

    # loop through each variable and map from sparse to global
    ds_list = []
    for var in varlist:
        var_ds = global_from_sparse(sparse_grid, ds[var], ds).to_dataset(name=var)
        var_ds[var] = var_ds[var]
        ds_list.append(var_ds)

    return xr.merge(ds_list)



def get_annual_means(ds: xr.Dataset, varlist: list[str], var_dict: dict, sparse: bool=False):
    ds_list = []
    for var in varlist:
        ds_list.append(
            calculate_annual_mean(
                ds[var], var_dict[var]["cf_time"], var_dict[var]["annual_units"]
            ).to_dataset(name=var)
        )

    ds_out = xr.merge(ds_list)
    if sparse:
        ds_out["grid1d_lat"] = ds.grid1d_lat
        ds_out["grid1d_lon"] = ds.grid1d_lon

    return ds_out


def get_monthly_means(ds, varlist, var_dict, sparse=False):
    ds_list = []
    for var in varlist:
        if var_dict[var]["cf_time"] == 1 / 365:
            cf_time = 1.0
        else:
            cf_time = var_dict[var]["cf_time"]
        ds_list.append(calculate_monthly_mean(ds[var], cf_time).to_dataset(name=var))

    ds_out = xr.merge(ds_list)
    if sparse:
        ds_out["grid1d_lat"] = ds.grid1d_lat
        ds_out["grid1d_lon"] = ds.grid1d_lon

    return ds_out


def get_area_means(ds, varlist, var_dict, land_area):
    ds_list = []
    for var in varlist:
        ds_list.append(
            area_mean(ds[var], var_dict[var]["cf_area"], land_area).to_dataset(name=var)
        )

    return xr.merge(ds_list)


def get_month_of_maxes(ds, varlist, sparse=False):
    ds_list = []
    for var in varlist:
        ds_list.append(ds[var].idxmax(dim="month").to_dataset(name=var))

    ds_out = xr.merge(ds_list)

    if sparse:
        ds_out["grid1d_lat"] = ds.grid1d_lat
        ds_out["grid1d_lon"] = ds.grid1d_lon

    return ds_out


def get_sparse_area_means(ds, domain, varlist, var_dict, land_area, biome):
    ds_list = []
    for var in varlist:
        ds_list.append(
            area_mean_from_sparse(
                ds[var], biome, domain, var_dict[var]["cf_area"], land_area
            ).to_dataset(name=var)
        )

    return xr.merge(ds_list)
