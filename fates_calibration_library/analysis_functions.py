"""Functions to assist with global analysis of observations and model outputs"""

import functools
import os
from datetime import date
import xarray as xr
import numpy as np


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


def get_zonal(
    da: xr.DataArray,
    new_units,
    land_area: xr.DataArray = None,
    conversion_factor: float = None,
) -> xr.DataArray:
    """Calculates zonal (by longitude) sum for an input global data array

    Args:
        da (xr.DataArray): input data array
        new_units (str): new units
        land_area (xr.DataArray, optional): land area array. Defaults to None
        conversion_factor (float, optional): conversion factor. Defaults to None.
    Returns:
        xr.DataArray: output data array
    """

    # convert if needed
    if land_area is not None:
        da = da * land_area
    if conversion_factor is not None:
        da = da * conversion_factor

    # sum up by latitude
    by_lat = da.sum(dim="lon")

    # update units
    by_lat.attrs["units"] = new_units

    return by_lat


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
    data_array: xr.DataArray, conversion_factor: float = 1 / 365, new_units: str = ""
) -> xr.DataArray:
    """Calculates annual mean of an input DataArray, applies a conversion factor if supplied

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float, optional): Conversion factor. Defaults to 1/365.
        new_units (str, optional): new units, defaults to empty

    Returns:
        xr.DataArray: output DataArray
    """

    months = data_array["time.daysinmonth"]
    annual_mean = conversion_factor * (months * data_array).groupby("time.year").sum()
    annual_mean.name = data_array.name
    if new_units != "":
        annual_mean.attrs["units"] = new_units
    return annual_mean


def calculate_monthly_mean(
    data_array: xr.DataArray, conversion_factor: float
) -> xr.DataArray:
    """Calculates monthly mean of an input DataArray, applies a conversion factor

    Args:
        da (xr.DataArray): input DataArray
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: output DataArray
    """

    months = data_array["time.daysinmonth"]
    monthly_mean = (
        conversion_factor * (months * data_array).groupby("time.month").mean()
    )
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


def get_clm_ds(
    files: list[str],
    data_vars: list[str],
    start_year: int,
    fates: bool = True,
    sparse: bool = True,
    ensemble: bool = False,
) -> xr.Dataset:
    """Reads in a CLM dataset and does some initial post-processing

    Args:
        files (list[str]): list of files
        data_vars (list[str]): data variables to read in
        start_year (int): start year
        fates (bool, optional): FATES or CLM. Defaults to True.
        sparse (bool, optional): sparse or full grid. Defaults to True.
        ensemble (bool, optional): is the ds part of an ensemble? Defaults to False.

    Returns:
        xr.Dataset: output dataset
    """

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        preprocess=functools.partial(preprocess, data_vars=data_vars),
        parallel=True,
        autoclose=True,
    )

    ds["time"] = xr.cftime_range(str(start_year), periods=len(ds.time), freq="MS")

    if fates:
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

    if sparse:
        ds0 = xr.open_dataset(files[0])
        extras = ["grid1d_lat", "grid1d_lon"]
        for extra in extras:
            ds[extra] = ds0[extra]

    if ensemble:
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
    area_mean = cf * area_weighted.groupby(grid).sum()

    if domain == "global":
        area_mean = area_mean.mean(dim="biome")  # get rid of gridcell dimension

    return area_mean


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


def get_sparse_maps(
    ds: xr.Dataset, sparse_grid: xr.Dataset, land_frac: xr.DataArray, varlist: list[str]
) -> xr.Dataset:
    """Gets a dataset of global maps of a list of variables from a sparse dataset

    Args:
        ds (xr.Dataset): sparse grid dataset
        sparse_grid (xr.Dataset): sparse grid file
        land_frac (xr.DataArray): land fraction [0-1]
        varlist (list[str]): list of variables

    Returns:
        xr.Dataset: output dataset
    """

    # loop through each variable and map from sparse to global
    ds_list = []
    for var in varlist:
        var_ds = global_from_sparse(sparse_grid, ds[var], ds).to_dataset(name=var)
        var_ds[var] = var_ds[var] * land_frac
        ds_list.append(var_ds)

    return xr.merge(ds_list)


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
    target_grid["landmask"] = ds["landmask"].fillna(0)
    target_grid["landfrac"] = ds["landfrac"].fillna(0)

    return target_grid


def get_annual_means(ds, varlist, var_dict, sparse=False):
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


def get_sparse_area_means(ds, domain, varlist, var_dict, land_area, biome):
    ds_list = []
    for var in varlist:
        ds_list.append(
            area_mean_from_sparse(
                ds[var], biome, domain, var_dict[var]["cf_area"], land_area
            ).to_dataset(name=var)
        )

    return xr.merge(ds_list)
