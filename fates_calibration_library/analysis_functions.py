"""Functions to assist with analysis of netcdf observations and model outputs"""

import xarray as xr
import numpy as np


def preprocess(data_set: xr.Dataset, data_vars: list[str]) -> xr.Dataset:
    """Preprocesses an xarray Dataset by subsetting to specific variables - to be used on read-in

    Args:
        data_set (xr.Dataset): input Dataset

    Returns:
        xr.Dataset: output Dataset
    """

    return data_set[data_vars]


def cyclic_month_difference(da1: xr.DataArray, da2: xr.DataArray) -> xr.DataArray:
    """Calculate the minimum cyclic difference between two xarray DataArrays of months (1-12).

    Args:
        da1 (xr.DataArray): first data array
        da2 (xr.DataArray): second data array

    Returns:
        xr.DataArray: output data array
    Raises:
        TypeError: inputs are not xarray.DataArray instances
        ValueError: input arrays are not the same shape
    """
    if not isinstance(da1, xr.DataArray) or not isinstance(da2, xr.DataArray):
        raise TypeError("Inputs must be xarray.DataArray instances.")
    if da1.shape != da2.shape:
        raise ValueError("Input arrays must have the same shape.")
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
    if lon_name not in ds.variables:
        raise ValueError(f"{lon_name} not found in dataset.")

    # adjust lon values to make sure they are within (-180, 180)
    ds["longitude_adjusted"] = xr.where(
        ds[lon_name] > 180, ds[lon_name] - 360, ds[lon_name]
    )

    # reassign the new coords to as the main lon coords
    # and sort DataArray using new coordinate values
    ds = (
        ds.swap_dims({lon_name: "longitude_adjusted"})
        .sortby("longitude_adjusted")
        .drop_vars(lon_name)
        .rename({"longitude_adjusted": lon_name})
    )

    if lon_name not in ds.coords:
        ds = ds.set_coords(lon_name)

    return ds


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
    return conversion_factor * area_weighted


def calculate_month_of_max(
    monthly_mean: xr.DataArray, month_dim: str = "month"
) -> xr.DataArray:
    """Calculates the month (or index) of the maximum value for each location in a monthly mean array

    Args:
        monthly_mean (xr.DataArray): monthly data array with a 'month' dimension.
        mond_dim (str): name of the dimension representing months. Defaults to 'month'.

    Returns:
        xr.DataArray: array with month of maximum value at each location.
                      returns NaN where all values are NaN or flat.
    Raises:
        ValueError: "Dimension month_dim not found in input DataArray"
    """

    if month_dim not in monthly_mean.dims:
        raise ValueError(f"Dimension '{month_dim}' not found in input DataArray.")

    # calculate the month of the maximum value
    month_of_max = monthly_mean.idxmax(dim=month_dim)

    # identify flat or all NaN slices
    is_flat = (monthly_mean.max(month_dim) - monthly_mean.min(month_dim)) == 0
    all_nan = monthly_mean.isnull().all(dim=month_dim)

    # replace flat with first month value (or NaN)
    month_of_max = month_of_max.where(~is_flat)
    month_of_max = month_of_max.where(~all_nan)

    return month_of_max


def _weighted_annual_mean(data_array: xr.DataArray) -> xr.DataArray:
    """Computes weighted annual mean using daysinmonth for missing-aware inputs.

    Args:
        data_array (xr.DataArray): input DataArray

    Returns:
        xr.DataArray: output DataArray
    """

    # multiply by number of days in month
    weighted = data_array * data_array["time.daysinmonth"]

    # compute number of valid days per year
    valid_days = data_array["time.daysinmonth"].where(data_array.notnull())

    # group and sum weighted data and valid days
    annual_sum = weighted.groupby("time.year").sum(dim="time", skipna=True)
    days_per_year = valid_days.groupby("time.year").sum(dim="time", skipna=True)

    return annual_sum / days_per_year


def _annual_sum(data_array: xr.DataArray, conversion_factor: float) -> xr.DataArray:
    """Computes annual sum

    Args:
        data_array (xr.DataArray): input DataArray
        conversion_factor (float): conversion factor

    Returns:
        xr.DataArray: annual sum output
    """

    months = data_array["time.daysinmonth"]
    return conversion_factor * (months * data_array).groupby("time.year").sum()


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
    Raises:
        ValueError: Input must have a 'time' dimension.
    """
    if "time" not in data_array.dims:
        raise ValueError("Input must have a 'time' dimension.")

    if conversion_factor is None:

        annual_mean = _weighted_annual_mean(data_array)

    else:
        annual_mean = _annual_sum(data_array, conversion_factor)

    annual_mean.name = data_array.name
    if new_units:
        annual_mean.attrs["units"] = new_units

    return annual_mean


def calculate_monthly_mean(
    data_array: xr.DataArray, conversion_factor: float = None, new_units: str = ""
) -> xr.DataArray:
    """Calculates monthly mean of an input DataArray, applies a conversion factor

    Args:
        data_array (xr.DataArray): input DataArray
        conversion_factor (float, optional): optional scaling factor. Defaults to empty
        new_units (str, optional): optional new units string. Defaults to empty.

    Returns:
        xr.DataArray: output DataArray
    Raises:
        ValueError: Input must have a 'time' dimension.
    """
    if "time" not in data_array.dims:
        raise ValueError("Input must have a 'time' dimension.")
    months = data_array["time.daysinmonth"]

    if conversion_factor is None:
        conversion_factor = 1.0

    weighted = conversion_factor * data_array * months
    valid_days = months.where(data_array.notnull())

    monthly_sum = weighted.groupby("time.month").sum(dim="time", skipna=True)
    total_days = valid_days.groupby("time.month").sum(dim="time", skipna=True)

    monthly_mean = monthly_sum / total_days

    # mask out grid cells that were all-NaN across the time axis
    all_nan_mask = data_array.isnull().all(dim="time")
    monthly_mean = monthly_mean.where(~all_nan_mask)

    monthly_mean.name = data_array.name

    if new_units:
        monthly_mean.attrs["units"] = new_units

    return monthly_mean


def area_mean(da: xr.DataArray, cf: float, land_area: xr.DataArray) -> xr.DataArray:
    """Calculates a global area-weighted mean of a global dataset

    Args:
        da (xr.DataArray): input data array
        cf (float): conversion factor
        land_area (xr.DataArray): land area data array

    Returns:
        xr.DataArray: output data array
    """

    # update conversion factor if need be
    land_area = land_area.where(~np.isnan(da))
    if cf is None:
        cf = 1 / land_area.sum(dim=["lat", "lon"]).values

    # weight by landarea
    area_weighted = land_area * da

    # calculate area mean
    weighted_mean = cf * area_weighted.sum(dim=["lat", "lon"])

    return weighted_mean


def compute_infl(da: xr.DataArray, dim="month") -> np.ndarray:
    """Calculates inflection points in a array indexed by month

    Args:
        da (xr.DataArray): input data array
        dim (str, optional): month dimension. Defaults to "month".

    Returns:
        np.ndarray: indices of inflection points in monthly values
    Raises:
        ValueError: Expected 12 months in dimension and got something else
    """
    if da.sizes[dim] != 12:
        raise ValueError(f"Expected 12 months in dimension '{dim}', got {da.sizes[dim]}")

    # compute the first and second derivatives
    da_vals = da.values
    first_diff = np.diff(da_vals, axis=-1) >= 0.0
    second_diff = np.diff(first_diff.astype(int), axis=-1) != 0

    # pad with two False values at the beginning to match original length
    pad = np.zeros((*second_diff.shape[:-1], 2), dtype=bool)
    padded = np.concatenate([pad, second_diff], axis=-1)
    
    infl = xr.DataArray(padded, coords=da.coords, dims=da.dims)

    # get months where inflection points occur
    infl_months = da[dim].where(infl).values
    # remove NaNs
    non_nan_months = infl_months[~np.isnan(infl_months)]

    return fix_infl_months(non_nan_months)


def fix_infl_months(inflection_months: np.ndarray) -> np.ndarray:
    """Ensures an array of monthly inflection points contains 3 values by
    appending January (1) or December (12) when needed.

    Args:
        inflection_months (np.ndarray): Array of inflection point months (1–12)

    Returns:
        np.ndarray: Sorted array of 3 inflection months
    Raises:
        ValueError: inflection_months must contain 1 to 3 elements
    """

    # hard-coded months
    JAN = 1
    DEC = 12
    JUN = 6
    
    # check number of months
    if len(inflection_months) == 3:
        return inflection_months  # return as is

    elif len(inflection_months) == 2:
        if inflection_months[0] <= JUN:
            # assume we're missing the last month
            return np.sort(np.append(inflection_months, DEC))
        else:
            # assume we're missing the first month
            return np.sort(np.append(JAN, inflection_months))
    elif len(inflection_months) == 1:
        # assume only have peak
        return np.sort(np.append(inflection_months, (DEC, JAN)))
    else:
        raise ValueError("inflection_months must contain 1 to 3 elements.")


def get_start_end_slopes(da, infl_months):

    start = da.sel(month=slice(infl_months[0].item(), infl_months[1].item()))
    end = da.sel(month=slice(infl_months[1].item(), infl_months[2].item()))

    x_start = start["month"].values
    y_start = start.values
    x_end = end["month"].values
    y_end = end.values

    # skip NaNs or too-short periods
    if len(x_start) >= 2 and len(x_end) >= 2:
        slope_start = np.polyfit(x_start, y_start, 1)[0]
        slope_end = np.polyfit(x_end, y_end, 1)[0]
    else:
        slope_start = np.nan
        slope_end = np.nan

    return slope_start, slope_end
