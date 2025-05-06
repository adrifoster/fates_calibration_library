"""Functions to calculate Whittaker biomes globally"""

import glob
import functools
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
from fates_calibration_library.analysis_functions import (
    calculate_annual_mean,
    preprocess,
)
from fates_calibration_library.plotting_functions import generate_subplots, map_function


def calc_whittaker_biome_by_hand(temperature: float, precip: float) -> int:
    """Calculates Whittaker biome by hand via input temperature and precipitation

    Args:
        temperature (float): annual temperature [degC]
        precip (float): annual precipitation [mm/c]

    Returns:
        int: Whittaker Biome class
    """

    temp_1 = -5.0  # tundra-boreal
    temp_2 = 3.0  # boreal-temperate
    temp_3 = 20.0  # temperate-tropical

    temperature_vals = [-np.inf, temp_1, temp_2, temp_3, np.inf]
    biome_vals = [9, 8, 4, 1]

    if (temperature > 15.0) & (precip < 60.0):
        # desert
        return 3

    for i in range(len(temperature_vals) - 1):
        if (temperature > temperature_vals[i]) & (
            temperature <= temperature_vals[i + 1]
        ):
            biome_int = biome_vals[i]

    return biome_int


def get_whittaker_biome_class(
    temperature: np.array, precip: np.array, gpp: np.array, data
) -> np.array:
    """Calculates global biome class from input arrays of temperature, precipitation,
    and GPP, as well as a Whittaker biomes shapefile

    Args:
        temperature (np.array): global temperature [degC]
        precip (np.array): global precipitation [cm/yr]
        gpp (np.array): global GPP [gC/m2/yr]
        data (shp): whittaker biomes shapefile

    Returns:
        np.array: biome class
    """

    ncell = len(temperature)
    biome_class = np.zeros(ncell) + np.nan

    for cell in range(ncell):
        if (temperature[cell] < 0.0) & (gpp[cell] == 0.0):
            # ice
            biome_class[cell] = 0
        else:
            ptf = gpd.GeoDataFrame(
                {"geometry": [Point(temperature[cell], precip[cell])]}
            )
            point = gpd.overlay(ptf, data, how="intersection")
            if len(point) > 0:
                biome_class[cell] = point.biome_id.values[0]
            else:
                biome_class[cell] = calc_whittaker_biome_by_hand(
                    temperature[cell], precip[cell]
                )

    return biome_class


def get_global_whittaker_vars(clm_sim: xr.Dataset):
    """Reads in and processes data needed for Whittaker biome calculation

    Args:
        clm_dir (xr.Dataset): CLM simulation dataset

    Returns:
        xr.DataArray: temperature, precipitation, and gpp
    """

    # calculate temperature, precipitation, and gpp for each gridcell
    tbot = calculate_annual_mean(clm_sim.TBOT).mean(dim="year") - 273.15  # degC

    conversion_factor = 24 * 60 * 60 * 365 / 10
    rain = conversion_factor * calculate_annual_mean(clm_sim.RAIN).mean(
        dim="year"
    )  # cm/yr
    snow = conversion_factor * calculate_annual_mean(clm_sim.SNOW).mean(
        dim="year"
    )  # cm/yr
    prec = rain + snow

    gpp = calculate_annual_mean(clm_sim.GPP).mean(dim="year")

    return tbot, prec, gpp


def read_in_clm_sim(clm_dir: str):
    """Reads in CLM simulation needed to calculate Whittaker biomes

    Args:
        clm_dir (str): path to CLM simulation

    Returns:
        xr.Dataset: CLM simulation dataset
    """

    data_vars = ["FPSN", "TBOT", "RAIN", "SNOW", "TSA", "area", "landfrac"]

    # load full grid CLM simulation at 2degree
    files = sorted(glob.glob(clm_dir + "/*h0*"))
    clm_sim = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        preprocess=functools.partial(preprocess, data_vars=data_vars),
        decode_times=False,
    )
    clm_sim["time"] = xr.cftime_range("2000", periods=len(clm_sim.time), freq="MS", 
                                      calendar="noleap")
    mod_years = np.unique(clm_sim.time.dt.year)
    last_20 = mod_years[-20:]
    clm_sim = clm_sim.sel(time=slice(f"{last_20[0]}-01-01", f"{last_20[-1]}-12-31"))
    clm_sim["time"] = xr.cftime_range('2000', periods=len(clm_sim.time), freq="MS")
    clm_sim = clm_sim.sel(time=slice("2000-01-01", "2014-12-31"))

    return clm_sim


def get_whittaker_biomes(
    clm_dir: str, whit_shp_file: str, whitkey: xr.DataArray
) -> xr.Dataset:
    """Calculates global Whittaker biomes

    Args:
        clm_dir (str): path to CLM simulation (for input data)
        whit_shp_file (str): path to Whittaker biomes shapefile
        whitkey (xr.DataArray): data array with biome integer-key pairs

    Returns:
        xr.Dataset: Biome Dataset
    """

    # get data needed to calculate Whittaker biomes
    clm_sim = read_in_clm_sim(clm_dir)
    tbot, prec, gpp = get_global_whittaker_vars(clm_sim)

    # reshape arrays
    tbot_reshape = tbot.values.reshape(-1, 1)
    prec_reshape = prec.values.reshape(-1, 1)
    gpp_reshape = gpp.values.reshape(-1, 1)

    # read in the whittaker biomes shapefile
    whittaker_shapefile = gpd.read_file(whit_shp_file)
    whittaker_shapefile.biome_id = np.array([9, 8, 7, 6, 5, 4, 1, 2, 3])

    # calculate biome class
    biome_class = get_whittaker_biome_class(
        tbot_reshape, prec_reshape, gpp_reshape, whittaker_shapefile
    )

    # rehape and turn into a DataSet
    biome_id = biome_class.reshape(np.shape(tbot)[0], np.shape(tbot)[1])

    da_biome_id = xr.DataArray(
        biome_id, dims=("lat", "lon"), coords={"lat": clm_sim.lat, "lon": clm_sim.lon}
    )
    ds_out = da_biome_id.to_dataset(name="biome")
    ds_out["biome_name"] = whitkey.biome_name
    ds_out["landfrac"] = clm_sim.isel(time=0).landfrac
    ds_out["area"] = clm_sim.isel(time=0).area
    ds_out["biome"] = xr.where(ds_out.landfrac > 0.0, ds_out.biome, -9999)
    ds_masked = ds_out.where(ds_out["biome"] != -9999)

    return ds_masked


def get_biome_palette() -> tuple[dict[float, str], str]:
    """Returns a palette for plotting whittaker biomes

    Returns:
       tuple[dict[float, str], str]: color palette, biome names
    """

    # set the hue palette as a dict for custom mapping
    biome_names = [
        "Ice sheet",
        "Tropical rain forest",
        "Tropical seasonal forest/savanna",
        "Subtropical desert",
        "Temperate rain forest",
        "Temperate seasonal forest",
        "Woodland/shrubland",
        "Temperate grassland/desert",
        "Boreal forest",
        "Tundra",
    ]
    colors = [
        "#ADADC9",
        "#317A22",
        "#A09700",
        "#DCBB50",
        "#75A95E",
        "#97B669",
        "#D16E3F",
        "#FCD57A",
        "#A5C790",
        "#C1E1DD",
    ]

    palette = {}
    for i in range(len(colors)):
        palette[float(i)] = colors[i]

    return palette, biome_names

def compute_biotemperature(temp_C: xr.DataArray) -> xr.DataArray:
    """Computes biotemperature from input temperature

    Args:
        temp_C (xr.DataArray): temperature [degrees C]

    Returns:
        xr.DataArray: biotemperature [degrees C]
    """
    
    # clip temperatures between 0 and 30 °C
    temp_bio = temp_C.where(temp_C >= 0, 0)
    temp_bio = temp_bio.where(temp_bio <= 30, 30)
    biotemperature = calculate_annual_mean(temp_bio).mean(dim='year')
    biotemperature.name = "biotemperature"
    biotemperature.attrs["units"] = "°C"
    
    return biotemperature

def compute_annual_precip(rain_kg_m2_s: xr.DataArray, snow_kg_m2_s: xr.DataArray) -> xr.DataArray:
    """Calculates annual precipitation in mm/yr

    Args:
        rain_kg_m2_s (xr.DataArray): rain [kg/m2/s]
        snow_kg_m2_s (xr.DataArray): snow [kg/m2/s]

    Returns:
        xr.DataArray: precipitation [mm/yr]
    """
    
    conversion_factor = 24 * 60 * 60 * 365
    rain = conversion_factor * calculate_annual_mean(rain_kg_m2_s).mean(
        dim="year"
    )  # mm/yr
    snow = conversion_factor * calculate_annual_mean(snow_kg_m2_s).mean(
        dim="year"
    )  # mm/yr
    precip = rain + snow
    precip.name = "precipitation"
    precip.attrs["units"] = "mm/year"
    return precip

def compute_thornthwaite_pet(temp_C: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """
    Compute monthly Potential Evapotranspiration (PET) using Thornthwaite's method.

    Args:
        temp_C: xr.DataArray of monthly mean air temperature in Celsius with a `month` dimension.
              (e.g., shape: [month, lat, lon])
        lat: xr.DataArray of latitude in degrees, with dimensions matching spatial dims of temp_C.

    Returns:
    -   PET: xr.DataArray of monthly PET [mm/month]
    """

    monthly_temp = temp_C.groupby('time.month').mean('time')
    
    months = xr.DataArray(np.arange(1, 13), dims="month", name="month")

    # 1. Heat index (I)
    I_index = ((monthly_temp / 5) ** 1.514).sum(dim='month')

    # 2. Empirical exponent alpha
    alpha = 6.75e-7 * I_index**3 - 7.71e-5 * I_index**2 + 1.792e-2 * I_index + 0.49239

    # 3. Days per month
    N = xr.DataArray(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], 
        dims="month", coords={"month": months}
    )

    # 4. Midpoints of each month in day-of-year
    doy_mid = xr.DataArray(
        [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349],
        dims="month", coords={"month": months}, name="day_of_year"
    )

    # 5. Compute m and day length L (hours)
    lat_rad = np.deg2rad(lat)
    lat_b = lat_rad.expand_dims(month=months)
    axis_rad = np.deg2rad(23.439)  # Earth's axial tilt
    day_angle = np.pi * doy_mid / 182.625
    tan_axis_cos = np.tan(axis_rad * np.cos(day_angle))

    m = 1 - xr.apply_ufunc(np.tan, lat_b) * tan_axis_cos

    # Clip the input to arccos to valid range
    arccos_input = xr.apply_ufunc(np.clip, 1 - m, -1 + 1e-8, 1 - 1e-8)
    L = 24 * xr.apply_ufunc(np.arccos, arccos_input) / np.pi  # Day length in hours

    # 6. PET equation
    pet_term = ((10 * monthly_temp / I_index) ** alpha)
    PET = 1.6 * (L / 12) * (N / 30) * pet_term

    PET.name = "PET"
    PET.attrs["units"] = "mm/month"
    PET.attrs["description"] = "Potential Evapotranspiration via Thornthwaite method"

    return PET


def compute_sealevel_biotemp(temp_C: xr.DataArray, 
                             elevation: xr.DataArray) -> xr.DataArray:
    
    """Caclulates sea-level biotemperature

    Args:
        biotemperature (xr.DataArray): temperature [degrees C]
        elevation (xr.DataArray): elevation [m]

    Returns:
        xr.DataArray: sea-level biotemperature [degrees C]
    """
    
    sealevel_temp = temp_C + 6.0*(elevation/1000.0)
    sealevel_temp = sealevel_temp.where(sealevel_temp >= 0, 0)
    sealevel_temp = sealevel_temp.where(sealevel_temp <= 30, 30)
    mean_sealevel_temp = calculate_annual_mean(sealevel_temp).mean(dim='year')
    mean_sealevel_temp.name = "sealevel_temp"
    mean_sealevel_temp.attrs["units"] = "°C"
    
    return mean_sealevel_temp

def compute_pet_ratio(PET: xr.DataArray, precip: xr.DataArray) -> xr.DataArray:
    """Calculates PET ratio of PET/Pprecip

    Args:
        PET (xr.DataArray): _description_
        precip (xr.DataArray): _description_

    Returns:
        xr.DataArray: _description_
    """
    pet_ratio = PET / precip.where(precip > 0.0)
    pet_ratio = xr.where(pet_ratio > 40.0, 40.0, pet_ratio)
    pet_ratio.name = "pet_ratio"
    return pet_ratio

def bin_biotemperature(biotemp_annual: xr.DataArray, frost_free: xr.DataArray) -> xr.DataArray:
    """Bin up biotemperature

    Args:
        biotemp_annual (xr.DataArray): biotemperature [degrees C]
        frost_free (xr.DataArray): frost-free area [0/1]

    Returns:
        xr.DataArray: binned biotemperature
    """
    
    # bins and names
    bio_bins = [1.5, 3, 6, 12, 24]
    bio_labels = ['alvar', 'alpine', 'subalpine', 'montane', 'premontane', '']
    bio_labels_expanded = ['alvar', 'alpine', 'subalpine', 'montane', 'lower montane', 
                           'premontane', '']
    
    # bin up data
    biotemp = biotemp_annual.load()
    biotemp_class = xr.apply_ufunc(np.digitize, biotemp, bio_bins) 
    bio_labels_arr = np.array(bio_labels)
    biotemp_zone = xr.apply_ufunc(
        lambda i: bio_labels_arr[i],
        biotemp_class,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str]
    )
    
    # correct for frost-free areas
    biotemp_zone = xr.where(
        (biotemp_zone == "premontane") & (~frost_free),
        "lower montane",
        biotemp_zone)
    biotemp_zone = xr.where(
        (biotemp_zone == "") & (~frost_free),
        "premontane",
        biotemp_zone)
    
    biotemp_zone.name = 'altitude_zone'
    
    # map zone names to integers
    zones_int = map_to_int(biotemp_zone.load(), bio_labels_expanded).to_dataset(name='altitude_zone_int')
    
    return xr.merge([biotemp_zone, zones_int])

def map_to_int(ds, labels):
    
    zone_map = {label: i for i, label in enumerate(labels)}
    zones = xr.apply_ufunc(lambda z: zone_map[z], ds, vectorize=True)
    
    return zones

def bin_sealeveltemp(sealevel_temp: xr.DataArray, frost_free: xr.DataArray) -> xr.DataArray:
    """Bin up sea-level temperature 

    Args:
        sealevel_temp (xr.DataArray): sea-level biotemperature [degrees C]
        frost_free (xr.DataArray): frost-free area [0/1]

    Returns:
        xr.DataArray: latitude bins
    """
    temp_bins = [1.68, 3.36, 6.72, 13.44, 26.89]
    temp_labels = ['polar', 'subpolar', 'boreal', 'cool temperate', 'subtropical', 'tropical']
    temp_labels_expanded = ['polar', 'subpolar', 'boreal', 'cool temperate', 'warm temperate',
                         'subtropical', 'tropical']
    
    seatemp = sealevel_temp.load()
    seatemp_class = xr.apply_ufunc(np.digitize, seatemp, temp_bins)
    lat_labels_arr = np.array(temp_labels)
    seatemp_zone = xr.apply_ufunc(
        lambda i: lat_labels_arr[i],
        seatemp_class,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[str]
    )

    seatemp_zone = xr.where(
        (seatemp_zone == "subtropical") & (~frost_free),
        "warm temperate",
        seatemp_zone)
    seatemp_zone = xr.where(
        (seatemp_zone == "tropical") & (~frost_free),
        "subtropical",
        seatemp_zone)
    
    seatemp_zone.name = 'latitude_zone'

    # map zone names to integers
    zones_int = map_to_int(seatemp_zone.load(), temp_labels_expanded).to_dataset(name='latitude_zone_int')

    return xr.merge([seatemp_zone, zones_int])

def classify_holdridge(clim_dat, holdridge_dat):
    
    clim_flat = clim_dat.stack(grid=('lat', 'lon'))
    
    # convert holdridge columns to np arrays
    hold_pet = holdridge_dat['PET_ratio'].values 
    hold_temp = holdridge_dat['biotemp'].values
    hold_precip = holdridge_dat['precip'].values
    
    # get CLM data
    pet_vals = clim_flat['pet_ratio'].values[:, np.newaxis] 
    biotemp_vals = clim_flat['biotemperature'].values[:, np.newaxis]
    precip_vals = clim_flat['precipitation'].values[:, np.newaxis]
    
    # compute euclidean distance
    pet_dist = (pet_vals - hold_pet) ** 2
    temp_dist = (biotemp_vals - hold_temp) ** 2
    precip_dist = (precip_vals - hold_precip) ** 2
    total_dist = pet_dist + temp_dist + precip_dist
    
    # find best match per grid cell
    best_idx = np.argmin(total_dist, axis=1)
    biome_labels = holdridge_dat['biome'].values[best_idx]

    biome_da = xr.DataArray(
        biome_labels.reshape(clim_dat['pet_ratio'].shape),
        coords=clim_dat['pet_ratio'].coords,
        dims=clim_dat['pet_ratio'].dims,
        name='holdridge_biome'
    )
    
    return biome_da

def get_frost_free(temp_C: xr.DataArray) -> xr.DataArray:
    """Returns a binary frost-free mask

    Args:
        temp_C (xr.DataArray): temperature [degrees C]

    Returns:
        xr.DataArray: frost-free
    """
    frost_months = (temp_C) < 0
    frost_occurred = frost_months.any(dim='time')
    return ~frost_occurred

def get_holdridge_dat(clm_ds, elevation):
    
    temp_C = clm_ds.TSA - 273.15
    
    # calculate annual metrics
    biotemperature = compute_biotemperature(temp_C)
    sealevel_temp = compute_sealevel_biotemp(temp_C, elevation)
    precipitation = compute_annual_precip(clm_ds.RAIN, clm_ds.SNOW)
    pet_ratio = (biotemperature*58.93)/precipitation
    pet_ratio.name = 'pet_ratio'
    frost_free = get_frost_free(temp_C)
    
    # bin up temperature
    biozones = bin_biotemperature(biotemperature, frost_free)
    latzones = bin_sealeveltemp(sealevel_temp, frost_free)
    
    # if actual biotemperature and sea-level temperature are in the same zone, altitudinal
    # zone is 'basal' (i.e. nothing)
    same_zone = biozones.altitude_zone_int == latzones.latitude_zone_int
    biozones['altitude_zone'] = xr.where(same_zone, '', biozones.altitude_zone)
    biozones['altitude_zone_int'] = xr.where(same_zone, 0, biozones.altitude_zone_int)
    
    clim_dat = xr.merge([biotemperature,
                     pet_ratio,
                     precipitation,
                     biozones, latzones])
    return clim_dat
    
                                             
def plot_whittaker_biomes(whit_ds):

    colors, names = get_biome_palette()
    cmap = matplotlib.colors.ListedColormap(list(colors.values()))

    figure, axes = generate_subplots(1)
    pcm = map_function(axes[0], whit_ds, "Whittaker Biomes", cmap, -0.5, 9.5)
    cbar = figure.colorbar(pcm, ax=axes[0], fraction=0.03, orientation="vertical")
    cbar.set_label("Biome", size=12, fontweight="bold")
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cbar.set_ticklabels(names)
