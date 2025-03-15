"""Functions to calculate Whittaker biomes globally
"""
import glob
import functools
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib
from fates_calibration_library.analysis_functions import calculate_annual_mean, preprocess
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
                biome_class[cell] = calc_whittaker_biome_by_hand(temperature[cell], precip[cell])

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
    
    data_vars = ["TLAI", "GPP", "TBOT", "RAIN", "SNOW", "area", "landfrac"]
    
    # load full grid CLM simulation at 2degree
    files = sorted(glob.glob(clm_dir + "*h0*"))[-84:]
    clm_sim = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="time",
        parallel=True,
        preprocess=functools.partial(preprocess, data_vars=data_vars),
        decode_times=False,
    )
    clm_sim["time"] = xr.cftime_range("2007", periods=84, freq="MS", calendar="noleap")

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
    biome_class = get_whittaker_biome_class(tbot_reshape, prec_reshape, gpp_reshape, whittaker_shapefile)

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
    biome_names = ['Ice sheet', 'Tropical rain forest',
                'Tropical seasonal forest/savanna', 'Subtropical desert',
                'Temperate rain forest', 'Temperate seasonal forest',
                'Woodland/shrubland', 'Temperate grassland/desert',
                'Boreal forest', 'Tundra']
    colors = ["#ADADC9", "#317A22", "#A09700", "#DCBB50", "#75A95E", "#97B669",
            "#D16E3F", "#FCD57A", "#A5C790", "#C1E1DD"]
    
    palette = {}
    for i in range(len(colors)):
        palette[float(i)] = colors[i]
        
    return palette, biome_names

def plot_whittaker_biomes(whit_ds):
    
    colors, names = get_biome_palette()
    cmap = matplotlib.colors.ListedColormap(list(colors.values()))
    
    figure, axes = generate_subplots(1)
    pcm = map_function(axes[0], whit_ds, "Whittaker Biomes", cmap, -0.5, 9.5)
    cbar = figure.colorbar(pcm, ax=axes[0], fraction=0.03, orientation='vertical')
    cbar.set_label('Biome', size=12, fontweight='bold')
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cbar.set_ticklabels(names)      
    