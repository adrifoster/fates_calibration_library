"""Functions to assist with plotting"""

import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geocollection import GeoQuadMesh
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
import seaborn as sns
from adjustText import adjust_text
import textwrap

from fates_calibration_library.analysis_functions import cyclic_month_difference
from fates_calibration_library.analysis_functions import calculate_zonal_mean

_COLS = [
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

_PFT_COLS = [
    "#23B44E",
    "#496041",
    "#F15A46",
    "#1DB58C",
    "#88F6E5",
    "#FFBF02",
    "#AADC32",
    "#AB2F5D",
    "#AB2F5D",
    "#738678",
    "#9DC183",
    "#2C728E",
    "#FFF3B0",
    "#E09F3F",
    "#BCB06F",
    "#9C9478",
]

# custom fill colors and labels for parameter categories
_CATEGORY_COLORS = {
    'hydrology': '#104E8B',
    'biophysics': '#8B008B',
    'stomatal': '#008B00',
    'biogeochemistry': '#8B5A2B',
}

_CATEGORY_LABELS = {
    'hydrology': 'Hydrology',
    'biophysics': 'Biophysics',
    'stomatal': 'Stomatal \nConductance & \nPhotosynthesis',
    'biogeochemistry': 'Biogeochemistry',
    'land use': 'Land Use',
    'fire': 'Fire'
}

_SUBCATEGORY_LABELS = {
        'fire': 'Fire',
        'land use': 'Land use',
        'allocation': 'Allocation',
        'allometry': 'Allometry',
        'decomposition': 'Decomposition',
        'mortality': 'Mortality',
        'nutrient uptake': 'Nutrient uptake',
        'phenology': 'Phenology',
        'recruitment': 'Recruitment',
        'respiration': 'Respiration',
        'vegetation dynamics': 'Vegetation dynamics',
        'acclimation': 'Acclimation',
        'photosynthesis': 'Photosynthesis',
        'vegetation water': 'Vegetation water',
        'latent': 'Sensible and latent heat',
        'radiation': 'Radiation',
        'soil water': 'Soil hydraulics',
        'snow': 'Snow',
        'thermal': 'Soil thermal properties',
}

_BIOME_NAMES = {
    0.0: 'Ice sheet',
    1.0: 'Tropical rain forest',
    2.0: "Tropical seasonal forest/savanna",
    3.0: "Subtropical desert",
    4.0: "Temperate rain forest",
    5.0: "Temperate seasonal forest",
    6.0: "Woodland/shrubland",
    7.0: "Temperate grassland/desert",
    8.0: "Boreal forest",
    9.0: "Tundra",
}

_MODEL_COLS = {
    'FATES':'#2066a8',
    'CLM': '#ea801c',
}

def wrap_labels(labels, width=20):
    return ['\n'.join(textwrap.wrap(label, width=width)) for label in labels]

def plot_zonal_mean_diff(
    data_arrays: list[xr.DataArray],
    dataset_names: list[str],
    var: str,
    long_name: str,
    units: str,
):
    """Plot an annual cycle of a variable

    Args:
        data_arrays (list[xr.DataArray]): data arrays to plot
        dataset_names (list[str]): names of datasets
        var (str): variable name
        ylabel (str): label for y axis
        units (str): units for y axis
    """
    assert len(data_arrays) == len(
        dataset_names
    ), "Each DataArray must have a corresponding name"

    # convert all to datasets and concatenate
    datasets = [da.to_dataset(name=var) for da in data_arrays]
    ds = xr.concat(datasets, dim="version")
    ds = ds.assign_coords(version=("version", dataset_names))

    df = pd.DataFrame(
        {
            "lat": np.tile(ds.lat, len(ds.version)),
            "version": np.repeat(ds.version, len(ds.lat)),
            var: ds[var].values.flatten(),
        }
    )

    # plot
    get_blank_plot()
    plt.ylim(-90, 90)
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot models
    for rank, version in enumerate(np.unique(df.version.values)):
        data = df[df.version == version]
        color = _COLS[rank % len(_COLS)]  # cycle through _COLS if needed
        plt.plot(data[var].values, data.lat.values, lw=2, color=color, label=version)

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {long_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    plt.title("Zonal Mean Difference")


def plot_model_obs_climatology_diff(ilamb_var: xr.DataArray, model_var: xr.DataArray, var_name: str, long_name: str, units: str):
    """Plots climatology for observations against modeled data

    Args:
        ilamb_var (xr.DataArray): observational data
        model_var (xr.DataArray): model data
        var_name (str): variable name
        long_name (str): long name
        units (str): units for plotting
    """

    ilamb_df = pd.DataFrame(
        {
            "month": np.tile(ilamb_var.month, len(ilamb_var.model)),
            "model": np.repeat(ilamb_var.model, len(ilamb_var.month)),
            var_name: ilamb_var.values.flatten(),
        }
    )

    get_blank_plot()

    # add latitude-specific ticks/lines
    plt.xlim(1, 12)
    plt.xticks(
        range(1, 13, 1),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        fontsize=10,
    )

    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot models
    for rank, model in enumerate(np.unique(ilamb_df.model.values)):
        data = ilamb_df[ilamb_df.model == model]
        plt.plot(
            data.month.values,
            data[var_name].values,
            lw=2,
            color=_COLS[rank],
            label=model,
        )

    plt.plot(
        model_var.month.values, model_var.values, lw=2, color="black", label="FATES"
    )

    plt.xlabel("Month", fontsize=11)
    plt.ylabel(f"{long_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")


def plot_model_obs_zonal_diff(
    ilamb_var,
    model_var,
    land_area,
    conversion_factor,
    var_name,
    long_name,
    units,
):

    ilamb_by_lat = calculate_zonal_mean(ilamb_var, land_area, conversion_factor)
    ilamb_by_lat = ilamb_by_lat.transpose("model", "lat")
    df = pd.DataFrame(
        {
            "lat": np.tile(ilamb_by_lat.lat, len(ilamb_by_lat.model)),
            "model": np.repeat(ilamb_by_lat.model, len(ilamb_by_lat.lat)),
            var_name: ilamb_by_lat.values.flatten(),
        }
    )

    get_blank_plot()

    # add latitude-specific ticks/lines
    plt.ylim(-90, 90)
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot models
    for rank, model in enumerate(np.unique(df.model.values)):
        data = df[df.model == model]
        plt.plot(
            data[var_name].values, data.lat.values, lw=2, color=_COLS[rank], label=model
        )

    plt.plot(model_var.values, model_var.lat.values, lw=2, color="black", label="FATES")

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {long_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")


def choose_subplot_dimensions(num_plots: int) -> tuple[int, int]:
    """Chooses a nice array size/dimension for plotting subplots based on the total
    number of input plots

    Args:
        num_plots (int): total number of plots

    Returns:
        tuple[int, int]: nrow, ncol for subplot dimensions
    """

    if num_plots < 2:
        return num_plots, 1
    if num_plots < 11:
        return math.ceil(num_plots / 2), 2
    # I've chosen to have a maximum of 3 columns
    return math.ceil(num_plots / 3), 3


def plot_model_obs_diff(model_ds, obs_da, land_frac, var, units, model_name):

    sub_list = []
    for model in obs_da.model:
        sub = obs_da.sel(model=model)
        sub = sub.where(np.abs(sub > 0.0))
        sub_list.append(sub)

    obs_da = xr.concat(sub_list, dim="model")

    mean_diff = model_ds[var] * land_frac - obs_da.mean(dim="model")
    vmax = mean_diff.max().values
    vmin = mean_diff.min().values

    models = obs_da.model.values
    figure, axes = generate_subplots(len(models) + 1)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx < len(models):
            obs_model = obs_da.sel(model=models[idx])
            diff = model_ds[var] * land_frac - obs_model
            title = f"{model_name} - {models[idx]}"
        else:
            diff = mean_diff
            title = f"{model_name} - {len(models)}-model average"

        pcm = map_function(ax, diff, title, "RdBu_r", vmin, vmax, diverging_cmap=True)
    cbar = figure.colorbar(
        pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation="horizontal"
    )
    cbar.set_label(f"{var} Difference ({units})", size=10, fontweight="bold")


def plot_zonal(da: xr.DataArray, xlabel: str, units: str, title: str):
    """Plot a zonal mean of a variable

    Args:
        da (xr.DataArray): data array
        var (str): variable name
        ylabel (str): label for y axis
        units (str): units for y axis
        title (str): title for plot
    """

    get_blank_plot()

    minval = da.min()
    minvar = minval
    maxvar = da.max()

    # add latitude-specific ticks/lines
    plt.xlim(minvar - 0.01, maxvar + 0.01)
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
    plt.plot(da.values, da.lat.values, lw=2)

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"{xlabel} ({units})", fontsize=11)
    plt.title({title})


def generate_subplots(
    num_plots: int, row_wise: bool = False
) -> tuple[plt.figure, np.ndarray]:
    """Generates subplots based on the number of input plots and adds ticks for the last axis in
    each column

    Args:
        num_plots (int): number of plots
        row_wise (bool, optional): row wise?. Defaults to False.

    Returns:
        tuple[plt.figure, np.ndarray]: figure, array of axes
    """

    nrow, ncol = choose_subplot_dimensions(num_plots)
    figure, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(13, 6),
        subplot_kw=dict(projection=ccrs.Robinson()),
        layout="compressed",
    )

    if not isinstance(axes, np.ndarray):
        return figure, np.array([axes])
    else:
        axes = axes.flatten(order=("C" if row_wise else "F"))
        for idx, ax in enumerate(axes[num_plots:]):
            figure.delaxes(ax)
            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = (
                idx + num_plots - ncol if row_wise else idx + num_plots - 1
            )
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)
        axes = axes[:num_plots]

        return figure, axes


def map_function(
    ax: plt.Axes,
    dat: xr.DataArray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    diverging_cmap: bool = False,
) -> GeoQuadMesh:
    """Plots a color mesh along with coastlines and ocean for a global data array

    Args:
        ax (plt.Axes): axes to plot on
        dat (xr.DataArray): data array to plot
        title (str): title of subplot/axes
        cmap (str): colormap to use
        vmax (float): maximum value for colormap
        vmin (float): minimum value for colormap
        diverging_cmap (bool, optional): whether a diverging colormap is used. Defaults to False.

    Returns:
        GeoQuadMesh: color mesh
    """

    # if we have a diverging colormap, make the min/max values even
    if diverging_cmap:
        vmin = min(vmin, -1.0 * vmax)
        vmax = max(vmax, np.abs(vmin))

    # add title, coastlines, ocean
    ax.set_title(title, loc="left", fontsize="large", fontweight="bold")
    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )
    # plot the color mesh
    pcm = ax.pcolormesh(
        dat.lon,
        dat.lat,
        dat,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return pcm


def get_blank_plot(width=7, height=5):
    """Generates a blank plot"""

    plt.figure(figsize=(width, height))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


def round_up(number: float, decimals: int = 0) -> float:
    """rounds a number to a specified number of decimals

    Args:
        number (float): number to round
        decimals (int, optional): number of decimals to round to. Defaults to 0.

    Returns:
        float: rounded number
    """
    multiplier = 10**decimals
    return math.ceil(number * multiplier) / multiplier


def round_down(number: float, decimals: int = 0) -> float:
    """Rounds a number down to a specified number of decimals

    Args:
        number (float): input number to round
        decimals (int, optional): number of decimals. Defaults to 0.

    Returns:
        float: rounded number
    """
    multiplier = 10**decimals
    return int(number * multiplier) / multiplier


def plot_two_model_diff(da1, da2, ds1_name, ds2_name, fates_var, units, cmap,
                        diverging=False, plot_type='abs'):

    vmin = np.min([da1.min().values, da2.min().values])
    vmax = np.max([da1.max().values, da2.max().values])

    figure, axes = generate_subplots(3)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx == 0:
            pcm = map_function(
                ax, da2, ds2_name, cmap, vmin, vmax, diverging_cmap=diverging
            )
        elif idx == 2:
            pcm = map_function(
                ax, da1, ds1_name, cmap, vmin, vmax, diverging_cmap=diverging
            )
        elif idx == 1:
            diff = da2 - da1
            pcmdiff = map_function(
                ax,
                diff,
                f"{ds2_name} - {ds1_name}",
                "RdBu_r",
                diff.min().values,
                diff.max().values,
                diverging_cmap=True,
            )
    cbar1 = figure.colorbar(pcm, ax=axes[2], shrink=1, orientation="vertical")
    cbar2 = figure.colorbar(pcmdiff, ax=axes[1], shrink=1, orientation="horizontal")
    if plot_type == 'diff':
        cbar1.set_label(f"Difference in {fates_var} ({units})", size=10, fontweight="bold")
        cbar2.set_label(f"Delta Difference ({units})", size=10, fontweight="bold")
        
    else:
        cbar1.set_label(f"Difference in {fates_var} ({units})", size=10, fontweight="bold")
        cbar2.set_label(f"{fates_var} Difference ({units})", size=10, fontweight="bold")
        


def plot_month_of_max_diff(da1, da2, ds1_name, ds2_name, fates_var):

    figure, axes = generate_subplots(3)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx == 0:
            pcm = map_function(
                ax, da2, ds2_name, "jet", 0.5, 12.5, diverging_cmap=False
            )
        elif idx == 2:
            pcm = map_function(
                ax, da1, ds1_name, "jet", 0.5, 12.5, diverging_cmap=False
            )
        elif idx == 1:
            diff = cyclic_month_difference(da1, da2)
            pcmdiff = map_function(
                ax, diff, f"{ds2_name} - {ds1_name}", "PRGn", -5, 5, diverging_cmap=True
            )
    cbar1 = figure.colorbar(pcm, ax=axes[2], shrink=1, orientation="vertical")
    cbar1.set_label("Month", size=12, fontweight="bold")
    cbar1.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    cbar1.set_ticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )

    cbar2 = figure.colorbar(pcmdiff, ax=axes[1], shrink=1, orientation="horizontal")
    cbar2.set_label("Difference in Month of Max", size=10, fontweight="bold")

    figure.suptitle(f"Comparison for Month of Max for {fates_var}")


def plot_heatmap(summary_df):
    """Plot a heatmap of dataset means and relative differences."""
    # create a mask: Keep only 'Relative Difference (%)' for coloring
    mask = summary_df.copy()
    mask.loc[:, mask.columns != "Relative Difference (%)"] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        cbar_kws={"label": "Relative Difference (%)"},
    )

    # mannually add text for the other columns (to keep them uncolored)
    for i in range(summary_df.shape[0]):
        for j in range(summary_df.shape[1]):
            if summary_df.columns[j] != "Relative Difference (%)":
                text = (
                    f"{summary_df.iloc[i, j]:.2f}"
                    if not np.isnan(summary_df.iloc[i, j])
                    else ""
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

    plt.title("Dataset Means and Differences")
    plt.yticks(rotation=0)


def summarize_differences(ds1, ds2, ds1_name, ds2_name, var_dict):
    """Summarize global differences between two xarray datasets, handling Dask arrays and
    adding units."""

    summary = []
    for var in ds1.data_vars:
        if var in ds2:

            unit = var_dict[var]["global_units"]
            unit_str = f" ({unit})" if unit else ""
            mean1 = ds1[var].values
            mean2 = ds2[var].values
            diff = mean2 - mean1
            rel_diff = (diff / mean1 * 100) if mean1 != 0 else None

            # Append data with unit in the variable name
            summary.append([f"{var}{unit_str}", f"Mean of {ds1_name}", mean1.item()])
            summary.append([f"{var}{unit_str}", f"Mean of {ds2_name}", mean2.item()])
            summary.append(
                [
                    f"{var}{unit_str}",
                    "Absolute Difference",
                    diff.item() if diff is not None else None,
                ]
            )
            summary.append(
                [
                    f"{var}{unit_str}",
                    "Relative Difference (%)",
                    rel_diff.item() if rel_diff is not None else None,
                ]
            )

    # convert list to DataFrame
    summary_df = pd.DataFrame(summary, columns=["Variable", "Statistic", "Value"])
    summary_df = summary_df.pivot(index="Variable", columns="Statistic", values="Value")

    # reorder columns
    desired_order = [
        f"Mean of {ds1_name}",
        f"Mean of {ds2_name}",
        "Absolute Difference",
        "Relative Difference (%)",
    ]
    summary_df = summary_df[desired_order]

    return summary_df


def plot_annual_cycle_diff(
    da1: xr.DataArray,
    da2: xr.DataArray,
    ds1_name: str,
    ds2_name: str,
    var: str,
    ylabel: str,
    units: str,
):
    """Plot an annual cycle of a variable

    Args:
        da1 (xr.DataArray): data array for dataset 1
        da2 (xr.DataArray): data array for dataset 2
        var (str): variable name
        ds1_name (str): name of first dataset
        ds2_name (str): name of second dataset
        ylabel (str): label for y axis
        units (str): units for y axis
    """

    # merge together
    ds = xr.concat([da1.to_dataset(name=var), da2.to_dataset(name=var)], dim="version")
    ds = ds.assign_coords(version=("version", [ds1_name, ds2_name]))

    df = pd.DataFrame(
        {
            "month": np.tile(ds.month, len(ds.version)),
            "version": np.repeat(ds.version, len(ds.month)),
            var: ds[var].values.flatten(),
        }
    )

    # plot
    get_blank_plot()
    plt.xlim(1, 12)
    plt.xticks(
        range(1, 13, 1),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        fontsize=10,
    )

    # add gridlines
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot models
    for rank, version in enumerate(np.unique(df.version.values)):
        data = df[df.version == version]
        plt.plot(
            data.month.values, data[var].values, lw=2, color=_COLS[rank], label=version
        )

    plt.xlabel("Month", fontsize=11)
    plt.ylabel(f"{ylabel} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    plt.title("Climatology Difference")


def plot_rel_sd_var(obs_ds, obs_var, name, points, pft="all"):

    obs = average_obs_by_model(obs_ds, ILAMB_MODELS[obs_var.upper()], obs_var)

    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))

    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        obs.lon,
        obs.lat,
        obs[f"{obs_var}_rel_sd"],
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="rainbow",
        vmin=0.0,
        vmax=6.0,
    )
    if pft != "all":
        points = points[points.pft == pft]

    ax.scatter(
        points.lon,
        points.lat,
        s=15,
        c="none",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
    )

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation="horizontal")
    cbar.set_label(
        f"Observed {name} Relative Standard Deviation", size=10, fontweight="bold"
    )


def plot_pft_percent(ds: xr.Dataset, var: str, pft: str):
    """Plot PFT percentage for a surface dataset

    Args:
        ds (xr.Dataset): subset surface dataset
        var (str): variable to plot, depends on whether a natural or crop functional type
        pft (str): PFT name
    """

    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        ds.lon,
        ds.lat,
        ds[var],
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=100,
    )
    fig.colorbar(pcm, ax=ax, fraction=0.03, orientation="vertical")
    plt.title(pft)


def plot_pft_grids(ds: xr.Dataset, pft_names: list[str], title: str):
    """Plot a map of PFT grids

    Args:
        ds (xr.Dataset): input pft grid
        pft_names (list[str]): list of pft names
        title (str): title for plot
    """

    cmap = matplotlib.colors.ListedColormap(_PFT_COLS)
    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        ds.lon,
        ds.lat,
        ds.pft,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=0.5,
        vmax=16.5,
    )
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation="vertical")
    cbar.set_label("PFT", size=12, fontweight="bold")
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    cbar.set_ticklabels([pft.replace("_", " ") for pft in pft_names])
    plt.title(title)
    
def param_col_graph(ax, pivot):
    
    legend_elements = {}
    for _, row in pivot.iterrows():
    
        y = row['y']
        category = row['category']
        fates_val = row.get('FATES', 0)
        clm_val = row.get('CLM', 0)
        color = _CATEGORY_COLORS.get(category, '#cccccc')

        if fates_val > 0:
            ax.barh(y, fates_val, color=color)
            t = ax.text(fates_val - 0.5, y, str(int(fates_val)), va='center', ha='center', 
                    fontsize=12, c='white', weight='bold')
            t.set_bbox(dict(facecolor=color, alpha=1, edgecolor=color))
            legend_elements[category] = Patch(facecolor=color, label=category)

        if clm_val > 0:
            ax.barh(y, clm_val, left=fates_val, color=color, hatch='///')
            t = ax.text(fates_val + clm_val - 0.5, y, str(int(clm_val)), va='center', ha='center', 
                    fontsize=12, c='white', weight='bold')
            t.set_bbox(dict(facecolor=color, alpha=1, edgecolor=color))
            legend_elements[category] = Patch(facecolor=color, label=category)

    return legend_elements

def plot_model_oaat(ax, param_dat: pd.DataFrame, model: str):
    """Plots a column graph of number of parameters in OAAT ensemble

    Args:
        param_dat (pd.DataFrame): data frame with information about parameters
        model (str): model name for title
    """
    
    # count up totals, update names of subcategory
    param_counts_total = (param_dat.groupby(['category', 'subcategory', 'type']).size().reset_index(name='num'))
    param_counts_total['subcategory_label'] = param_counts_total['subcategory'].map(_SUBCATEGORY_LABELS)
    
    # createa pivot table
    pivot = param_counts_total.pivot_table(index=['subcategory_label', 'category'], columns='type', values='num', fill_value=0)
    
    # re-order subcategory
    subcategory_order = ['Fire', 'Land use', 'Allocation', 'Allometry', 'Decomposition',
                         'Mortality', 'Nutrient uptake', 'Phenology', 'Recruitment',
                         'Respiration', 'Vegetation dynamics', 'Acclimation',
                         'Photosynthesis', 'Vegetation water', 'Radiation', 'Sensible and latent heat',
                         'Soil thermal properties', 'Soil hydraulics', 'Snow']
    
    # get rid of subcategories that are not present
    subcategory_present = param_counts_total['subcategory_label'].unique()
    filtered_order = [s for s in subcategory_order if s in subcategory_present]
    
    subcategory_to_y = {label: i for i, label in enumerate(filtered_order)}
    
    # re-order pivot table
    pivot = pivot.reset_index()
    pivot = pivot[pivot['subcategory_label'].isin(subcategory_order)]
    pivot['y'] = pivot['subcategory_label'].map(subcategory_to_y)
    pivot = pivot.sort_values('y')

    legend_elements = param_col_graph(ax, pivot)
    
    # add clm and fates parameters
    legend_elements['CLM parameters'] = Patch(facecolor='white', edgecolor='black', hatch='///', label='CLM parameters')
    legend_elements['FATES parameters'] = Patch(facecolor='white', edgecolor='black', label='FATES parameters')
    
    yticks = [subcategory_to_y[label] for label in subcategory_order if label in pivot['subcategory_label'].unique()]
    yticklabels = [label for label in subcategory_order if label in pivot['subcategory_label'].unique()]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('')
    ax.set_title(f'{model} Parameters')

    return legend_elements

def plot_oaat_params(param_dat_fates: pd.DataFrame, model_fates: str, param_dat_clm: pd.DataFrame, model_clm: str):
    sns.set_theme(style="whitegrid")    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    legend_elements = plot_model_oaat(axes[0], param_dat_clm, model_clm)
    legend_elements = plot_model_oaat(axes[1], param_dat_fates, model_fates)

    handles = legend_elements.values()
    labels = [_CATEGORY_LABELS.get(label, label) for label in legend_elements.keys()]
    axes[1].legend(
        handles=handles,
        labels=labels,
        title=None,
        loc='lower center',
        bbox_to_anchor=(1.2, 0.5),
        ncol=1
    )
    plt.tight_layout()

def plot_area_means(ds, default_ind, sum_var, variable, ylabel, units,
                    combined_sim=False):
    
    default_sim = ds.sel(ensemble=default_ind).sel(summation_var=sum_var)
    sumvar_ds = ds.sel(summation_var=sum_var)
    sumvar_ds = sumvar_ds.where(sumvar_ds.sum_diff > 0.0, drop=True)

    ensemble_valid = sumvar_ds.ensemble.values
    ensemble_plot_index = np.arange(1, len(ensemble_valid) + 1)
    ensemble_index_map = dict(zip(ensemble_valid, ensemble_plot_index))
    
    get_blank_plot(width=9)
    plt.axhline(y=default_sim[variable].mean().values, color='k', linestyle='--', linewidth=1)
    
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    
    marker_map = {
        "fates parameters": "o",  # circle
        "clm parameters": "x",  # square
    }
    
    cats = np.unique(sumvar_ds.category.dropna(dim='ensemble'))
    if combined_sim:
        sources = np.unique(sumvar_ds.sim_source.dropna(dim='ensemble'))
    else:
        sources = ['fates parameters']
        sumvar_ds['sim_source'] = 'fates parameters'
    
    for category in cats:
        for sim in sources:
            sub = sumvar_ds.where(
                (sumvar_ds.category == category) & (sumvar_ds.sim_source == sim), drop=True
            )
            if sub[variable].count() > 0:
                x_vals = [ensemble_index_map[e.item()] for e in sub.ensemble.values]
                plt.scatter(x_vals, 
                            sub[variable], 
                            alpha=0.5, 
                            s=20,
                            color=_CATEGORY_COLORS[category],
                            marker=marker_map.get(str(sim), "x"),
                            label=category)

    category_handles = [
        Line2D([0], [0], marker='o', color='w', label=_CATEGORY_LABELS[cat],
               markerfacecolor=_CATEGORY_COLORS[cat], markersize=6, linestyle='None')
        for cat in cats
    ]
    source_labels = {
        "fates parameters": "FATES parameters",
        "clm parameters": "CLM parameters"
    }
    source_handles = [
        Line2D([0], [0], marker=marker_map[src], color='k', label=source_labels[src],
               linestyle='None', markersize=6)
        for src in sources
    ]
    
    legend1 = plt.legend(
        handles=category_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 0.9),
        borderaxespad=-0.0,
        fontsize="10",
        title=None,
        ncol=1,
        frameon=False,
    )
    if combined_sim:
        plt.legend(
            handles=source_handles,
            loc='upper left',
            bbox_to_anchor=(1.02, 0.6),
            borderaxespad=0.0,
            frameon=False,
            fontsize="10",
            ncol=1,
        )
        plt.gca().add_artist(legend1) 
    plt.xlabel('Ensemble Member')
    plt.ylabel(f"{ylabel} ({units})")
    plt.tight_layout()
    
    
def _plot_top_n(ax, ds, default, variable, biome_label=None):
    
    # loop through the rows of the sorted dataframe
    for _, row in ds.iterrows():
    
        # get the color for the current category
        category_color = _CATEGORY_COLORS.get(row['category'], '#000000')
        
        # line connecting min and max variable
        ax.plot(
            [row['min_val'], row['max_val']],
            [row['parameter_name'], row['parameter_name']],
            color=category_color,
            linewidth=1,
            zorder=1
        )

    # plot max values as filled circles with category color
    ax.scatter(
        ds['max_val'],
        ds['parameter_name'],
        c=ds['category'].map(_CATEGORY_COLORS),
        label=None,
        zorder=2
    )
    
    # plot min values as open circles with category color
    ax.scatter(
        ds['min_val'],
        ds['parameter_name'],
        facecolors='none',
        edgecolors=ds['category'].map(_CATEGORY_COLORS),
        label=None,
        zorder=2
    )
    
    ax.axvline(x=default[variable].mean().values, color='k', linestyle='--', linewidth=1)
    
    ax.set_title(biome_label)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.5)
    

    
def plot_top_n(ds, default, variable, ylabel, units, xmin=None, xmax=None, by_biomes=False):
    
    if by_biomes:
        
        biomes = np.unique(ds.biome.values)
        biomes = biomes[1:]  # don't plot ice 
        n_biomes = len(biomes)
        
        # determine grid size
        ncols = 3
        nrows = math.ceil(n_biomes/ncols)
        start_of_last_row = (nrows - 1) * ncols
        bottom_row_indices = range(start_of_last_row, nrows * ncols)
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows), 
                                 sharex=False)
        axes = axes.flatten()
        
        for i, biome in enumerate(biomes):
            
            biome_default = default.sel(biome=biome)
            biome_ds = ds[ds.biome == biome]
            
            _plot_top_n(axes[i], biome_ds, biome_default, variable, _BIOME_NAMES[biome])
            if i in bottom_row_indices:
                axes[i].set_xlabel(f'{ylabel} ({units})')
    
    else:
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        _plot_top_n(ax, ds, default, variable)
    
    
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    
    # create custom legend handles for categories
    category_handles = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=category)
        for category, color in _CATEGORY_COLORS.items()
    ]
    max_value_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='black', 
                                     markersize=9, label='Max Value')
    min_value_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='none',
                                     markeredgecolor='black',
                                     markersize=8, markeredgewidth=1, label='Min Value')
    
    default_line_handle = mlines.Line2D(
        [0], [0], color='black', linestyle='--', linewidth=1, label='Default Value'
    )
    
    # Combine the category legend and the min/max markers into one legend
    handles = [max_value_handle, min_value_handle, default_line_handle] + category_handles
    labels = ['Max parameter', 'Min parameter', 'Default'] + [_CATEGORY_LABELS[label] for label in list(_CATEGORY_COLORS.keys())]
    
    handles = handles[:3] + [mlines.Line2D([], [], color='white')] + handles[3:]
    labels = labels[:3] + [''] + labels[3:]
    
    plt.legend(
        handles=handles,
        labels=labels,
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        frameon=False
    )
    
    plt.xlabel(f'{ylabel} ({units})')
    plt.ylabel('Parameter')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

def plot_oaat_climatology(climatology_ens, default, variable, variable_name, 
                          units, interesting_ensembles):
    
    texts = []
    
    plt.figure(figsize=(12, 6))
    
    for ens in climatology_ens.ensemble.values:
        sub = climatology_ens.sel(ensemble=ens)
        plt.plot(sub[variable].month, sub[variable], color='gray', alpha=0.4)
    
    plt.plot(default[variable].month, default[variable], color='red')
    plt.plot([], [], color='k', alpha=0.4, label='Perturbed Ensembles')
    plt.plot([], [], color='red', label='Default')
    
    # Highlight & annotate interesting ensembles
    if interesting_ensembles is not None:
        for ens in interesting_ensembles:
            sub = climatology_ens.sel(ensemble=ens)
            param_name = climatology_ens.sel(ensemble=ens).parameter_name.values
            type = climatology_ens.sel(ensemble=ens).type.values
            label = f'{param_name} {type}'
            line = plt.plot(sub[variable].month, sub[variable], color='k')[0]
            x = 6
            y = sub[variable].sel(month=x).values
            texts.append(
                plt.text(
                    x + 0.2, y, label,
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.8)
                )
            )
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='->', color='black'),
        only_move={'points': 'y', 'texts': 'xy'}
    )
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper left')
    plt.xlabel('Month')
    plt.ylabel(f'{variable_name} ({units})')
    plt.xticks(ticks=range(1, 13), labels=[
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ])
    plt.tight_layout()
    
def plot_oaat_zonal(zonal_ens, default, variable, variable_name, units,
    interesting_ensembles=None):
    
    texts = []
    
    get_blank_plot()
    plt.ylim(-90, 90)
    plt.grid(
        True,
        which="both",
        axis="y",
        linestyle="--",
        linewidth=0.5,
        color="black",
        alpha=0.3,
    )
    plt.tick_params(bottom=False, top=False, left=False, right=False)

    # plot models
    for ens in zonal_ens.ensemble.values:
        sub = zonal_ens.sel(ensemble=ens)
        plt.plot(sub[variable].values, sub.lat.values, color='gray', alpha=0.4)
    plt.plot(default[variable].values, default.lat.values, color='red')
    
    plt.plot([], [], color='k', alpha=0.4, label='Perturbed Ensembles')
    plt.plot([], [], color='red', label='Default')
        
    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {variable_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
def plot_params(default_param_data, param_ds, parameter):
    ens = [int(e) for e in param_ds.ensemble]
    pfts = np.unique(param_ds.fates_pft)
    da = param_ds[parameter]
    if 'fates_pft' in da.dims:
        fig, axes = plt.subplots(4, 4, figsize=(13, 6), sharex=True, sharey=True)
        axes = axes.flatten(order="F")
        for idx, ax in enumerate(axes):
            if 'fates_plant_organs' in da.dims:
                sub = da.sel(fates_pft = pfts[idx]).isel(fates_plant_organs=0)
                sub_def = default_param_data.sel(fates_pft = pfts[idx]).isel(fates_plant_organs=0)[parameter].values
            else:
                sub = da.sel(fates_pft = pfts[idx])
                sub_def = default_param_data.sel(fates_pft = pfts[idx])[parameter].values
            ax.scatter(ens, sub, label=pfts[idx])
            ax.set_title(pfts[idx])
            ax.axhline(y=sub_def, color='r', linestyle='--')
    else:
        sub_def = default_param_data[parameter].values
        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        ax.scatter(ens, da)
        ax.axhline(y=sub_def, color='r', linestyle='--')
    plt.suptitle(parameter)
    
def plot_variable_variance(ax, ds1, ds2, ds1_name, ds2_name, variable, long_name, units,
                           obs=None):
    
    both = pd.concat([ds1, ds2])
    
    ds1_mean_val = ds1[variable].mean()
    ds1_yerr = np.array([ds1_mean_val - ds1[variable].min(),
                        ds1[variable].max() - ds1_mean_val])
                        
    ds2_mean_val = ds2[variable].mean()
    ds2_yerr = np.array([ds2_mean_val - ds2[variable].min(),
                        ds2[variable].max() - ds2_mean_val])
    g = sns.stripplot(data=both, x="model", y=variable,
                      hue="category", jitter=True, dodge=True, 
                      alpha=0.7, ax=ax,
                      palette=_CATEGORY_COLORS)

    ax.errorbar(x=[ds1_name], y=ds1_mean_val,
                yerr=[[ds1_yerr[0]], [ds1_yerr[1]]],
                fmt='o', color='black', ecolor='black',
                capsize=15)

    ax.errorbar(x=[ds2_name], y=ds2_mean_val,yerr=[[ds2_yerr[0]], [ds2_yerr[1]]],
                fmt='o', color='black', ecolor='black',
                capsize=15)
    
    if obs is not None:
        ax.axhline(obs, color='red', linestyle='--')

    ax.set_xlabel(None)
    ax.set_ylabel(f"{long_name} ({units})")
    
def plot_ensemble_variance(ds1, ds2, ds1_name, ds2_name, variables, var_dict,
                           obs=None):

    handles, labels = None, None
    fig, axes = plt.subplots(1, len(variables), figsize=(4*len(variables), 10), sharex=True)
    for i, variable in enumerate(variables):
        plot_variable_variance(axes[i], ds1, ds2, ds1_name, ds2_name,
                       variable, var_dict[variable]['long_name'],
                               var_dict[variable]['global_units'],
                               obs=obs)

        # collect legend once
        if handles is None:  
            handles, labels = axes[i].get_legend_handles_labels()
        # remove individual legend
        axes[i].legend_.remove()
        
    fig.legend(handles, [_CATEGORY_LABELS.get(lbl, lbl) for lbl in labels],
               title='Parameter Grouping', bbox_to_anchor=(1.02, 0.5),
               loc='center left', borderaxespad=0.)
    plt.tight_layout()
    
def plot_sample(ensemble, obs_mean, obs_sd, pft_id, var, units):
    
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    my_hist, _ = np.histogram(ensemble, bins=40)
    
    plt.hist(ensemble, fc="darkgray", bins=40, alpha=0.75)
    maxv = my_hist.max()

    patch = Rectangle((obs_mean - obs_sd, 0), 2*obs_sd, maxv,
                        facecolor='red', alpha=0.4, label="Observed ± SD")
    ax.add_patch(patch)
    ax.axvline(x=obs_mean, ymin=0.0, ymax=maxv, color='r', label="Observed Mean")
    
    plt.xlabel(f"{pft_id}: Annual {var} ({units})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    
def plot_mean_var(obs, lats, lons, cmap):

    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))

    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        obs.lon,
        obs.lat,
        obs,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
    )
    ax.scatter(
        lons,
        lats,
        s=15,
        c="none",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
    )

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation="horizontal")


def plot_pft_matrix(fates_to_clm, clm_index_to_name):

    # Prepare matrix
    fates_labels = list(fates_to_clm.keys())
    clm_labels = [clm_index_to_name[i] for i in range(17)]
    
    matrix = pd.DataFrame(0, index=fates_labels, columns=clm_labels)
    
    # Fill the matrix with 1s where mappings exist
    for fates, clm_indices in fates_to_clm.items():
        for idx in clm_indices:
            clm_name = clm_index_to_name.get(idx, f"CLM_{idx}")
            matrix.loc[fates, clm_name] = 1
    
    # Plot heatmap
    plt.figure(figsize=(10, 9))
    sns.heatmap(matrix, cmap='Blues', cbar=False, linewidths=0.5, linecolor='lightgray', fmt='d')
    plt.xlabel("CLM PFTs")
    plt.ylabel("FATES PFTs")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
def plot_cumulative_variance(variables, variances, var_dict):

    n_vars = len(variables)
    ncols = 2
    nrows = (n_vars + 1) // 2
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), sharex=False)
    
    for idx, variable in enumerate(variables):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col] if nrows > 1 else axs[col]
    
        width = 1.5
        fates_cumvar = variances[variable]['FATES']
        clm_cumvar = variances[variable]['CLM']
        ax.bar(fates_cumvar.nparams[:10] - width/2, fates_cumvar[:10], width=width, label='FATES')
        ax.bar(clm_cumvar.nparams[:10] + width/2, clm_cumvar[:10], width=width, label='CLM', hatch='///')
    
        ax.set_ylim([0, 1])
        ax.set_xticks(clm_cumvar.nparams[:10])
        ax.set_title(var_dict[variable]['long_name'])
    
    fig.text(0.5, 0.04, '# of parameters', ha='center', va='center', fontsize=12)
    fig.text(0.04, 0.5, 'Fraction of total variance', ha='center', va='center',
             rotation='vertical', fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False
    )
    plt.tight_layout(rect=[0.05, 0.08, 1, 1])
    
def plot_var_diff(df, variable, tol, long_name, units=None, param_sub=None, include_sd=False):

    if param_sub is not None:
        df = df[df.index.isin(param_sub)]

    if units is None:
        units = "%"
        type = "Percent"
    else:
        type = "Absolute"

    if include_sd:
        var_name = f"{variable}_mean"
        df[f"{variable}_se"] = 2.0*df[f"{variable}_sd"]/np.sqrt(20)
    else:
        var_name = variable
    
    # filter out parameters below tolerence
    below_tol = df.groupby('parameter')[var_name].apply(lambda g: (g <= tol).all().all())
    params_to_keep = below_tol[~below_tol].index
    df_filtered = df[df['parameter'].isin(params_to_keep)]
    
    # sort parameters by magnitude
    sorted_params = df_filtered.groupby('parameter')[var_name].mean().sort_values(ascending=True).index.tolist()

    # get the unique models
    models = df_filtered['model'].unique()
    bar_height = 0.35

    plt.figure(figsize=(9, 10))
    ax = plt.gca()
    
    # offset bars for each model
    for i, model in enumerate(models):
        subset = df_filtered[df_filtered['model'] == model].set_index('parameter').reindex(sorted_params)
        
        # Position bars vertically based on parameter index
        y_positions = np.arange(len(sorted_params)) + (i - 0.5) * bar_height

        ax.barh(
            y=y_positions,
            width=subset[var_name],
            height=bar_height,
            label=model,
            xerr=subset[f"{variable}_se"] if include_sd else None,
            color=_MODEL_COLS[model],
            capsize=4,
            alpha=0.9,
            edgecolor='black'
        )

        # set y-ticks to parameter names
        ax.set_yticks(np.arange(len(sorted_params)))
        ax.set_yticklabels(sorted_params)

        plt.xlabel(f"{type} Difference in {long_name} ({units})")
        plt.ylabel('Parameter')
        plt.legend(loc='lower right')
        plt.tight_layout()
    
    
def plot_2_top_n(ds1, ds2, default1, default2, variable, ylabel, units):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_top_n(axes[0], ds1, default1, variable)
    _plot_top_n(axes[1], ds2, default2, variable)
    
    axes[0].set_xlabel(f'{ylabel} ({units})')
    axes[1].set_xlabel(f'{ylabel} ({units})')
    axes[0].set_ylabel('Parameter')
    axes[0].set_title('FATES')
    axes[1].set_title('CLM')
    
    # create custom legend handles for categories
    category_handles = [
        mlines.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=category)
        for category, color in _CATEGORY_COLORS.items()
    ]
    max_value_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='black', 
                                     markersize=9, label='Max Value')
    min_value_handle = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='none',
                                     markeredgecolor='black',
                                     markersize=8, markeredgewidth=1, label='Min Value')
    
    default_line_handle = mlines.Line2D(
        [0], [0], color='black', linestyle='--', linewidth=1, label='Default Value'
    )
    
    # Combine the category legend and the min/max markers into one legend
    handles = [max_value_handle, min_value_handle, default_line_handle] + category_handles
    labels = ['Max parameter', 'Min parameter', 'Default'] + [_CATEGORY_LABELS[label] for label in list(_CATEGORY_COLORS.keys())]
    
    handles = handles[:3] + [mlines.Line2D([], [], color='white')] + handles[3:]
    labels = labels[:3] + [''] + labels[3:]
    
    fig.legend(
        handles=handles,
        labels=labels,
        loc='upper left',
        bbox_to_anchor=(0.99, 0.99),
        frameon=False
    )
    plt.tight_layout()