"""Functions to assist with plotting"""

import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geocollection import GeoQuadMesh
import seaborn as sns
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
    # maximum of 3 columns
    return math.ceil(num_plots / 3), 3

def generate_subplots(
    num_plots: int, row_wise: bool = False,
    width=13, height=6,
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
        figsize=(width, height),
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
            summary.append([f"{var_dict[var]['long_name']}{unit_str}", f"Mean of {ds1_name}", mean1.item()])
            summary.append([f"{var_dict[var]['long_name']}{unit_str}", f"Mean of {ds2_name}", mean2.item()])
            summary.append(
                [
                    f"{var_dict[var]['long_name']}{unit_str}",
                    "Absolute Difference",
                    diff.item() if diff is not None else None,
                ]
            )
            summary.append(
                [
                    f"{var_dict[var]['long_name']}{unit_str}",
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
    
    
def plot_sample(ensemble, obs_mean, obs_sd, pft_id, var, units):
    
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    my_hist, _ = np.histogram(ensemble, bins=40)
    
    plt.hist(ensemble, fc="darkgray", bins=40, alpha=0.75)
    maxv = my_hist.max()

    patch = Rectangle((obs_mean - obs_sd, 0), 2*obs_sd, maxv,
                        facecolor='red', alpha=0.4, label="Observed ± SD")
    ax.add_patch(patch)
    ax.axvline(x=obs_mean, ymin=0.0, ymax=maxv, color='r', label="Observed Mean")
    
    plt.xlabel(f"{pft_id}: Annual {var} ({units})", fontsize=16)
    plt.ylabel("Count", fontsize=16)
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
            
def print_correlation(parameter_corr):
    return parameter_corr.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1).highlight_null(color='#f1f1f1').format(precision=2)

def plot_global_sensitivity(sens_df, variable, type='Sobol'):

    plt.figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
    plt.rcParams.update({'font.size': 12})

    ax = plt.subplot(1, 1, 1)
    ax.bar(sens_df.parameter, sens_df['ST'], color='lightgrey', label='interactions')
    ax.bar(sens_df.parameter, sens_df['S1'], color='darkolivegreen', label='main effects')
    ax.tick_params(axis='x', labelrotation=90)
    plt.legend(loc='upper right')
    plt.ylabel('Proportion of total \n emulated variance')
    plt.title(f'Global Sensitivity Analysis ({type}) - {variable}')
    plt.tight_layout()
    
def plot_oaat_sens(param_names, emulator, default_vals):

    num_params = len(param_names)

    # hold all parameters at median value 
    n = 50
    unif = pd.concat([pd.DataFrame(np.tile(0.5, n))]*num_params, axis=1)
    unif.columns = param_names
    
    s = np.linspace(0, 1, n)
    param = np.array([])
    oaats = np.array([])
    vars = np.array([])
    samps = np.array([])
    
    sample = unif
    
    plt.figure(figsize=[18, 16])
    for i, p in enumerate(param_names):
        
        # save old value and update to be 0-1
        save = sample[p]
        sample[p] = s

        default_val = default_vals[p].values
        
        # oaat prediction
        oaat, v = emulator(sample.values)
        oaat = oaat.numpy().flatten()
        v = v.numpy().flatten()
        ax = plt.subplot(7, 5, i + 1)
        ax.fill_between(s, oaat - 2.0*v**0.5, oaat + 2.0*v**0.5, color='peru',
                        alpha=0.4)  # shade two standard deviations
        ax.plot(s, oaat, c='k')
        ax.axvline(x=default_val, color='r', linestyle='--')
        ax.set_xlabel(p)

        oaats = np.append(oaats, oaat)
        vars = np.append(vars, v)
        param = np.append(param, np.repeat(p, n))
        samps = np.append(samps, s)
        
        # set column back to before
        sample[p] = save 

    plt.tight_layout()
    df = pd.DataFrame({'sample': samps,
      'predict': oaats,
      'variance': vars,
      'parameter': param})
    dataf = pd.DataFrame(df)
    return dataf

def plot_emulated_sample(pred_sampled, obs_mean, obs_sd, pft_id, var, units,
                         ensemble_vals=None, label_1='Emulated', label_2='FATES'):
    
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    my_hist, _ = np.histogram(pred_sampled, bins=40)
    
    plt.hist(pred_sampled, fc="darkgray", bins=40, label=label_1, alpha=0.75)
    if ensemble_vals is not None:
        plt.hist(ensemble_vals, fc="dodgerblue", bins=40, label=label_2, alpha=0.5)
        
    maxv = max(my_hist.max(), np.histogram(ensemble_vals, bins=40)[0].max() if ensemble_vals is not None else 0)
    
    patch = Rectangle((obs_mean - obs_sd, 0), 2*obs_sd, maxv,
                        facecolor='red', alpha=0.4, label="Observed ± SD")
    ax.add_patch(patch)
    ax.axvline(x=obs_mean, ymin=0.0, ymax=maxv, color='r', label="Observed Mean")
    
    plt.xlabel(f"Emulated {pft_id} Annual {var} ({units})", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    
    if ensemble_vals is not None:
        plt.legend()
        
    plt.tight_layout()
    
def plot_implausibility_histogram(implaus, tol=3.0):
    plt.figure(figsize=(7, 5))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xticks(fontsize=11)

    counts, bins = np.histogram(implaus, bins=40)
    for i in range(len(bins) - 1):
        bin_left = bins[i]
        bin_right = bins[i + 1]
        bin_center = (bin_left + bin_right) / 2
        color = "darkgray" if bin_center > tol else "dodgerblue"
        ax.bar(bin_left, counts[i], width=bin_right - bin_left, color=color, align='edge')

    custom_legend = [
        plt.Line2D([0], [0], color='dodgerblue', lw=6, label='plausible'),
        plt.Line2D([0], [0], color='darkgray', lw=6, label='implausible')
    ]
    ax.legend(handles=custom_legend)
    plt.xlabel("Implausibility Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()

def plot_emulator_validation(test_df, variable, r2, rmse):
    """Plots emulator testing set (actual vs. pred) with error bars

    Args:
        y_test (np.ndarray): Testing target of shape (n_samples,)
        y_pred (np.ndarray): Predicted target of shape (n_samples,)
        y_pred_sd (np.ndarray): Standard deviation of predictions
        r2 (float): R² score on test set
        RMSE (float): RMSE of test set
    """
    
    em_sd = np.mean(test_df[f"{variable}_sd"].values)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(test_df[f"{variable}_test"], test_df[f"{variable}_pred"])
    plt.errorbar(test_df[f"{variable}_test"], test_df[f"{variable}_pred"], 
                 yerr=2*test_df[f"{variable}_sd"], fmt="o")
    plt.plot([min(test_df[f"{variable}_test"]),
              max(test_df[f"{variable}_test"])],
              [min(test_df[f"{variable}_test"]), 
               max(test_df[f"{variable}_test"])], linestyle='--', c='k')
    plt.text(0.02, 0.98, f'$R^2$ = {r2:.2f}', fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.text(0.02, 0.93, f'Emulator std ≈ {em_sd:.2f}', 
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.text(0.02, 0.88, f'RSME = {rmse:.3f}', 
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left')
    plt.xlabel(f'FATES mean annual mean {variable}')
    plt.ylabel(f'Emulated mean annual mean {variable}')
    plt.tight_layout()
    
def plot_parameter_hists(sample, default_pars):
    
    pars = [par for par in sample.columns if par != 'run_id']
    
    plt.figure(figsize=[18, 16])
    for i, par in enumerate(pars):
        p = default_pars[par].values
        
        ax = plt.subplot(7, 5, i + 1)
        ax.hist(sample[par])
        ax.axvline(x=p, color='r', linestyle='-')
        ax.set_xlabel(par)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    