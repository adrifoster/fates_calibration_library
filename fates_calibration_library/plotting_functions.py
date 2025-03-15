import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geocollection import GeoQuadMesh
import seaborn as sns


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
    elif num_plots < 11:
        return math.ceil(num_plots / 2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(num_plots / 3), 3


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
    vmax: float,
    vmin: float,
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


def get_blank_plot():
    """Generates a blank plot"""

    plt.figure(figsize=(7, 5))
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


def plot_pft_grid(colors, names, obs_data, obs_df):

    cmap = matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))
    ax.coastlines()
    ocean = ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        obs_data.lon,
        obs_data.lat,
        obs_data.biome,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=9.5,
    )
    scatter = ax.scatter(
        obs_df.lon,
        obs_df.lat,
        s=10,
        c="none",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
    )
    cbar = fig.colorbar(pcm, ax=ax, fraction=0.03, orientation="vertical")
    cbar.set_label("Biome", size=12, fontweight="bold")
    cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cbar.set_ticklabels(names)
    ax.set_extent([-50, 180, -10, 10])


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


def plot_mean_var(obs_ds, obs_var, name, units, points, cmap, pft="all"):

    obs = average_obs_by_model(obs_ds, ILAMB_MODELS[obs_var.upper()], obs_var)
    vmin = obs[obs_var].min().values
    vmax = obs[obs_var].max().values

    fig, ax = plt.subplots(figsize=(13, 6), subplot_kw=dict(projection=ccrs.Robinson()))

    ax.coastlines()
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "ocean", "110m", facecolor="white")
    )

    pcm = ax.pcolormesh(
        obs.lon,
        obs.lat,
        obs[obs_var],
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
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
    cbar.set_label(f"Observed {name} ({units})", size=10, fontweight="bold")


def plot_obs_hists(obs_df, pft, vars, names, units):

    palette, biome_names = get_biome_palette()

    pft_df = obs_df[obs_df.pft == pft]
    fig, axes = plt.subplots(figsize=(12, 12), nrows=2, ncols=2)
    axes = axes.flatten(order=("C"))
    for i, ax in enumerate(axes):
        sns.histplot(
            data=pft_df,
            x=vars[i],
            hue="biome",
            stat="count",
            edgecolor=None,
            palette=palette,
            multiple="stack",
            ax=ax,
        )
        ax.set_ylabel("Number of Gridcells", fontsize=11)
        ax.set_xlabel(f"Observed {names[i]} ({units[i]})", fontsize=11)
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        title="Biome",
        labels=np.flip([biome_names[int(b)] for b in np.unique(pft_df.biome)]),
    )
    fig.suptitle(f"Observations for {pft} grids")
    fig.tight_layout()
