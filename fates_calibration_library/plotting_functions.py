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

from fates_calibration_library.analysis_functions import month_difference
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

def plot_zonal_mean_diff(da1: xr.DataArray, da2: xr.DataArray, ds1_name: str, 
                           ds2_name: str, var: str, long_name: str, units: str):
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
    ds = xr.concat([da1.to_dataset(name=var), da2.to_dataset(name=var)], dim='version')
    ds = ds.assign_coords(version=("version", [ds1_name, ds2_name]))
    
    df = pd.DataFrame({
        "lat": np.tile(ds.lat, len(ds.version)),
        "version": np.repeat(ds.version, len(ds.lat)),
        var: ds[var].values.flatten()}
    )
    
    # plot
    get_blank_plot()
    plt.ylim(-90, 90)
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    
    # plot models
    for rank, version in enumerate(np.unique(df.version.values)):
        data = df[df.version == version]
        plt.plot(data[var].values, data.lat.values, lw=2, color=_COLS[rank], label=version)

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {long_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    plt.title("Zonal Mean Difference")
    
def plot_model_obs_climatology_diff(ilamb_var, model_var, var_name, model_name, long_name, units):

    ilamb_df = pd.DataFrame({
        "month": np.tile(ilamb_var.month, len(ilamb_var.model)),
        "model": np.repeat(ilamb_var.model, len(ilamb_var.month)),
        var_name: ilamb_var.values.flatten()}
    )
    
    
    get_blank_plot()
    
    # add latitude-specific ticks/lines
    plt.xlim(1, 12)
    plt.xticks(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=10)

    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    
    # plot models
    for rank, model in enumerate(np.unique(ilamb_df.model.values)):
        data = ilamb_df[ilamb_df.model == model]
        plt.plot(data.month.values, data[var_name].values, lw=2, color=_COLS[rank], label=model)
    
    plt.plot(model_var.month.values, model_var.values, lw=2, color='black', label='FATES')
    
    plt.xlabel("Month", fontsize=11)
    plt.ylabel(f"{long_name} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    
def plot_model_obs_zonal_diff(ilamb_var, model_var, land_area, conversion_factor, var_name, model_name, long_name, units):

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
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    
    # plot models
    for rank, model in enumerate(np.unique(df.model.values)):
        data = df[df.model == model]
        plt.plot(data[var_name].values, data.lat.values, lw=2, color=_COLS[rank], label=model)
    
    plt.plot(model_var.values, model_var.lat.values, lw=2, color='black', label='FATES')
    
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
    elif num_plots < 11:
        return math.ceil(num_plots / 2), 2
    else:
        # I've chosen to have a maximum of 3 columns
        return math.ceil(num_plots / 3), 3
    
def plot_model_obs_diff(model_ds, obs_da, land_frac, var, units, model_name):

    sub_list = []
    for model in obs_da.model:
        sub = obs_da.sel(model=model)
        sub = sub.where(np.abs(sub > 0.0))
        sub_list.append(sub)

    obs_da = xr.concat(sub_list, dim='model')

    mean_diff = model_ds[var]*land_frac - obs_da.mean(dim='model')
    vmax = mean_diff.max().values
    vmin = mean_diff.min().values
    
    models = obs_da.model.values
    figure, axes = generate_subplots(len(models)+1)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx < len(models):
            obs_model = obs_da.sel(model=models[idx])
            diff = model_ds[var]*land_frac - obs_model
            title = f"{model_name} - {models[idx]}"
        else:
            diff = mean_diff
            title = f"{model_name} - {len(models)}-model average"
        
        pcm = map_function(ax, diff, title, 'RdBu_r', vmin, vmax, diverging_cmap=True)
    cbar = figure.colorbar(pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation="horizontal")
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

def plot_two_model_diff(da1, da2, ds1_name, ds2_name, fates_var, units, cmap):

    vmin = np.min([da1.min().values, da2.min().values])
    vmax = np.max([da1.max().values, da2.max().values])
    
    figure, axes = generate_subplots(3)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx == 0:
            pcm = map_function(ax, da2, ds2_name, cmap, vmin, vmax,
                                        diverging_cmap=False)
        elif idx == 2:
            pcm = map_function(ax, da1, ds1_name, cmap, vmin, vmax,
                                        diverging_cmap=False)
        elif idx == 1:
            diff = da2 - da1
            pcmdiff = map_function(ax, diff, f'{ds2_name} - {ds1_name}', 'RdBu_r',
                                            diff.min().values, diff.min().values,
                                            diverging_cmap=True)
    cbar1 = figure.colorbar(pcm, ax=axes[2], shrink=1, orientation="vertical")
    cbar1.set_label(f"{fates_var} ({units})", size=10, fontweight="bold")
    
    cbar2 = figure.colorbar(pcmdiff, ax=axes[1], shrink=1, orientation="horizontal")
    cbar2.set_label(f"{fates_var} Difference ({units})", size=10, fontweight="bold")
    
    figure.suptitle(f'Comparison for {fates_var}')
    
def plot_month_of_max_diff(da1, da2, ds1_name, ds2_name, fates_var):

    figure, axes = generate_subplots(3)
    axes = axes.flatten(order="F")
    for idx, ax in enumerate(axes):
        if idx == 0:
            pcm = map_function(ax, da2, ds2_name, 'jet', 0.5, 12.5,
                                        diverging_cmap=False)
        elif idx == 2:
            pcm = map_function(ax, da1, ds1_name, 'jet', 0.5, 12.5,
                                        diverging_cmap=False)
        elif idx == 1:
            diff = month_difference(da1, da2)
            pcmdiff = map_function(ax, diff, f'{ds2_name} - {ds1_name}', 'PRGn',
                                            -5, 5,
                                            diverging_cmap=True)
    cbar1 = figure.colorbar(pcm, ax=axes[2], shrink=1, orientation="vertical")
    cbar1.set_label('Month', size=12, fontweight='bold')
    cbar1.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    cbar1.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                     'Sep', 'Oct', 'Nov', 'Dec'])
    
    cbar2 = figure.colorbar(pcmdiff, ax=axes[1], shrink=1, orientation="horizontal")
    cbar2.set_label("Difference in Month of Max", size=10, fontweight="bold")
    
    figure.suptitle(f'Comparison for Month of Max for {fates_var}')


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


def plot_heatmap(summary_df):
    """Plot a heatmap of dataset means and relative differences."""
    # create a mask: Keep only 'Relative Difference (%)' for coloring
    mask = summary_df.copy()
    mask.loc[:, mask.columns != "Relative Difference (%)"] = np.nan 

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(mask, annot=True, fmt=".2f", 
                cmap="coolwarm", center=0, linewidths=0.5, linecolor="gray",
                ax=ax, cbar_kws={"label": "Relative Difference (%)"})

    # mannually add text for the other columns (to keep them uncolored)
    for i in range(summary_df.shape[0]):
        for j in range(summary_df.shape[1]):
            if summary_df.columns[j] != "Relative Difference (%)":
                text = f"{summary_df.iloc[i, j]:.2f}" if not np.isnan(summary_df.iloc[i, j]) else ""
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=10, color="black")

    plt.title("Dataset Means and Differences")
    plt.yticks(rotation=0)
    
def summarize_differences(ds1, ds2, ds1_name, ds2_name, var_dict):
    """Summarize global differences between two xarray datasets, handling Dask arrays and adding units."""
    
    summary = []
    for var in ds1.data_vars:
        if var in ds2:
            
            unit = var_dict[var]['global_units']  
            unit_str = f" ({unit})" if unit else "" 
            mean1 = ds1[var].values  
            mean2 = ds2[var].values
            diff = mean2 - mean1
            rel_diff = (diff / mean1 * 100) if mean1 != 0 else None

            # Append data with unit in the variable name
            summary.append([f"{var}{unit_str}", f"Mean of {ds1_name}", mean1.item()])
            summary.append([f"{var}{unit_str}", f"Mean of {ds2_name}", mean2.item()])
            summary.append([f"{var}{unit_str}", "Absolute Difference", diff.item() if diff is not None else None])
            summary.append([f"{var}{unit_str}", "Relative Difference (%)", rel_diff.item() if rel_diff is not None else None])

    # convert list to DataFrame
    summary_df = pd.DataFrame(summary, columns=["Variable", "Statistic", "Value"])
    summary_df = summary_df.pivot(index="Variable", columns="Statistic", values="Value")

    # reorder columns
    desired_order = [f"Mean of {ds1_name}", f"Mean of {ds2_name}", "Absolute Difference", "Relative Difference (%)"]
    summary_df = summary_df[desired_order]

    return summary_df
    
def plot_annual_cycle_diff(da1: xr.DataArray, da2: xr.DataArray, ds1_name: str, 
                           ds2_name: str, var: str, ylabel: str, units: str):
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
    ds = xr.concat([da1.to_dataset(name=var), da2.to_dataset(name=var)], dim='version')
    ds = ds.assign_coords(version=("version", [ds1_name, ds2_name]))
    
    df = pd.DataFrame({
        "month": np.tile(ds.month, len(ds.version)),
        "version": np.repeat(ds.version, len(ds.month)),
        var: ds[var].values.flatten()}
    )
    
    # plot
    get_blank_plot()
    plt.xlim(1, 12)
    plt.xticks(range(1, 13, 1), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 
                                 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=10)
    
    # add gridlines
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='black', alpha=0.3)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    
    # plot models
    for rank, version in enumerate(np.unique(df.version.values)):
        data = df[df.version == version]
        plt.plot(data.month.values, data[var].values, lw=2, color=_COLS[rank], label=version)

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
