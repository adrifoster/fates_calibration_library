"""Methods for plotting ILAMB data"""

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr

from fates_calibration_library.analysis_functions import get_zonal
from fates_calibration_library.ilamb_functions import get_model_da
from fates_calibration_library.plotting_functions import (
    round_down,
    round_up,
    get_blank_plot,
    generate_subplots,
    map_function,
)

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

def plot_month_of_max(da, varname):
    
    models = da.model.values
    num_plots = len(models)
    
    # get the emtpy subplots
    figure, axes = generate_subplots(num_plots)

    if num_plots > 1:
        axes = axes.flatten(order="F")
        for idx, ax in enumerate(axes):
            pcm = map_function(
                ax,
                da.sel(model=models[idx]),
                models[idx],
                'jet',
                0.5,
                12.5,
            )
        cbar = figure.colorbar(
            pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation="horizontal"
        )
    else:
        pcm = map_function(
            axes[0],
            da.sel(model=models[0]),
            models[0],
            'jet',
            0.5,
            12.5,
        )
        cbar = figure.colorbar(pcm, ax=axes[0], shrink=0.5, orientation="horizontal")
    
    cbar.set_label('Month', size=12, fontweight='bold')
    cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    cbar.set_ticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                     'Sep', 'Oct', 'Nov', 'Dec'])
    figure.suptitle(f'Month of Max for {varname}')
    

def plot_global(
    da: xr.DataArray,
    varname: str,
    units: str,
    cmap: str,
    vlims: list[float] = [],
    diverging_cmap: bool = False,
):
    """Plots a global data array of ILAMB models, one subplot per model

    Args:
        da (xr.DataArray): data array
        varname (str): variable name for legend
        units (str): units for legend
        cmap (str): colormap to use
        vlims (list[float]): variables limits, defaults to empty
        diverging_cmap (bool, optional): whether the colormap is a diverging scale.
                                    Defaults to False.
    """

    if len(vlims) == 0:
        vmin = da.min().values
        vmax = da.max().values
    else:
        vmin = vlims[0]
        vmax = vlims[1]
    models = da.model.values
    num_plots = len(models)

    # get the emtpy subplots
    figure, axes = generate_subplots(num_plots)

    if num_plots > 1:
        axes = axes.flatten(order="F")
        for idx, ax in enumerate(axes):
            pcm = map_function(
                ax,
                da.sel(model=models[idx]),
                models[idx],
                cmap,
                vmin,
                vmax,
                diverging_cmap=diverging_cmap,
            )
        cbar = figure.colorbar(
            pcm, ax=axes.ravel().tolist(), shrink=0.5, orientation="horizontal"
        )
    else:
        pcm = map_function(
            axes[0],
            da.sel(model=models[0]),
            models[0],
            cmap,
            vmin,
            vmax,
            diverging_cmap=diverging_cmap,
        )
        cbar = figure.colorbar(pcm, ax=axes[0], shrink=0.5, orientation="horizontal")
    cbar.set_label(f"{varname} ({units})", size=10, fontweight="bold")
    figure.suptitle(f"Observed {varname} from ILAMB Models")


def plot_by_lat(
    da: xr.DataArray,
    units: str,
    var: str,
    varname: str,
    conversion_config: dict = None,
):
    """Plots zonal (by latitude) ILAMB data for each model

    Args:
        da (xr.DataArray): data array
        units (str): units description for by_latitude data
        var (str): variable plotting
        varname (str): variable name for axes
        conversion_config (dict, optional): configuration dictionary with 'land_area' and
                'conversion_factor'. Defaults to None.
    """

    conversion_config = conversion_config or {
        "land_area": None,
        "conversion_factor": None,
    }

    # get by latitude
    by_lat = get_zonal(
        da,
        units,
        conversion_config["land_area"],
        conversion_config["conversion_factor"],
    )

    # turn into pandas data frame for easier plotting
    df = pd.DataFrame(
        {
            "lat": np.tile(by_lat.lat, len(by_lat.model)),
            "model": np.repeat(by_lat.model, len(by_lat.lat)),
            var: by_lat.values.flatten(),
        }
    )

    # get min/max values
    minval = df[var].min()
    minvar = round_up(np.abs(minval)) * -1.0 if minval < 0 else round_down(minval)
    maxvar = round_up(df[var].max())

    # get a blank plot
    get_blank_plot()

    # add latitude-specific ticks/lines
    plt.xlim(minvar, maxvar)
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

    # plot models
    for rank, model in enumerate(np.unique(df.model.values)):
        data = df[df.model == model]
        plt.plot(data[var].values, data.lat.values, lw=2, color=_COLS[rank], label=model)

    plt.ylabel("Latitude (º)", fontsize=11)
    plt.xlabel(f"Annual {varname} ({units})", fontsize=11)
    plt.title(
        f"Observed Annual {varname}" + " by latitude for different data products",
        fontsize=11,
    )
    plt.legend(loc="upper right")
    
def plot_annual_cycle(da, var, ylabel, units, title):
    df = pd.DataFrame({
        "month": np.tile(da.month, len(da.model)),
        "model": np.repeat(da.model, len(da.month)),
        var: da.values.flatten()}
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
    for rank, model in enumerate(np.unique(df.model.values)):
        data = df[df.model == model]
        plt.plot(data.month.values, data[var].values, lw=2, color=_COLS[rank], label=model)

    plt.xlabel("Month", fontsize=11)
    plt.ylabel(f"{ylabel} ({units})", fontsize=11)
    plt.legend(loc="upper right")
    plt.title(title)

def plot_ilamb_var(
    ilamb_dat: xr.Dataset,
    var: str,
    plot_config: dict,
    vlims: list[float] = [],
):
    """Plots ILAMB data, globally and by latitude, for a variable for all models

    Args:
        ilamb_dat (xr.Dataset): input ILAMB compiled dataset
        var (str): variable to plot
        plot_config (dict): configuration dictionary with keys:
            - models (list[str]): list of ILAMB models
            - conversion_factor (float): conversion factor for going to latitude sums
            - varname (str): variable name for plotting
            - global_units (str): global units for axes
            - lat_units (str): latitude units for axes
            - cmap (str): color map for global plot
            - diverging_cmap (bool): whether the cmap is diverging or not
        vlims (list[float]): variables limits [min,max], defaults to empty
    """

    # get the annual data for just this variable
    da_annual = get_model_da(ilamb_dat, var, plot_config["models"])

    # plot globally
    plot_global(
        da_annual,
        plot_config["varname"],
        plot_config["global_units"],
        plot_config["cmap"],
        vlims=vlims,
        diverging_cmap=plot_config["diverging_cmap"],
    )
    
    # get the month of max data for just this variable 
    da_month = get_model_da(ilamb_dat, f"{var}_month_of_max", plot_config["models"])

    # get conversion factor
    conversion_factor = plot_config.get("conversion_factor", None)
    
    # plot month of max
    plot_month_of_max(da_month, plot_config["varname"])

    # plot by latitude
    if conversion_factor is not None:
        conversion_dict = {
            "land_area": ilamb_dat.land_area,
            "conversion_factor": conversion_factor,
        }
        plot_by_lat(
            da_annual, plot_config["lat_units"], var, plot_config["varname"], conversion_dict
        )
    else:
        plot_by_lat(da_annual, plot_config["lat_units"], var, plot_config["varname"])
        
    # get the climatology
    da_cycle = get_model_da(ilamb_dat, f"{var}_cycle", plot_config["models"])
    plot_annual_cycle(da_cycle, f"{var}_cycle", f"Annual Cycle of {plot_config['varname']}", 
                      plot_config["lat_units"].replace("yr", "month"),
                      f"Observed Annual Cycle of {plot_config['varname']}" + " for different data products")
    
    da_anomaly = get_model_da(ilamb_dat, f"{var}_anomaly", plot_config["models"])
    plot_annual_cycle(da_anomaly, f"{var}_anomaly", f"{plot_config['varname']} Anomaly", 
                      plot_config["lat_units"].replace("yr", "month"),
                      f"Observed {plot_config['varname']} Anomaly" + " for different data products")
