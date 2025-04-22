"""Functions to assist with analyzing one-at-a-time ensembles"""

import xarray as xr
import numpy as np
import pandas as pd

def get_fates_param_dat(fates_param_list_file: str, oaat_key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns pandas DataFrames with information about FATES parameters associated with a 
    one-at-a-time ensemble

    Args:
        fates_param_list_file (str): path to FATES parameter list file (excel)
        oaat_key (pd.DataFrame): one-at-a-time ensemble key

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: data about all parameters and just those 
        associated with the ensemble
    """
    
    # information about the parameters - only ones we can calibrate
    param_dat = pd.read_excel(fates_param_list_file)
    param_dat = param_dat[param_dat['calibrate'] == 'Y']

    # fix this - we called it 'fates_nonhydro_smpsc' in the oaat key
    param_dat["fates_parameter_name"] = param_dat["fates_parameter_name"].replace({
        "smpsc_delta": "fates_nonhydro_smpsc"
    })

    param_dat_oaat = param_dat[param_dat.fates_parameter_name.isin(np.unique(oaat_key.parameter_name))]

    # merge with key
    param_info = pd.merge(
        param_dat_oaat[['fates_parameter_name', 'category', 'subcategory']],
        oaat_key,
        left_on="fates_parameter_name",
        right_on="parameter_name",
        how="inner"
    )
    param_info = param_info.drop(columns=["fates_parameter_name"])
    param_info = param_info.set_index("ensemble").to_xarray()

    return param_info

def get_clm_param_dat(param_list):
    
    clm_param_dat = pd.read_csv(param_list)
    clm_param_dat.columns = ['parameter_name', 'ensemble', 'type', 'category', 'subcategory']
    clm_param_dat.ensemble = [int(e.replace('CLM6SPoaat', '')) for e in clm_param_dat.ensemble]
    clm_param_dat = clm_param_dat.set_index("ensemble").to_xarray()

    return clm_param_dat

def get_differences(ds: xr.Dataset, out_vars: list[str], default: xr.Dataset) -> xr.Dataset:
    """Gets differences between the default and the ensemble member for all input variables

    Args:
        ds (xr.Dataset): ensemble dataset
        out_vars (list[str]): list of variables to compare
        default (xr.Dataset): default ensemble member

    Returns:
        xr.Dataset: output difference dataset
    """
    
    diff_dfs = []
    for variable in out_vars:
        diff = np.abs(ds[variable] - default[variable])
        diff.name = 'absolute_difference'
        diff_dfs.append(diff)

    diff = xr.concat(diff_dfs, dim='variable')
    diff = diff.assign_coords(variable=("variable", out_vars))

    return diff

def get_area_means_diffs(ds: xr.Dataset, param_info: xr.Dataset, out_vars: list[str], 
                         default_ind: int=0) -> xr.Dataset:
    """Gets the sum of all differences between mean and iav for across all history variables
    for each ensemble member

    Args:
        ds (xr.Dataset): ensemble dataset
        param_info (xr.Datset): data frame with information about parameters
        out_vars (list[str]): list of output variables
        default_ind (int, optional): index of default simulation. Defaults to 0.

    Returns:
        xr.Dataset: output dataset with differences
    """

    default_mean = ds.sel(ensemble=default_ind).sel(summation_var='mean')
    default_iav = ds.sel(ensemble=default_ind).sel(summation_var='iav')
    mean_vals = ds.sel(summation_var='mean')
    iav_vals = ds.sel(summation_var='iav')

    mean_diffs = get_differences(mean_vals, out_vars, default_mean)
    mean_iavs = get_differences(iav_vals, out_vars, default_iav)

    mean_sum_diff = mean_diffs.sum(dim='variable')
    mean_iav_diff = mean_iavs.sum(dim='variable')

    ds['sum_diff'] = mean_sum_diff + mean_iav_diff
    ds = xr.merge([ds, param_info])

    return ds

def get_min_max_diff(ds: xr.Dataset) -> pd.DataFrame:
    """Gets differences between min and max ensemble members for all variables

    Args:
        ds (xr.Dataset): ensemble dataset
        sumvar (str): summation variable ['mean', 'iav']

    Returns:
        pd.DataFrame: output dataframe
    """

    # we don't want to look at these data variables
    skip_vars = ['parameter_name', 'type', 'category', 'subcategory', 'sum_diff']
    vars_to_check = [v for v in ds.data_vars if v not in skip_vars]

    # group by parameter name
    grouped = ds.groupby('parameter_name')
    diffs = {}
    for param, group in grouped:
        # select the min and max rows
        min_val = group.where(group.type == 'min', drop=True)
        max_val = group.where(group.type == 'max', drop=True)
        
        # sanity check: if either is missing, skip
        if min_val.sizes['ensemble'] == 0 or max_val.sizes['ensemble'] == 0:
            continue
        
        # assume one row per type per parameter
        min_val = min_val.isel(ensemble=0)
        max_val = max_val.isel(ensemble=0)
        
        # compute differences for each variable
        diffs[param] = {}
        for var in vars_to_check:
            diffs[param][var] = np.abs((max_val[var] - min_val[var])).item()

    df_diffs = pd.DataFrame.from_dict(diffs, orient='index')
    df_diffs.index.name = 'parameter_name'

    return df_diffs

def get_top_n(ds: xr.Dataset, df_diffs: pd.DataFrame, variable: str, n: int) -> pd.DataFrame:
    """Gets the top n ensemble members with the most impact on variable

    Args:
        ds (xr.Dataset): ensemble dataset
        df_diffs (pd.DataFrame): difference data frame
        variable (str): variable name
        n (int): number to include
        sumvar (str): summation variable ['mean' or 'iav']

    Returns:
        pd.DataFrame: output data frame
    """

    # get top n parameters for this variable
    top_params = df_diffs[variable].sort_values(ascending=False).head(n).index

    results = []
    for param in top_params:
        sub = ds.where(ds.parameter_name == param, drop=True)
        min_run = sub.where(sub.type == 'min', drop=True).isel(ensemble=0)
        max_run = sub.where(sub.type == 'max', drop=True).isel(ensemble=0)

        results.append({
            'parameter_name': param,
            'min_val': min_run[variable].item(),
            'max_val': max_run[variable].item(),
            'difference': max_run[variable].item() - max_run[variable].item(),
            'category': min_run['category'].item(),
            'subcategory': min_run['subcategory'].item()
        })
    return pd.DataFrame(results)



