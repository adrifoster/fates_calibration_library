"""Functions to assist with analyzing one-at-a-time ensembles"""

import xarray as xr
import numpy as np
import os
import pandas as pd

from fates_calibration_library.analysis_functions import compute_infl, get_start_end_slopes

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
    
    not_in = [180, 181, 306, 307, 308, 309, 312, 313, 314,
          315, 316, 317, 318, 319, 320, 321, 322, 323,
          324, 325, 326, 327, 328, 329, 330, 331, 332,
          333]
    
    clm_param_dat = pd.read_csv(param_list)
    clm_param_dat.columns = ['parameter_name', 'ensemble', 'type', 'category', 'subcategory']
    clm_param_dat.ensemble = [int(e.replace('CLM6SPoaat', '')) for e in clm_param_dat.ensemble]
    clm_param_dat = clm_param_dat[~clm_param_dat.ensemble.isin(not_in)]
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

def get_area_means_diffs(file: str, param_info: xr.Dataset, out_vars: list[str], 
                         default_ind: int=0) -> xr.Dataset:
    """Gets the sum of all differences between mean and iav for across all history variables
    for each ensemble member

    Args:
        file (str): path to ensemble dataset
        param_info (xr.Datset): data frame with information about parameters
        out_vars (list[str]): list of output variables
        default_ind (int, optional): index of default simulation. Defaults to 0.

    Returns:
        xr.Dataset: output dataset with differences
    """
    
    ds = xr.open_dataset(file)
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

def get_combined(ds1, ds2, name1, name2):

    ds1 = ds1.assign(sim_source=("ensemble", [name1] * ds1.sizes['ensemble']))
    ds2 = ds2.assign(sim_source=("ensemble", [name2] * ds2.sizes['ensemble']))
    
    ds2_shifted = ds2.assign_coords(ensemble=ds2.ensemble + ds1.sizes['ensemble'])
    
    return xr.concat([ds1, ds2_shifted], dim="ensemble")

def get_min_max_diff(ds: xr.Dataset) -> pd.DataFrame:
    """Gets differences between min and max ensemble members for all variables

    Args:
        ds (xr.Dataset): ensemble dataset
        sumvar (str): summation variable ['mean', 'iav']

    Returns:
        pd.DataFrame: output dataframe
    """

    # we don't want to look at these data variables
    skip_vars = ['parameter_name', 'type', 'category', 'subcategory', 'sum_diff',
                 'sim_source']
    vars_to_check = [v for v in ds.data_vars if v not in skip_vars]
    
    default_ds = ds.where(ds.ensemble == 0, drop=True)

    # group by parameter name
    grouped = ds.groupby('parameter_name')
    diffs = {}
    for param, group in grouped:
        # select the min and max rows
        if (group.type == 'min').any():
            min_val = group.where(group.type == 'min', drop=True)
        else:
            min_val = default_ds

        # Check if 'max' exists in the group
        if (group.type == 'max').any():
            max_val = group.where(group.type == 'max', drop=True)
        else:
            max_val = default_ds
        
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

def get_top_n(ds: xr.Dataset, df_diffs: pd.DataFrame, variable: str, n: int,
              exclude_list=None) -> pd.DataFrame:
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
    if exclude_list is not None:
        df_diffs = df_diffs.loc[~df_diffs.index.isin(exclude_list)]
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


def get_ensemble_slopes(ds, fates_param_dat,
                        skip_vars=['category', 'subcategory', 'type', 'parameter_name']):
    if skip_vars is None:
        skip_vars = []
    slope_start_vars = {}
    slope_end_vars = {}

    for variable in ds.data_vars:
        if variable in skip_vars:
            continue
        if 'month' not in ds[variable].dims:
            continue
        try:
            infl_months = compute_infl(ds[variable].mean(dim='ensemble'))
            slopes_start = []
            slopes_end = []
            for ens in ds.ensemble.values:
                da_ens = ds[variable].sel(ensemble=ens)
                slope_start, slope_end = get_start_end_slopes(da_ens, infl_months)
                slopes_start.append(slope_start)
                slopes_end.append(slope_end)
            
            slope_start_da = xr.DataArray(slopes_start, 
                                          coords={'ensemble': ds.ensemble},
                                          dims='ensemble')
            slope_end_da = xr.DataArray(slopes_end, 
                                        coords={'ensemble': ds.ensemble},
                                        dims='ensemble')
            slope_start_vars[variable] = slope_start_da
            slope_end_vars[variable] = slope_end_da
            
        except Exception as e:
            print(f"Skipping variable {variable} due to error: {e}")
            continue
    
    slope_start_ds = xr.Dataset(slope_start_vars)
    slope_end_ds = xr.Dataset(slope_end_vars)
    slope_start_ds = xr.merge([slope_start_ds, fates_param_dat])
    slope_end_ds = xr.merge([slope_end_ds, fates_param_dat])
    
    return slope_start_ds, slope_end_ds

def get_nonzero_params(ds, var='sum_diff'):
    return np.unique(ds.where(ds[var] > 0.0, drop=True).parameter_name.values)

def get_params(fates_ds, fates_clm_ds, clm_ds, var='sum_diff'):
    
    fates_only_parameters = get_nonzero_params(fates_ds, var=var)
    fates_clm_parameters = get_nonzero_params(fates_clm_ds, var=var)
    clm_parameters = get_nonzero_params(clm_ds, var=var)
    
    clm_only_parameters = [param for param in clm_parameters if param not in fates_clm_parameters]
    shared_parameters = [param for param in clm_parameters if param in fates_clm_parameters]

    out_dict = {'fates_only': fates_only_parameters,
                'fates_clm': fates_clm_parameters,
                'clm_parameters': clm_parameters,
                'clm_only': clm_only_parameters,
                'shared': shared_parameters}
    
    return out_dict

def get_fates_paramdiffs(param_dir, param_prefix, default_param, fates_params,
                         param_dat):

    diffs = {}
    for param in fates_params:
        diffs[param] = {}
        ensembles = param_dat.where(param_dat.parameter_name == param, drop=True)
        min = ensembles.where(ensembles.type == 'min', drop=True).ensemble.values[0]
        max = ensembles.where(ensembles.type == 'max', drop=True).ensemble.values[0]
        min_file = os.path.join(param_dir, f"{param_prefix}{str(min).zfill(3)}.nc")
        max_file = os.path.join(param_dir, f"{param_prefix}{str(max).zfill(3)}.nc")
        ds_min = xr.open_dataset(min_file)
        ds_max = xr.open_dataset(max_file)
    
        if param == 'fates_stoich_nitr_1':
            pmax = np.mean(ds_max['fates_stoich_nitr'].isel(fates_plant_organs=0).values)
            pmin = np.mean(ds_min['fates_stoich_nitr'].isel(fates_plant_organs=0).values)
            pbaseline = np.mean(default_param['fates_stoich_nitr'].isel(fates_plant_organs=0).values)
        else:
            pmax = np.mean(ds_max[param].values)
            pmin = np.mean(ds_min[param].values)
            pbaseline = np.mean(default_param[param].values)
            
        diffs[param]['min_reldiff'] = np.abs(pmin - pbaseline)/pbaseline
        diffs[param]['max_reldiff'] = np.abs(pmax - pbaseline)/pbaseline
        diffs[param]['min_diff'] = np.abs(pmin - pbaseline)
        diffs[param]['max_diff'] = np.abs(pmax - pbaseline)
        diffs[param]['diff'] = np.abs(pmax - pmin)
        diffs[param]['reldiff'] = np.abs(pmax - pmin)/np.abs(pbaseline)

    return pd.DataFrame.from_dict(diffs, orient='index')

def get_clm_paramdiffs(param_dir, param_prefix, default_param, clm_params,
                        param_dat, nlmods):
    
    diffs = {}
    for param in clm_params:
        diffs[param] = {}
        ensembles = param_dat.where(param_dat.parameter_name == param, drop=True)
        
        if param in default_param.data_vars:
            pbaseline = np.mean(default_param[param].values)
        else: 
            nlmods_sub = nlmods[nlmods.parameter_name == param]
            pbaseline = nlmods_sub['default_value'].values[0]

        if (ensembles.type == 'min').any():
            
            min = ensembles.where(ensembles.type == 'min', drop=True).ensemble.values[0]
            
            if param in default_param.data_vars:
                min_file = os.path.join(param_dir, f"{param_prefix}{str(min).zfill(4)}.nc")
                ds_min = xr.open_dataset(min_file)
                pmin = np.mean(ds_min[param].values)
            else:
                nlmods_sub = nlmods[nlmods.parameter_name == param]
                pmin = nlmods_sub['min_value'].values[0]

            diffs[param]['min_reldiff'] = np.abs(pmin - pbaseline)/pbaseline
            diffs[param]['min_diff'] = np.abs(pmin - pbaseline)
        else:
            diffs[param]['min_reldiff'] = np.nan
            diffs[param]['min_diff'] = np.nan
            pmin = pbaseline

        if (ensembles.type == 'max').any():
            max = ensembles.where(ensembles.type == 'max', drop=True).ensemble.values[0]
            
            if param in default_param.data_vars:
                max_file = os.path.join(param_dir, f"{param_prefix}{str(max).zfill(4)}.nc")
                ds_max = xr.open_dataset(max_file)
                pmax = np.mean(ds_max[param].values)
            else:
                nlmods_sub = nlmods[nlmods.parameter_name == param]
                pmax = nlmods_sub['max_value'].values[0]
            
            diffs[param]['max_reldiff'] = np.abs(pmax - pbaseline)/pbaseline
            diffs[param]['max_diff'] = np.abs(pmax - pbaseline)
        else:
            diffs[param]['max_reldiff'] = np.nan
            diffs[param]['max_diff'] = np.nan
            pmax = pbaseline
        
        diffs[param]['diff'] = np.abs(pmax - pmin)
        diffs[param]['reldiff'] = np.abs(pmax - pmin)/np.abs(pbaseline)
        
    return pd.DataFrame.from_dict(diffs, orient='index')

def get_vardiff(da, baseline_dat, variables, params, reldiff=False):
    
    all_var_dfs = {}
    for variable in variables:
        var_diffs = {}

        for param in params:
            var_diffs[param] = {}

            dat = da.where(da.parameter_name == param, drop=True)
            
            if (dat.type == 'min').any():
                var_min = dat.where(dat.type == 'min', drop=True)
            else:
                var_min = da.isel(ensemble=0)
            
            if (dat.type == 'max').any():
                var_max = dat.where(dat.type == 'max', drop=True)
            else:
                var_max = da.isel(ensemble=0)
            
            if reldiff:
                var_diff = np.abs(var_max[variable].values - var_min[variable].values)/baseline_dat[variable].values*100.0
            else:
                var_diff = np.abs(var_max[variable].values - var_min[variable].values)
            
            diff = var_diff[0]
            var_diffs[param][variable] = diff

        var_df = pd.DataFrame.from_dict(var_diffs, orient='index')
        all_var_dfs[variable] = var_df
    
    return pd.concat(all_var_dfs.values(), axis=1)

def get_S1diff(da, baseline_dat, variables, params, diff_df, reldiff=False):
    
    all_var_dfs = {}
    for variable in variables:
        var_diffs = {}

        for param in params:
            var_diffs[param] = {}

            param_diffs = diff_df[diff_df.index == param]
            dat = da.where(da.parameter_name == param, drop=True)
            
            if (dat.type == 'min').any():
                var_min = dat.where(dat.type == 'min', drop=True)
                if reldiff:
                    var_mindiff = (np.abs(var_min[variable].values - baseline_dat[variable].values)/baseline_dat[variable].values)
                else:
                    var_mindiff = np.abs(var_min[variable].values - baseline_dat[variable].values)
                s1_min = var_mindiff[0]/(param_diffs.min_reldiff.values[0])
            else:
                s1_min = np.nan
            
            if (dat.type == 'max').any():
                var_max = dat.where(dat.type == 'max', drop=True)
                if reldiff:
                    var_maxdiff = (np.abs(var_max[variable].values - baseline_dat[variable].values)/baseline_dat[variable].values)
                else:
                    var_maxdiff = np.abs(var_max[variable].values - baseline_dat[variable].values)
                s1_max = var_maxdiff[0]/(param_diffs.max_reldiff.values[0])
            else:
                s1_max = np.nan
                
            s1 = np.abs(np.nanmean([s1_min, s1_max]))
            var_diffs[param][variable] = s1

        var_df = pd.DataFrame.from_dict(var_diffs, orient='index')
        all_var_dfs[variable] = var_df
    return pd.concat(all_var_dfs.values(), axis=1)

def get_param_variance(parameters, variable, ds, default_ind):
    
    default = ds.isel(ensemble=default_ind).sel(summation_var='mean')
    
    variances = []
    for parameter in parameters:
        this_par = ds.where(ds.parameter_name == parameter, drop=True).sel(summation_var='mean')
        if (this_par.type == 'min').any():
            min_par = this_par.where(this_par.type == 'min', drop=True)
        else:
            min_par = default
        if (this_par.type == 'max').any():
            max_par = this_par.where(this_par.type == 'max', drop=True)
        else:
            max_par = default
    
        variance = (default[variable].values - min_par[variable].values)**2 + (max_par[variable].values - default[variable].values)**2
        variances.append(variance[0])
    
    return pd.DataFrame({'parameter_name': parameters, 'variance': variances})

def get_cumulative_variance(df, parameters, param_chunks):

    df = df.set_index('parameter_name').to_xarray()
    chunks = xr.DataArray(param_chunks + param_chunks*np.floor(np.arange(len(parameters))/param_chunks),
                          dims='parameter_name', name='nparams')
    return df['variance'].sortby(df['variance'], ascending=False).groupby(chunks).sum().cumsum(dim='nparams')/df['variance'].sum()

def get_categorical_cumulative_variance(df, param_info, parameters, param_chunks):
    
    categories = pd.merge(df, param_info.drop_duplicates(), on='parameter_name', how='inner')
    
    categories = categories.sort_values(by='variance', ascending=False)
    categories['chunk'] = param_chunks + param_chunks*np.floor(np.arange(len(parameters))/param_chunks)
    
    counts = categories.groupby(['chunk', 'category']).size().reset_index(name='n')
    counts['freq'] = counts.groupby('chunk')['n'].transform(lambda x: x / x.sum())
    
    variance = get_cumulative_variance(df, parameters, param_chunks).to_dataset().to_pandas()

    grouped = pd.merge(variance, counts, on='chunk', how='inner')
    grouped['cum_sum_cat'] = grouped.variance*grouped.freq
    subset = grouped[grouped.chunk <= 50]
    
    return subset.pivot(index='chunk', columns='category', values='cum_sum_cat').fillna(0)
