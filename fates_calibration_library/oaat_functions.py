"""Functions to assist with analyzing one-at-a-time ensembles"""

import xarray as xr
import numpy as np
import os
import pandas as pd
from scipy import stats
from collections import Counter
from collections import defaultdict

from fates_calibration_library.analysis_functions import compute_infl, get_start_end_slopes

def get_fates_param_dat(fates_param_list_file: str, oaat_key: pd.DataFrame,
                        to_xarray=True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns pandas DataFrames with information about FATES parameters associated with a 
    one-at-a-time ensemble

    Args:
        fates_param_list_file (str): path to FATES parameter list file (excel)
        oaat_key (pd.DataFrame): one-at-a-time ensemble key
        to_xarray (optional, bool): whether or not to converto xarray dataset. Defaults to True

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

    # merge with key
    param_info = pd.merge(
        param_dat[['fates_parameter_name', 'long_name', 'category', 'subcategory']],
        oaat_key,
        left_on="fates_parameter_name",
        right_on="parameter_name",
    )
    param_info = param_info.drop(columns=["fates_parameter_name"])
    
    if to_xarray:
        param_info = param_info.set_index("ensemble").to_xarray()

    return param_info

def get_clm_param_dat(param_info_file, param_key, to_xarray=True):
    
    
    param_dat = pd.read_csv(param_info_file, index_col=[0]).drop(columns=['min', 'max', 'location']).drop_duplicates()
    param_dat.columns = ['parameter_name', 'long_name', 'category', 'subcategory']

    param_info = pd.merge(
        param_dat,
        param_key,
        on="parameter_name",
    )
    param_info.ensemble = [int(str(e).replace('CLM6SPoaat', '')) for e in param_info.ensemble]
    if to_xarray:
        param_info = param_info.set_index("ensemble").to_xarray()

    return param_info

def get_all_parameters(clm_param_dat, fates_param_dat):
    clm_param = clm_param_dat.to_pandas().reset_index().drop(columns=['type', 'ensemble']).drop_duplicates()
    clm_param['model'] = 'CLM'

    fates_param = fates_param_dat.to_pandas().reset_index().drop(columns=['type', 'ensemble']).drop_duplicates()
    fates_param['model'] = 'FATES'

    return pd.concat([clm_param, fates_param])


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
                         default_ind: int=0, remove_vars: list[str]=None) -> xr.Dataset:
    """Gets the sum of all differences between mean and iav for across all history variables
    for each ensemble member

    Args:
        file (str): path to ensemble dataset
        param_info (xr.Datset): data frame with information about parameters
        out_vars (list[str]): list of output variables
        default_ind (int, optional): index of default simulation. Defaults to 0.
        remove_vars (list[str], optional): list of variables to remove from ensemble. Defaults to None.

    Returns:
        xr.Dataset: output dataset with differences
    """
    
    ds = xr.open_dataset(file)
    ds['WUE'] = ds['GPP']/ds['QVEGT'].where(ds.QVEGT > 0.0)
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
    
    if remove_vars is not None:
        ds = ds.where(~ds.parameter_name.isin(remove_vars), drop=True)
        
    ds_mean = ds.sel(summation_var='mean')
    ds_iav = ds.sel(summation_var='iav')

    return ds, ds_mean, ds_iav

def get_biome_df(biome_ds, model):
    biomes = biome_ds.biome.values
    biome_diffs = []
    for biome in biomes:
        df = get_min_max_diff(biome_ds.sel(biome=biome), model)
        df['biome'] = biome
        biome_diffs.append(df)
    return pd.concat(biome_diffs)

def get_biome_top_n(biome_ds, biome_df, variable, n=10):
    biomes = biome_ds.biome.values
    topns = []
    for biome in biomes:
        biome_mean = biome_ds.sel(biome=biome)
        diff_df = biome_df[biome_df.biome == biome]
        top_n = get_top_n(biome_mean, diff_df, variable, n, biome_mean.sel(ensemble=0))
        top_n['biome'] = biome
        topns.append(top_n)
    return pd.concat(topns)

def get_combined(ds1, ds2, name1, name2):

    ds1 = ds1.assign(sim_source=("ensemble", [name1] * ds1.sizes['ensemble']))
    ds2 = ds2.assign(sim_source=("ensemble", [name2] * ds2.sizes['ensemble']))
    
    ds2_shifted = ds2.assign_coords(ensemble=ds2.ensemble + ds1.sizes['ensemble'])
    
    return xr.concat([ds1, ds2_shifted], dim="ensemble")

def get_active_ensemble_df(clm_ds, fates_ds):
    clm_active_ens = clm_ds.where(clm_ds.sum_diff > 0.0, drop=True)
    clm_active_ens = clm_active_ens.to_pandas().reset_index().drop(columns=['ensemble'])
    clm_active_ens['model'] = 'CLM'

    fates_active_ens = fates_ds.where(fates_ds.sum_diff > 0.0, drop=True)
    fates_active_ens = fates_active_ens.to_pandas().reset_index().drop(columns=['ensemble'])
    fates_active_ens['model'] = 'FATES'

    return pd.concat([clm_active_ens, fates_active_ens])

def get_min_max_diff(ds: xr.Dataset, model: str) -> pd.DataFrame:
    """Gets differences between min and max ensemble members for all variables

    Args:
        ds (xr.Dataset): ensemble dataset
        sumvar (str): summation variable ['mean', 'iav']

    Returns:
        pd.DataFrame: output dataframe
    """

    # we don't want to look at these data variables
    skip_vars = ['parameter_name', 'type', 'category', 'subcategory', 'sum_diff',
                 'sim_source', 'long_name']
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
    df_diffs['parameter'] = df_diffs.index
    df_diffs['model'] = model

    return df_diffs

def get_top_n(ds: xr.Dataset, df_diffs: pd.DataFrame, variable: str, n: int,
              default_ds, exclude_list=None) -> pd.DataFrame:
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
        
        if (sub.type == 'min').any():
            min_run = sub.where(sub.type == 'min', drop=True).isel(ensemble=0)
            category = min_run['category'].item()
            subcategory = min_run['subcategory'].item()
        else:
            min_run = default_ds

        # Check if 'max' exists in the group
        if (sub.type == 'max').any():
            max_run = sub.where(sub.type == 'max', drop=True).isel(ensemble=0)
            category = max_run['category'].item()
            subcategory = max_run['subcategory'].item()
        else:
            max_run = default_ds
    
        results.append({
            'parameter_name': param,
            'min_val': min_run[variable].item(),
            'max_val': max_run[variable].item(),
            'default': default_ds[variable].item(),
            'difference': max_run[variable].item() - min_run[variable].item(),
            'category': category,
            'subcategory': subcategory
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

def classify_params(all_params, nonzero_params):
    
    clm_only_params = all_params[all_params.parameter_name.isin(nonzero_params['clm_only'])].copy()
    clm_only_params['type'] = 'CLM only'

    fates_only_params = all_params[all_params.parameter_name.isin(nonzero_params['fates_only'])].copy()
    fates_only_params['type'] = 'FATES only'

    common_params = all_params[all_params.parameter_name.isin(nonzero_params['common'])].copy()
    common_params['type'] = 'common'

    return pd.concat([clm_only_params, fates_only_params, common_params])

def get_nonzero_params(ds, var='sum_diff'):
    return np.unique(ds.where(ds[var] > 0.0, drop=True).parameter_name.values)

def count_parameters(param_key):
    param_key = param_key[param_key.type != 'default']
    return len(param_key.parameter_name.unique())

def count_if_PFT_independent(param_key, param_dat, FATES=True):
    
    param_key = param_key[param_key.type != 'default']
    params = param_key.parameter_name.unique()

    pft_dim = 'fates_pft' if FATES else 'pft'

    pft_params = []
    global_params = []
    for parameter in params:
        if parameter in param_dat.data_vars:
            if pft_dim in param_dat[parameter].dims:
                pft_params.append(parameter)
            else:
                global_params.append(parameter)
        else:
            global_params.append(parameter)

    return len(pft_params)*16 + len(global_params)

def get_params(fates_ds, fates_clm_ds, clm_ds, var='sum_diff'):
    
    fates_fates_nonzero = get_nonzero_params(fates_ds, var=var)
    fates_clm_nonzero = get_nonzero_params(fates_clm_ds, var=var)
    clm_nonzero = get_nonzero_params(clm_ds, var=var)
    
    all_nonzero = np.unique(np.append(np.append(fates_fates_nonzero, fates_clm_nonzero), 
                                      clm_nonzero))
    
    fates_only_parameters = np.append([param for param in fates_clm_nonzero if param not in clm_nonzero],
                                      fates_fates_nonzero)
    
    clm_only_parameters = [param for param in clm_nonzero if param not in fates_clm_nonzero]
    shared_parameters = [param for param in clm_nonzero if param in fates_clm_nonzero]

    out_dict = {'fates': fates_fates_nonzero,
                'fates_clm': fates_clm_nonzero,
                'clm': clm_nonzero,
                'clm_only': clm_only_parameters,
                'fates_only': fates_only_parameters,
                'common': shared_parameters,
                'all_nonzero': all_nonzero}
    
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

def get_vardiff(da, baseline_dat, variables, params, n, reldiff=False, include_sd=True):
    
    all_var_dfs = {}
    all_var_sd_dfs = {}
    for variable in variables:
        var_diffs = {}
        var_sds = {}

        for param in params:
            var_diffs[param] = {}
            var_sds[param] = {}

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
                var_diff = (var_max.sel(summation_var='mean')[variable].values - var_min.sel(summation_var='mean')[variable].values)/baseline_dat.sel(summation_var='mean')[variable].values*100.0
                sd_diff = np.sqrt(var_max.sel(summation_var='iav')[variable].values/n + var_min.sel(summation_var='iav')[variable].values/n)/baseline_dat.sel(summation_var='mean')[variable].values*100.0
            else:
                var_diff = (var_max.sel(summation_var='mean')[variable].values - var_min.sel(summation_var='mean')[variable].values)
                sd_diff = np.sqrt(var_max.sel(summation_var='iav')[variable].values/n + var_min.sel(summation_var='iav')[variable].values/n)
            
            diff = np.atleast_1d(var_diff)[0]
            diff_sd = np.atleast_1d(sd_diff)[0]
            var_diffs[param][variable] = diff
            var_sds[param][variable] = diff_sd
            

        var_df = pd.DataFrame.from_dict(var_diffs, orient='index')
        var_sd_df = pd.DataFrame.from_dict(var_sds, orient='index')
        all_var_dfs[variable] = var_df
        all_var_sd_dfs[variable] = var_sd_df
        
    mean_df = pd.concat(all_var_dfs.values(), axis=1)
    sd_df = pd.concat(all_var_sd_dfs.values(), axis=1)
    
    if include_sd:
        return pd.merge(mean_df, sd_df, left_index=True, right_index=True, suffixes=('_mean', '_sd'))
    else:
        return mean_df

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
    
    default = ds.isel(ensemble=default_ind)
    
    variances = []
    for parameter in parameters:
        
        this_par = ds.where(ds.parameter_name == parameter, drop=True)
        
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

def find_cumulative_params(params, variable, df, cutoff=0.9):
    variance_df = get_param_variance(params, variable, df, 0)
    variance_df = variance_df.sort_values(by='variance', ascending=False).reset_index()
    variance_df['cum_sum'] = variance_df.variance.cumsum() / variance_df.variance.sum()

    mask = variance_df.cum_sum >= cutoff
    first_true = mask.idxmax()
    return variance_df.iloc[:(first_true+1)]['parameter_name'].values

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

def get_ensemble_ranges(ensemble_df, vars):
    mean_vals = {}
    max_vals = {}
    min_vals = {}
    diff = {}
    stds = {}
    variance = {}
    q1s = {}
    q3s = {}
    for variable in vars:
        mean_vals[variable] = ensemble_df[variable].mean()
        max_vals[variable] = ensemble_df[variable].max()
        min_vals[variable] = ensemble_df[variable].min()
        stds[variable] = ensemble_df[variable].std()
        variance[variable] = ensemble_df[variable].var()
        diff[variable] = np.abs(ensemble_df[variable].max() - ensemble_df[variable].min())
        q1s[variable] = ensemble_df[variable].quantile(0.25)
        q3s[variable] = ensemble_df[variable].quantile(0.75)
    
    df = pd.DataFrame({
        'mean': mean_vals,
        'max': max_vals,
        'min': min_vals,
        'range': diff,
        'std': stds,
        'variance': variance,
        'q1': q1s,
        'q3': q3s,
    })
    df['variable'] = df.index
    df['CV'] = df['std']/df['mean']
    df['iqr'] = df['q3'] - df['q1']
    
    return df

def print_ensemble_range(df,  model, variable, units):
    
    var_df = df[df.variable == variable]
    mod_df = var_df[var_df.model == model]
    
    print(f'{model} {variable} ranges from', 
          round(mod_df['min'].values[0], 2), 'to', 
          round(mod_df['max'].values[0], 2), units)
    
    print('This is a range of ', round(mod_df['range'].values[0], 2), units)
    print('And a mean of ', round(mod_df['mean'].values[0], 2), units)
    print('And a standard devaiation of ', round(mod_df['std'].values[0], 2), units)
    print('And a variance of ', round(mod_df['variance'].values[0], 2), units)
    print('And an IQR of ', round(mod_df['iqr'].values[0], 2), units)

def get_both_ranges(active_df, vars):
    
    fates_ensemble = active_df[active_df.model == 'FATES']
    clm_ensemble = active_df[active_df.model == 'CLM']
    
    fates_df = get_ensemble_ranges(fates_ensemble, vars)
    fates_df['model'] = 'FATES'

    clm_df = get_ensemble_ranges(clm_ensemble, vars)
    clm_df['model'] = 'CLM'

    range_df = pd.concat([fates_df, clm_df])

    mean_df = np.abs(range_df.groupby('variable')['mean'].mean())
    
    range_df['range_norm'] = range_df.apply(lambda row: row['range'] / mean_df[row['variable']], axis=1)

    return range_df

def get_all_cumulative_variance(variables, clm_pars, clm_glob, fates_pars, fates_glob, 
                                n=5):
    variances = {}
    for variable in variables:
        variances[variable] = {}
    
        clm_var = get_param_variance(clm_pars, variable, clm_glob, 0)
        variances[variable]['CLM'] = get_cumulative_variance(clm_var, clm_pars, n)
    
        fates_var = get_param_variance(fates_pars, variable, fates_glob, 0)
        variances[variable]['FATES'] = get_cumulative_variance(fates_var, fates_pars, n)

    return variances

def get_number_required(df, model, variable, tol=0.9):
    arr = df[variable][model].values
    return np.argmax(arr >= tol)

def get_all_required(df, vars, tol=0.9):

    parameter_number_CLM = {}
    for variable in vars:
        parameter_number_CLM[variable] = get_number_required(df, 'CLM', variable, tol)
    
    clm_df = pd.DataFrame({
        'parameters': parameter_number_CLM,
    })
    clm_df['model'] = 'CLM'
    
    parameter_number_FATES = {}
    for variable in vars:
        parameter_number_FATES[variable] = get_number_required(df, 'FATES', variable, tol)
    
    fates_df = pd.DataFrame({
        'parameters': parameter_number_FATES,
    })
    fates_df['model'] = 'FATES'

    df = pd.concat([fates_df, clm_df])
    df['variable'] = df.index
    return df

def get_param_counts_in_top_n(top_params):

    rows = []
    for variable, params in top_params.items():
        for param in params:
            rows.append({'parameter': param, 'variable': variable, 'top10_count': 1})
    
    df = pd.DataFrame(rows)
    param_counts = Counter(df.parameter)
    
    df = pd.DataFrame.from_dict(param_counts, orient='index', columns=['count'])
    df['parameter'] = df.index
    return df.sort_values(by='count', ascending=False)

def get_param_count_summary(top10_params_by_variable, output_groups):
    
    rows = []
    for variable, params in top10_params_by_variable.items():
        for param in params:
            rows.append({'parameter': param, 'variable': variable, 'top10_count': 1})
    
    df = pd.DataFrame(rows)
    var_to_group = {var: group for group, vars_ in output_groups.items() for var in vars_}
    df['group'] = df['variable'].map(var_to_group)

    param_stats = defaultdict(lambda: {'total_count': 0, 'groups': set(), 'per_group': defaultdict(int)})
    
    for _, row in df.iterrows():
        param = row['parameter']
        group = row['group']
        count = row['top10_count']
    
        param_stats[param]['total_count'] += count
        param_stats[param]['groups'].add(group)
        param_stats[param]['per_group'][group] += count

    summary = []
    
    for param, info in param_stats.items():
        summary.append({
            'parameter': param,
            'n_groups': len(info['groups']),
            'total_top10_count': info['total_count'],
            'avg_per_group': info['total_count'] / len(info['groups']) if info['groups'] else 0
        })
    
    summary_df = pd.DataFrame(summary).sort_values(by=['n_groups', 'total_top10_count'], ascending=False)

    return summary_df

def get_all_vardiffs(variables, clm_ds, fatesclm_ds, fates_ds,
                     nonzero_params, n, reldiff=False):
    
    clm_diffs = get_vardiff(clm_ds, clm_ds.sel(ensemble=0),
                                variables, nonzero_params['clm'], n,
                                reldiff=reldiff)
    clm_diffs['model'] = 'CLM'
    clm_diffs['parameter'] = clm_diffs.index
    
    fatesclm_diffs = get_vardiff(fatesclm_ds, fatesclm_ds.sel(ensemble=0),
                                     variables, nonzero_params['fates_clm'], n,
                                     reldiff=reldiff)
    fatesclm_diffs['model'] = 'FATES'
    fatesclm_diffs['parameter'] = fatesclm_diffs.index

    fates_diffs = get_vardiff(fates_ds, fates_ds.sel(ensemble=0),
                                         variables, nonzero_params['fates'], n,
                                         reldiff=reldiff)
    fates_diffs['model'] = 'FATES'
    fates_diffs['parameter'] = fates_diffs.index

    clm_sub = clm_diffs[clm_diffs.index.isin(fatesclm_diffs.parameter)]
    fates_clm_diff = pd.concat([fatesclm_diffs, clm_sub])
    fates_diffs = pd.concat([fatesclm_diffs, fates_diffs])

    return clm_diffs, fates_diffs, fates_clm_diff

def get_parameter_data(parameter, fates_maps, fate_clm_maps, clm_maps, nonzero_params):
    if parameter in nonzero_params['fates_only']:
        dat_fates = fates_maps.where(fates_maps.parameter_name == parameter, drop=True)
        dat_clm = None
    elif parameter in nonzero_params['fates_clm']:
        dat_fates = fate_clm_maps.where(fate_clm_maps.parameter_name == parameter, drop=True)
        dat_clm = clm_maps.where(clm_maps.parameter_name == parameter, drop=True)
    else:
        dat_clm = clm_maps.where(clm_maps.parameter_name == parameter, drop=True)
        dat_fates = None

    return dat_fates, dat_clm

def get_compare_df(clm_reldiffs, fates_reldiffs, clm_parameters, fates_parameters):
    
    clm_sub = clm_reldiffs[clm_reldiffs.parameter.isin(clm_parameters)].reset_index().drop(columns=['index'])
    fates_sub = fates_reldiffs[fates_reldiffs.parameter.isin(fates_parameters)].reset_index().drop(columns=['index'])
    
    both_sub = pd.concat([clm_sub, fates_sub]).melt(id_vars=['model', 'parameter'])
    both_sub[['base_var', 'stat']] = both_sub['variable'].str.extract(r'(.*)_(mean|sd)')
    df_wide = both_sub.pivot_table(index=['model', 'base_var', 'parameter'], columns='stat', values='value').reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={'mean': 'mean_value', 'sd': 'sd_value', 'base_var': 'variable'})

    return df_wide

def get_compare_df_3(clm_reldiffs, fates_reldiffs, fates_reldiffs2, fates_reldiffs3, 
                     clm_parameters, fates_parameters):
    
    clm_sub = clm_reldiffs[clm_reldiffs.parameter.isin(clm_parameters)].reset_index().drop(columns=['index'])
    fates_sub = fates_reldiffs[fates_reldiffs.parameter.isin(fates_parameters)].reset_index().drop(columns=['index'])
    fates_sub2 = fates_reldiffs2[fates_reldiffs2.parameter.isin(fates_parameters)].reset_index().drop(columns=['index'])
    fates_sub3 = fates_reldiffs3[fates_reldiffs3.parameter.isin(fates_parameters)].reset_index().drop(columns=['index'])
    
    both_sub = pd.concat([clm_sub, fates_sub, fates_sub2, fates_sub3]).melt(id_vars=['model', 'parameter'])
    both_sub[['base_var', 'stat']] = both_sub['variable'].str.extract(r'(.*)_(mean|sd)')
    df_wide = both_sub.pivot_table(index=['model', 'base_var', 'parameter'], columns='stat', values='value').reset_index()
    df_wide.columns.name = None
    df_wide = df_wide.rename(columns={'mean': 'mean_value', 'sd': 'sd_value', 'base_var': 'variable'})

    return df_wide

def get_extra_simulations(hist_dir, fname_fates, fates_param_dat, variables, default_index,
                          special_vars, clm_glob):
    fates_glob_all, _, _ = get_area_means_diffs(os.path.join(hist_dir, fname_fates),
                                                      fates_param_dat, variables, default_index)
    
    fates_glob_subset = fates_glob_all.where(fates_glob_all.parameter_name.isin(np.append(special_vars, 'default')), drop=True)
    clm_subset = clm_glob.where(clm_glob.ensemble.isin([0, 364, 365, 366, 367, 378, 379, 380, 381, 386, 387, 388, 389]), drop=True)
    
    clm_diffs = get_vardiff(clm_subset, clm_subset.sel(ensemble=0),
                                variables, ['jmaxha', 'jmaxhd', 'jmaxse_sf', 'vcmaxha', 'vcmaxhd', 'vcmaxse_sf'], 20,
                                reldiff=True)
    clm_diffs['model'] = 'CLM'
    clm_diffs['parameter'] = clm_diffs.index
    
    fates_diffs = get_vardiff(fates_glob_subset, fates_glob_subset.sel(ensemble=289),
                                variables, special_vars, 20,
                                reldiff=True)
    fates_diffs['model'] = 'FATES'
    fates_diffs['parameter'] = fates_diffs.index

    return fates_diffs, clm_diffs
    
def get_pct_diff(active_df, variable, default_ds, tol=1.0):
    default_value = default_ds[variable].values
    all_len = len(active_df.parameter_name.unique())
    active_df['pct_diff'] = np.abs((active_df[variable] - default_value)/default_value*100)
    
    above_tol = active_df[active_df.pct_diff > tol]
    params = above_tol.parameter_name.unique()
    num_above = len(params)
    
    return num_above/all_len*100.0, num_above


def create_combined_mini_oaat_data(variable, fates_glob_combo_mean2, fates_meandiffs2,
                                  fates_glob_combo_mean3, fates_meandiffs3, fates_glob_combo_mean,
                                  fates_meandiffs_sub, clm_mean, clm_meandiffs_sub, 
                                  corresponding_params):

    fates_top10_2 = get_top_n(fates_glob_combo_mean2, fates_meandiffs2,
                                   variable, 10, fates_glob_combo_mean2.sel(ensemble=0))
    fates_top10_2['version'] = 'CLM-FATES parameter update'
    fates_top10_3 = get_top_n(fates_glob_combo_mean3, fates_meandiffs3, variable,
                                 10, fates_glob_combo_mean3.sel(ensemble=0))
    fates_top10_3['version'] = 'CLM-FATES parameter & water stress update'
    fates_top10_sub = get_top_n(fates_glob_combo_mean, fates_meandiffs_sub, variable,
                                 10, fates_glob_combo_mean.sel(ensemble=0))
    fates_top10_sub['version'] = 'CLM-FATES standard configuration'
    clm_top10_sub = get_top_n(clm_mean, clm_meandiffs_sub, variable,
                                 10, clm_mean.sel(ensemble=0))
    clm_top10_sub['version'] = 'CLM'
    
    all_top = pd.concat([fates_top10_2, fates_top10_3, fates_top10_sub, clm_top10_sub])
    all_top['analagous_parameter'] = all_top['parameter_name'].map(corresponding_params).fillna(all_top['parameter_name'])

    return all_top

def get_slope(df, varx, vary, model, category):
    
    df_model = df[df['model_name'] == model]
    df_cat = df_model[df_model.category_subset == category]
    x = df_cat[varx]
    y = df_cat[vary]
    slope, _, _, _, _ = stats.linregress(x, y)

    return slope