"""Functions to assist with parameter generation"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import qmc






def get_sample_values(default_value, i, param_dict, jags_dict, parameter, actual_param_name):
    
    pfts = [int(pft.replace('pft_', '')) for pft in list(param_dict.keys())]
    cols = list(jags_dict[parameter]['cols'].values())
    for k, par in enumerate(actual_param_name):
        for pft in pfts:
            value = param_dict[f"pft_{pft}"].iloc[i,][cols[k]]
            default_value[k][pft-1] = value
    return default_value

def create_lh_param_ensemble(
    params: list[str],
    num_samples: int,
    default_param_data: xr.Dataset,
    param_dat: dict,
    out_dir: str,
    param_prefix: str,
    lh_sample: np.ndarray[float]=None,
    keep_pfts: list[int]=None,
    jags_dict: dict=None,
    by_pft: bool=False,
):
    """Generates an ensemble of Latin Hypercube parameter files

    Args:
        params (list[str]): list of parameter names to include in ensemble
        num_samples (int): number of samples to create
        default_param_data (xr.Dataset): default parameter file
        param_dat (dict): dictionary of parameter information
        out_dir (str): output directory to write files to
        param_prefix (str): parameter file prefix
        lh_sample (np.ndarray[float], optional): Latin Hypercube sample. Defaults to None.
        keep_pfts (list[str], optional): list of pft to be left at default values. Defaults to None.
        by_pft (bool, optional). Whether or not to treat each pft-specific parameter as its own parameter. Default to False.
    """

    if keep_pfts is None:
        keep_pfts = []
    
    num_params = len(params)
    
    # get information about all parameter data
    main_param = param_dat["main"]
    
    if by_pft:
        pft_params, global_params = get_pft_and_global_params(params, main_param)
        params = np.append(pft_params, global_params)
        
        pftnames = default_param_data['fates_pftname'].values
        all_pfts = np.arange(1, len(pftnames)+1)
        pfts = [pft for pft in all_pfts if pft not in keep_pfts]
        num_pfts = len(pfts)
    
    if lh_sample is None:
        # create a latin hypercube
        if by_pft:
            total_dims = len(pft_params) * num_pfts + len(global_params)
            sampler = qmc.LatinHypercube(d=total_dims)
        else:
            sampler = qmc.LatinHypercube(d=num_params)
        lh_sample = sampler.random(n=num_samples)
    else:
        # check to make sure input latin hypercube sample matches other inputs
        if lh_sample.shape[0] != num_params:
            raise Exception(f"LH sample size {lh_sample.shape[0]} is not the same size as num_samples {num_samples}")
        elif lh_sample.shape[1] != num_params:
            raise Exception(f"LH parameter number {lh_sample.shape[1]} is not the same size as parameter number {num_params}")

    if jags_dict is not None:
        sample_dict = {}
        for par, info in jags_dict.items():
            param_dict = {}
            for pft, pft_dat in info['data'].items():
                param_dict[pft] = pft_dat.sample(n=num_samples)
            sample_dict[par] = param_dict
    else:
        sample_dict = None

    # loop through each row of latin hypercube and create a parameter file
    for i_sample, sample in enumerate(lh_sample):

        # new parameter file
        ds = default_param_data.copy(deep=False)
        
        if by_pft:
            param_values = {}
            pft_size = len(pft_params) * num_pfts
            pft_section = sample[:pft_size]
            global_section = sample[pft_size:]

            pft_reshaped = pft_section.reshape((len(pft_params), num_pfts))

            for i, param in enumerate(pft_params):
                param_values[param] = pft_reshaped[i, :]  # shape: (num_pfts,)

            for i, param in enumerate(global_params):
                param_values[param] = global_section[i]  # scalar
            
            sample = param_values
            
        # loop through each column (i.e. parameter)
        for j, value in enumerate(sample):
            if by_pft:
                value = sample[value]
                
            # get information about this parameter
            sub = main_param[main_param.fates_parameter_name == params[j]]

            # set parameter value for this parameter
            ds = set_lh_param_value(
                ds,
                params,
                sample,
                value,
                sub["param_type"].values[0],
                default_param_data,
                params[j],
                keep_pfts,
                param_dat,
                i_sample,
                jags_dict,
                sample_dict,
                by_pft=by_pft,
                pfts=None
            )

        # output to file
        ds.to_netcdf(os.path.join(out_dir, f"{param_prefix}_{generate_suffix(i_sample+1)}.nc"))
        ds.close()

    # write out the key and list of files
    lh_key = pd.DataFrame(lh_sample)
    if by_pft:
        columns = []
        for param in params:
            if param in pft_params:
                for pft in pfts:
                    columns.append(f"{param}_{pft}")
            else:
                columns.append(param)
    else:
        columns = params
        
    lh_key.columns = columns
    
    lh_key["ensemble"] = [
        f"{param_prefix}_{generate_suffix(ens)}"
        for ens in np.arange(1, num_samples + 1)
    ]
    lh_key.to_csv(os.path.join(out_dir, f"{param_prefix.lower()}_key.csv"))

    write_ensemble_list(param_prefix, lh_key.ensemble.values, out_dir)

def get_pft_info(pft, default_param_file, pft_ids):
    
    default_param = xr.open_dataset(default_param_file)
    all_pfts = [str(pft).replace("b'", "").replace("'", "").strip() for pft in default_param.fates_pftname.values]
    pft_name = all_pfts[pft-1]
    pft_id = pft_ids[pft_name]
    
    return pft_name, pft_id
