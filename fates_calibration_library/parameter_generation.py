"""Functions to assist with parameter generation"""

import os
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import qmc

# parameter information for parameter proxies
PARAM_INFO = {
    "smpsc_delta": {
        "root_param": "fates_nonhydro_smpso",
        "actual_param": "fates_nonhydro_smpsc",
        "default_value": np.array([189000]),
    }
}


def create_param_from_df(
    df: pd.DataFrame, default_param_data: xr.Dataset, skip_pfts: list[int] = None
) -> xr.Dataset:
    """Creates a new FATES parameter file from an input default parameter and a data frame
    with parameter values to update

    Args:
        df (pd.DataFrame): input data frame with parameter names and values to update
        default_param_data (xr.Dataset): default parameter set
        skip_pfts (list[int], optional): list of pft indices to skip updating. Defaults to [].

    Returns:
        xr.Dataset: new parameter file
    """

    if pfts is None:
        pfts = []

    # copy new parameter file
    new_param = default_param_data.copy(deep=False)

    # loop through all parameters and update
    params = sorted(np.unique(df.parameter.values))
    for param in params:
        sub = df[df.parameter == param]

        default_val = new_param[param].values
        new_val = default_val
        pfts = sorted([pft for pft in np.unique(sub.pft) if pft not in skip_pfts])
        for pft in pfts:
            sub_val = sub[sub.pft == pft].value.values[0]
            if param == "fates_leaf_vcmax25top":
                new_val[0, pft - 1] = round(sub_val, 3)
            else:
                new_val[pft - 1] = round(sub_val, 3)
        new_param[param].values = new_val

    return new_param


def make_param_list(param_data: xr.Dataset) -> pd.DataFrame:
    """Generates a pandas dataframe with parameter names, units, and coordinates

    Args:
        param_data (xr.Dataset): input parameter dataset

    Returns:
        pd.DataFrame: output parameter information
    """

    # loop through all data variables and grab dimensions and units
    df_list = []
    for parameter in param_data.data_vars:
        dims = list(param_data[parameter].dims)
        attrs = param_data[parameter].attrs
        df = pd.DataFrame(
            {
                "parameter_name": [parameter],
                "coord": [dims],
            }
        )
        for attr in attrs:
            df[attr] = attrs[attr]
        df_list.append(df)
    param_df = pd.concat(df_list)

    return param_df


def get_param_dictionary(param_list_file: str) -> dict:
    """Given an input excel file, returns a dictionary with information about all
    parameters ('main') as well as information about any individual parameters that should be
    indexed by pft

    Args:
        param_list_file (str): path to excel file

    Returns:
        dict: dictionary with information about parameters
    """

    # get sheet names of excel file
    param_list = pd.ExcelFile(param_list_file)
    sheet_names = param_list.sheet_names

    # loop through sheet and add to dictionary
    param_dat = {}
    for sheet in sheet_names:
        param_dat[sheet] = pd.read_excel(param_list_file, sheet_name=sheet)

    return param_dat


def generate_suffix(ensemble_number: int, pad_length: int = 3) -> str:
    """Generate a suffix for an ensemble member

    Args:
        ensemble_number (int): ensemble number
        pad_length (int, optional): pad length. Defaults to 3.

    Returns:
        str: output string
    """
    return str(ensemble_number).zfill(pad_length)


def get_oaat_ds(
    param_dat: dict, parameter: str, default_param_data: xr.Dataset, oaat_type: str
) -> xr.Dataset:
    """Creates a OAAT (min/max) parameter dataset

    Args:
        param_dat (dict): dictionary with information about parameter values
        param_list_file (str): list to dataset with information about parameters
        parameter (str): parameter name
        default_param_data (xr.Dataset): default parameter data
        oaat_type (str): param_min or param_max

    Returns:
        xr.Dataset: output parameter dataset
    """

    # get parameter information and default parameter values
    main_param = param_dat["main"]
    sub = main_param[main_param.fates_parameter_name == parameter]

    default_value = default_param_data[parameter].values

    # get parameter value
    param_value = get_param_value(
        str(sub[oaat_type].values[0]), default_value, param_dat, parameter, oaat_type
    )

    # create a copy and set new value
    ds = default_param_data.copy(deep=False)
    ds[parameter].values = param_value

    return ds


def write_ensemble_list(param_prefix: str, ensembles: list[str], out_dir: str):
    """Writes out a list of ensemble members to supply to the run_ens script

    Args:
        param_prefix (str): parameter file prefix
        ensembles (list[str]): list of ensembles
        out_dir (str): output directory to write file to
    """

    file_names = [f"{param_prefix}_{ens}" for ens in ensembles]
    with open(os.path.join(out_dir, f"{param_prefix}.txt"), "w", encoding="utf-8") as f:
        for line in file_names:
            f.write(f"{line}\n")


def create_oaat_param_ensemble(
    param_dat: dict,
    parameters: list[str],
    default_param_data: xr.Dataset,
    out_dir: str,
    param_prefix: str,
):
    """Generates an ensemble of OAAT (i.e. min/max) parameter files

    Args:
        param_dat (dict): dictionary with information about parameter values
        parameters (list[str]): list of parameters to use
        default_param_data (xr.Dataset): default FATES parameter file
        out_dir (str): directory to write ensemble to
        param_prefix (str): prefix for parameter files
    """

    # loop through all parameters in list and make a min and max parameter file
    oaat_key_list = []
    ens = 1
    for parameter in parameters:

        # min
        ds_min = get_oaat_ds(param_dat, parameter, default_param_data, "param_min")
        ds_min.to_netcdf(
            os.path.join(out_dir, f"{param_prefix}_{generate_suffix(ens)}.nc")
        )
        df_min = pd.DataFrame(
            {
                "ensemble": [generate_suffix(ens)],
                "type": ["min"],
                "parameter_name": [parameter],
            }
        )
        oaat_key_list.append(df_min)
        ens += 1

        # max
        ds_max = get_oaat_ds(param_dat, parameter, default_param_data, "param_max")
        ds_max.to_netcdf(
            os.path.join(out_dir, f"{param_prefix}_{generate_suffix(ens)}.nc")
        )
        df_max = pd.DataFrame(
            {
                "ensemble": [generate_suffix(ens)],
                "type": ["max"],
                "parameter_name": [parameter],
            }
        )
        oaat_key_list.append(df_max)
        ens += 1

    # write out the key and list of files
    oaat_key = pd.concat(oaat_key_list)
    oaat_key.to_csv(os.path.join(out_dir, f"{param_prefix.lower()}_key.csv"))

    write_ensemble_list(param_prefix, oaat_key.ensemble.values, out_dir)


def get_lh_values(
    lh_value: float, param_dat: dict, parameter: str, default_value: np.ndarray
) -> np.ndarray:
    """Gets values for a parameter based on an input latin hypercube value (0-1), appropriately
    scaled between min and max parameter values

    Args:
        lh_value (float): latin hypercube value (0-1)
        param_dat (dict): dictionary with information about the parameters
        parameter (str): parameter name
        default_value (np.ndarray): default parameter values

    Returns:
        np.ndarray: output parameter values
    """

    # get information about this parameter
    main_param = param_dat["main"]
    sub = main_param[main_param.fates_parameter_name == parameter]

    # get min and max parameter values
    change_str_min = str(sub["param_min"].values[0])
    change_str_max = str(sub["param_max"].values[0])

    min_value = get_param_value(
        change_str_min, default_value, param_dat, parameter, "param_min"
    )
    max_value = get_param_value(
        change_str_max, default_value, param_dat, parameter, "param_max"
    )

    # create updated value from lh_value
    param_value = unnormalize(lh_value, min_value, max_value)

    return param_value


def get_root_values(
    params: list[str],
    root_param: str,
    default_param_data: xr.Dataset,
    sample: np.ndarray,
    param_dat: dict,
) -> np.ndarray:
    """Gets the root values associated with a parameter that should scale off of another parameter

    Args:
        params (list[str]): list of all parameters to be calibrated
        root_param (str): root parameter name
        default_param_data (xr.Dataset): default parameter dataset
        sample (np.ndarray): latin hypercube sample
        param_dat (dict): dictionary with parameter information

    Returns:
        np.ndarray: output values of root parameter
    """

    if root_param in params:
        default_root = default_param_data[root_param].values
        root_loc = params.index(root_param)
        return get_lh_values(sample[root_loc], param_dat, root_param, default_root)

    return default_param_data[root_param].values


def set_default_pftvals(
    param_value: np.ndarray, default_value, keep_pfts: list[int], dims: list[str]
) -> np.ndarray:
    """For pfts that should be held fixed at default values, this loops through the default
    data and sets them based on input list

    Args:
        param_value (np.ndarray): updated parameter values
        default_value (_type_): default parameter values
        keep_pfts (list[int]): list of pft indices to set
        dims (list[str]): dimensions of parameter

    Returns:
        np.ndarray: output parameter values
    """

    for pft in keep_pfts:
        if len(dims) == 2:
            param_value[0, pft - 1] = default_value[0, pft - 1]
        else:
            param_value[pft - 1] = default_value[pft - 1]

    return param_value


def set_lh_param_value(
    ds: xr.Dataset,
    params: list[str],
    sample: np.ndarray,
    value: float,
    param_type: str,
    default_param_data: xr.Dataset,
    parameter: str,
    keep_pfts: list[int],
    param_dat: dict,
) -> xr.Dataset:
    """Sets a parameter value on an input dataset for a latin hypercube ensemble

    Args:
        ds (xr.Dataset): input parameter set
        params (list[str]): list of all parameters to calibrate
        sample (np.ndarray): latin hypercube sample
        value (float): latin hypercube value
        param_type (str): string to determine how to scale parameter
        default_param_data (xr.Dataset): default parameter dataset
        parameter (str): parameter name
        keep_pfts (list[int]): list of pft integers to fix at default values
        param_dat (dict): dictionary of information about parameters

    Returns:
        xr.Dataset: output parameter set
    """

    # get the actual parameter name (some could be proxies)
    if param_type == "default":
        actual_param_name = parameter
    else:
        actual_param_name = PARAM_INFO[parameter]["actual_param"]

    # get default parameter values and dimensions
    default_value = default_param_data[actual_param_name].values
    dims = list(default_param_data[actual_param_name].dims)

    if param_type == "default":

        # just get the values normally
        param_value = get_lh_values(value, param_dat, actual_param_name, default_value)

    elif param_type == "scale_from_rootparam":

        # need to scale off of a root value - get the values for the root parameter
        # for this LH sample
        root_value = get_root_values(
            params,
            PARAM_INFO[parameter]["root_param"],
            default_param_data,
            sample,
            param_dat,
        )

        # get the change value
        delta_value = get_lh_values(
            value, param_dat, parameter, PARAM_INFO[parameter]["default_value"]
        )

        # actual value is root + delta
        param_value = root_value + delta_value

    # set default pfts if they are supplied
    if len(keep_pfts) > 0 and "fates_pft" in dims:
        param_value = set_default_pftvals(param_value, default_value, keep_pfts, dims)

    # set the values
    ds[actual_param_name].values = param_value

    return ds


def create_lh_param_ensemble(
    params: list[str],
    num_samples: int,
    default_param_data: xr.Dataset,
    param_dat: dict,
    out_dir: str,
    param_prefix: str,
    keep_pfts: list[int] = None,
):
    """Generates an ensemble of Latin Hypercube parameter files

    Args:
        params (list[str]): list of parameter names to include in ensemble
        num_samples (int): number of samples to create
        default_param_data (xr.Dataset): default parameter file
        param_dat (dict): dictionary of parameter information
        out_dir (str): output directory to write files to
        param_prefix (str): parameter file prefix
        keep_pfts (list[str], optional): list of pft to be left at default values. Defaults to [].
    """

    # create a latin hypercube
    num_params = len(params)
    sampler = qmc.LatinHypercube(d=num_params)
    lh_sample = sampler.random(n=num_samples)

    # get information about all parameter data
    main_param = param_dat["main"]

    # loop through each row of latin hypercube and create a parameter file
    for i, sample in enumerate(lh_sample):

        # new parameter file
        ds = default_param_data.copy(deep=False)

        # loop through each column (i.e. parameter)
        for j, value in enumerate(sample):

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
            )

        # output to file
        ds.to_netcdf(os.path.join(out_dir, f"{param_prefix}_{generate_suffix(i+1)}.nc"))

    # write out the key and list of files
    lh_key = pd.DataFrame(lh_sample)
    lh_key.columns = params
    lh_key["ensemble"] = [
        f"{param_prefix}_{generate_suffix(ens)}"
        for ens in np.arange(1, num_samples + 1)
    ]
    lh_key.to_csv(os.path.join(out_dir, f"{param_prefix.lower()}_key.csv"))

    write_ensemble_list(param_prefix, lh_key.ensemble.values, out_dir)


def get_percentage_change(param_change: str) -> float:
    """Gets a percentage change (float) from an input string, i.e. '50percent'

    Args:
        param_change (str): input string

    Returns:
        float: percentage change [%]
    """
    return float(param_change.replace("percent", ""))


def get_param_pft_change(param_dat: dict, parameter: str, oaat_type: str) -> np.ndarray:
    """Gets the min/max for parameters that should change based on pft

    Args:
        param_dat (dict): dictionary with information about the parameters
                          this houses pft-specific values
        parameter (str): parameter name
        oaat_type (str): param_min or param_max

    Returns:
        np.ndarray: new parameter values
    """
    pft_values = param_dat[parameter.replace("fates_", "")]
    return pft_values[oaat_type].values


def get_param_percent_change(
    change_str: str, default_value: np.ndarray, oaat_type: str
) -> np.ndarray:
    """Gets the min/max for parameters that should change based on a percentage string

    Args:
        change_str (str): change string
        default_value (np.ndarray): default parameter values
        oaat_type (str): param_min or param_max

    Returns:
        np.ndarray: new parameter values
    """
    percent_change = get_percentage_change(change_str)
    if oaat_type == "param_min":
        return default_value - np.abs(default_value * (percent_change / 100.0))
    if oaat_type == "param_max":
        return default_value + np.abs(default_value * (percent_change / 100.0))

    print("Need to supply param_min or param_max")
    return None


def get_param_numeric_change(change_str: str, default_value: np.ndarray) -> np.ndarray:
    """Gets the min/max for parameters that should change based on an input number

    Args:
        change_str (str): change string
        default_value (np.ndarray): default parameter values

    Returns:
        np.ndarray: new parameter values
    """
    numeric_value = float(change_str)
    if default_value.size == 1:
        return numeric_value

    return np.full_like(default_value, numeric_value)


def get_param_value(
    change_str: str,
    default_value: np.ndarray,
    param_dat: dict,
    parameter: str,
    oaat_type: str,
) -> np.ndarray:
    """Gets new parameter values based on input string

    Args:
        change_str (str): input string
        default_value (np.ndarray): default parameter values
        param_dat (dict): information about parameters
        parameter (str): parameter name
        oaat_type (str): param_min or param_max

    Returns:
        np.ndarray: new parameter values
    """

    # get new parameter values
    if change_str == "pft":
        param_value = get_param_pft_change(param_dat, parameter, oaat_type)
    elif "percent" in change_str:
        param_value = get_param_percent_change(change_str, default_value, oaat_type)
    else:
        param_value = get_param_numeric_change(change_str, default_value)

    # reset values that should be -999.9
    if default_value.size == 1:
        if default_value == -999.9:
            param_value = -999.9
    else:
        param_value[default_value == -999.9] = -999.9

    return param_value


def normalize(value, min_value, max_value):
    """Normalizes value to be from 0 to 1

    Args:
        value (float): value to normalize
        min_value (float): mininum
        max_value (float): maximum

    Returns:
        float: normalized value
    """
    return (value - min_value) / (max_value - min_value)


def unnormalize(value, min_value, max_value):
    """Rescales a value (0-1) to be between the min and max

    Args:
        value (float): value to rescale
        min_value (float): minimum
        max_value (float): maximum

    Returns:
        float: rescaled value
    """
    return (max_value - min_value) * value + min_value




