import os
import pandas as pd
import argparse
import numpy as np
import xarray as xr
from mpi4py import MPI

import fates_calibration_library.emulator_functions as em
import fates_calibration_library.utils as utils

PFT_ID_CONFIG = '/glade/work/afoster/FATES_calibration/fates_calibration_library/configs/fates_pft_ids.yaml'
OBS_CONFIG_FILE = '/glade/work/afoster/FATES_calibration/fates_calibration_library/configs/ilamb_conversion.yaml'
CALIBRATION_VARS = ['GPP', 'EFLX_LH_TOT', 'FSH', 'EF']

def commandline_args():
    """Parse and return command-line arguments"""

    description = """
    Typical usage: python calibrate_scipy.py --pft 1

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default='/glade/work/afoster/FATES_calibration/emulator_configs/dom_pft.config',
        help='Config file with information about ensemble\n',
    )
    parser.add_argument(
        "--emulator-dir",
        type=str,
        default='/glade/work/afoster/FATES_calibration/emulators',
        help='Directory with emulators\n',
    )
    parser.add_argument(
        "--pft",
        type=int,
        default=1,
        help="PFT index to calibrate\n",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10,
        help="Number of times to run calibration\n",
    )
    parser.add_argument(
        "--sobol",
        type=float,
        default=0.01,
        help="Sobol index minimum\n",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default='/glade/work/afoster/FATES_calibration/parameter_outputs',
        help="Output directory for parameter files\n",
    )
    args = parser.parse_args()

    return args

def load_parameter_metadata(ensemble_config):
    
    # load Latin Hypercube key to get number of parameters
    lhc_key = pd.read_csv(ensemble_config['lhc_key_file'], index_col=[0])
    lhc_key = lhc_key.drop(columns=['ensemble'])
    param_names = lhc_key.columns
    num_params = len(param_names)
    
    # get normalized default parameter values
    default_norm = pd.read_csv(ensemble_config['default_norm'], index_col=[0])
    
    return param_names, num_params, default_norm
    
def get_pft_info(pft, default_param_file):
    
    pft_ids = utils.get_config_file(PFT_ID_CONFIG)
    default_param = xr.open_dataset(default_param_file)
    all_pfts = [str(pft).replace("b'", "").replace("'", "").strip() for pft in default_param.fates_pftname.values]
    pft_name = all_pfts[pft-1]
    pft_id = pft_ids[pft_name]
    
    return pft_name, pft_id

def load_emulator_and_obs_data(ensemble_config, pft_name, pft_id, emulator_dir):
    
    # load observations
    obs = pd.read_csv(ensemble_config['obs_df'], index_col=[0])
    obs_pft = obs[obs.pft == pft_name]
    
    # load parameter sensitivity
    sens_df = pd.read_csv(ensemble_config['sens_df'], index_col=[0])
    sens_pft = sens_df[sens_df.pft == pft_id]
    
    obs_config = utils.get_config_file(OBS_CONFIG_FILE)
    emulators, targets, sds = em.prep_calibration_data(obs_pft, CALIBRATION_VARS, obs_config, 
                                                    emulator_dir, pft_name, pft_id)
    
    return emulators, targets, sds, sens_pft

def build_optimization_config(ensemble_config):
    
    # build config
    config = {
        'maxiter': ensemble_config['maxiter'],
        'epsilon': 0.5,
        'lambda_penalty': None,
        'barrier_strength': 0,
        'loss_fn': em.implausibility_loss,
        'default_penalty_fn': em.default_penalty_l1,
        'barrier_penalty_fn': em.barrier_penalty,
        'tol': 1e-3
    }
    
    return config

def save_results(df, out_dir, pft_name):
    rank = MPI.COMM_WORLD.rank
    file_name = f"params_{pft_name}_rank{rank}.csv"
    df.to_csv(os.path.join(out_dir, file_name))
    

def main():
    
    # get arguments and config file
    args = commandline_args()
    ensemble_config = utils.get_config_file(args.config)
    
    # parameter information
    param_names, num_params, default_norm = load_parameter_metadata(ensemble_config)
    
    # pft information
    pft_name, pft_id = get_pft_info(args.pft, ensemble_config['default_param'])
    
    # emulators plus targets, SDS, and parameter sensitivity
    emulators, targets, sds, sens_pft = load_emulator_and_obs_data(
        ensemble_config, pft_name, pft_id, args.emulator_dir
    )    
    
    # default parameter values (normalized)
    params_default = em.get_default_pft_values(default_norm, args.pft)
    
    # get parameters that are going to be fixed vs. optimized
    fixed_indices, optimize_indices, num_optimize = em.get_params_to_optimize(sens_pft,
                                                                              param_names, 
                                                                              num_params, sobol_threshold=args.sobol)
    
    config = build_optimization_config(ensemble_config)
    all_results = em.run_batch_optimization(emulators, targets, sds, fixed_indices, params_default, num_optimize, 
                           param_names, optimize_indices, config, num_batch=args.bootstraps)
    
    save_results(all_results, args.out_dir, pft_name)


if __name__ == "__main__":
    main()
