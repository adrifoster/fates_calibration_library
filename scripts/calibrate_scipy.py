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

def commandline_args():
    """Parse and return command-line arguments"""

    description = """
    Typical usage: python calibrate_scipy --pft 1

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
        default=1,
        help="Number of times to run calibration\n",
    )
    parser.add_argument(
        "--sobol",
        type=float,
        default=0.01,
        help="Sobol index minimum\n",
    )
    args = parser.parse_args()

    return args

def main():
    
    # pft IDS and information about history variables
    pft_ids = utils.get_config_file(PFT_ID_CONFIG)
    obs_config = utils.get_config_file(OBS_CONFIG_FILE)
    
    # calibration variables
    calibration_vars = ['GPP', 'EFLX_LH_TOT', 'FSH', 'EF']
    
    # get arguments and config file
    args = commandline_args()
    ensemble_config = utils.get_config_file(args.config)
    
    # load Latin Hypercube key to get number of parameters
    lhc_key = pd.read_csv(ensemble_config['lhc_key_file'], index_col=[0])
    lhc_key = lhc_key.drop(columns=['ensemble'])
    param_names = lhc_key.columns
    num_params = len(param_names)
        
    # load parameter sensitivity
    sens_df = pd.read_csv(os.path.join(args.emulator_dir, 
                                       f"sensitivity_df_{ensemble_config['ensemble_name']}.csv"), index_col=[0])
    
    # get pfts
    default_param = xr.open_dataset(ensemble_config['default_param'])
    all_pfts = [str(pft).replace("b'", "").replace("'", "").strip() for pft in default_param.fates_pftname.values]
    
    # get normalized default parameter values
    default_norm = pd.read_csv(ensemble_config['default_norm'], index_col=[0])
    
    pft_name = all_pfts[args.pft-1]
    pft_id = pft_ids[pft_name]
    
    # load observations
    obs = pd.read_csv(ensemble_config['obs_df'], index_col=[0])
    obs_pft = obs[obs.pft == pft_name]
    
    sens_pft = sens_df[sens_df.pft == pft_id]
    
    emulators, targets, sds = em.prep_calibration_data(obs_pft, calibration_vars, obs_config, 
                                                       args.emulator_dir, pft_name, pft_id)
    
    
    params_default = em.get_default_pft_values(default_norm, args.pft)
    
    fixed_indices, optimize_indices, num_optimize = em.get_params_to_optimize(sens_pft,
                                                                              param_names, 
                                                                              num_params, sobol_threshold=args.sobol)
    
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
    
    all_results = em.run_batch_optimization(emulators, targets, sds, fixed_indices, params_default, num_optimize, 
                           param_names, optimize_indices, config, num_batch=args.bootstraps)
    all_results.to_csv('test_out.csv')
    
        
if __name__ == "__main__":
    main()
