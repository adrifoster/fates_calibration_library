import os
import pandas as pd
import argparse
import numpy as np
import xarray as xr
from mpi4py import MPI

import fates_calibration_library.calibration as cal
import fates_calibration_library.parameter_generation as param
import fates_calibration_library.emulator_functions as em
import fates_calibration_library.utils as utils


def commandline_args():
    """Parse and return command-line arguments"""

    description = """
    Typical usage: python calibrate_history_matching.py --pft 1

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default='/glade/work/afoster/FATES_calibration/emulator_configs/dom_pft.yaml',
        help='Config file with information about ensemble\n',
    )
    parser.add_argument(
        "--calib-vars",
        type=str,
        default='/glade/work/afoster/FATES_calibration/emulator_configs/calibration_vars.yaml',
        help='Config file with information about calibration variables to use for each PFT\n',
    )
    parser.add_argument(
        "--pft-ids",
        type=str,
        default='/glade/work/afoster/FATES_calibration/fates_calibration_library/configs/fates_pft_ids.yaml',
        help='Config file with information about PFT ids\n',
    )
    parser.add_argument(
        "--obs-config",
        type=str,
        default='/glade/work/afoster/FATES_calibration/fates_calibration_library/configs/ilamb_conversion.yaml',
        help='Config file with information about observational data\n',
    )
    parser.add_argument(
        "--param-update-file",
        type=str,
        default='/glade/work/afoster/FATES_calibration/emulator_configs/param_min_max.yaml',
        help='Config file with updated (normalized) parameter min/maxes\n',
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
        default=100,
        help="Number of times to run calibration\n",
    )
    parser.add_argument(
        "--sobol",
        type=float,
        default=0.1,
        help="Sobol index minimum\n",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Output directory for parameter files\n",
    )
    args = parser.parse_args()

    return args

def save_results(df: pd.DataFrame, out_dir: str, pft_name: str):
    """Save result from calibration to a file

    Args:
        df (pd.DataFrame): Pandas DataFrame with "final" parameter values
        out_dir (str): output directory to write file
        pft_name (str): PFT name for file naming
    """
    
    rank = MPI.COMM_WORLD.rank
    file_name = f"params_{pft_name}_rank{rank}.csv"
    
    df.to_csv(os.path.join(out_dir, file_name))
    

def main():
    
    # get arguments and config files
    args = commandline_args()
    
    ensemble_config = utils.get_config_file(args.config)
    calib_vars_config = utils.get_config_file(args.calib_vars)
    pft_ids = utils.get_config_file(args.pft_ids)
    obs_config = utils.get_config_file(args.obs_config)
        
    # Latin Hypercube information
    param_names, _ = em.load_lhc_metadata(ensemble_config)
    
    # pft information
    pft_name, pft_id = param.get_pft_info(args.pft, ensemble_config['default_param'], pft_ids)
    
    # emulators plus targets and SDs
    emulators, targets, sds, _ = cal.load_emulator_and_obs_data(
        ensemble_config, pft_name, pft_id, args.emulator_dir, calib_vars_config[pft_id],
        obs_config
    )    
    
    if args.pft == 14:
        is_ch4 = True
    else:
        is_ch4 = False
    
    # run through and history match n times
    results_list = []
    for i in range(args.bootstraps):
        result_row = cal.run_history_matching(emulators, calib_vars_config[pft_id],
                                                    targets, sds, param_names,
                                                    implaus_tol=1.0, is_ch4=is_ch4)
        if result_row is not None:
            result_row['run_id'] = i 
            results_list.append(result_row)
            
    final_results_df = pd.concat(results_list, axis=0, ignore_index=True)
    save_results(final_results_df, args.out_dir, pft_name)

if __name__ == "__main__":
    main()
