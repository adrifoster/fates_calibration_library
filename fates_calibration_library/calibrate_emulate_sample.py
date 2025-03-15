import os
import pandas as pd
import argparse
import numpy as np
from mpi4py import MPI

import fates_calibration.emulation_functions as emf
from fates_calibration.FATES_calibration_constants import FATES_PFT_IDS, FATES_INDEX, IMPLAUS_TOL

DEFAULT_PARS = {
    'broadleaf_evergreen_tropical_tree': [],
    'needleleaf_evergreen_extratrop_tree': [],
    'needleleaf_colddecid_extratrop_tree': [],
    'arctic_c3_grass': [],
    'cool_c3_grass': [],
    'c4_grass': [],
    'broadleaf_evergreen_extratrop_tree': [],
    'broadleaf_hydrodecid_tropical_tree': [],
    "broadleaf_evergreen_arctic_shrub": [],
    "broadleaf_colddecid_arctic_shrub": [],
    'broadleaf_colddecid_extratrop_tree': [],
    'broadleaf_colddecid_extratrop_shrub': [],
    'c3_crop': [],
    'c3_irrigated': []
}

def choose_params(sample_df, sens_df, vars, implausibility_tol, sens_tol):

    # subset out anything over implausibility tolerance
    implaus_vars = [f"{var}_implausibility" for var in vars]
    sample_df['implaus_sum'] = emf.calculate_implaus_sum(sample_df, implaus_vars)

    implaus_diff = np.max(sample_df.implaus_sum) - np.min(sample_df.implaus_sum)
    if implaus_diff <= 0.5:
       return None
    
    sample_sub = emf.subset_sample(sample_df, implaus_vars, implausibility_tol)
    

    # grab only the sensitive parameters
    sensitive_pars = emf.find_sensitive_parameters(sens_df, vars, sens_tol)

    if sample_sub.shape[0] > 0 and len(sensitive_pars) > 0:
        best_sample = emf.find_best_parameter_sets(sample_sub)
        sample_out = best_sample.loc[:, sensitive_pars]
    
        return sample_out.reset_index(drop=True)
    else:
        return None
    
def calibration_wave(emulators, param_names, n_samp, obs_df, pft_id, out_dir, wave,
                     implausibility_tol, sens_tol, update_vars=None, default_pars=None,
                     plot_figs=False):
    
    sens_df, oaat_df = emf.sensitivity_analysis(emulators, param_names, pft_id, out_dir, wave,
                                   update_vars=update_vars, default_pars=default_pars,
                                   plot_figs=plot_figs)
    
    sample_df = emf.sample_emulators(emulators, param_names, n_samp, obs_df, out_dir, pft_id,
                     update_vars=update_vars, default_pars=default_pars,
                     plot_figs=plot_figs)
    
    best_sample = choose_params(sample_df, sens_df, list(emulators.keys()),
                                implausibility_tol, sens_tol)

    return best_sample

def find_best_parameters(num_waves, emulators, param_names, n_samp, obs_df, pft_id, out_dir,
        implausibility_tol, sens_tol, default_pars=None):

    update_vars = None
    for wave in range(num_waves):
        if wave == 0:
            best_sample = calibration_wave(emulators, param_names, n_samp,
                                           obs_df, pft_id, out_dir, wave,
                                           implausibility_tol, sens_tol,
                                           update_vars=None, default_pars=default_pars)
        else:
            if best_sample is not None:
                if update_vars is None:
                    update_vars = best_sample
                else:
                    update_vars = pd.concat([update_vars, best_sample], axis=1)
                best_sample = calibration_wave(emulators, param_names, n_samp,
                                               obs_df, pft_id, out_dir, wave,
                                               implausibility_tol, sens_tol,
                                               update_vars=update_vars, 
                                               default_pars=default_pars)
            else:
                return update_vars, wave
    return update_vars, wave

def run_calibration(out_dir, pft, vars, emulator_dir, lhckey, obs_file, n_samp,
                    implausibility_tol, sens_tol, num_waves):
    
    pft_id = FATES_PFT_IDS[pft]
    
    lhkey_df = pd.read_csv(lhckey)
    lhkey_df = lhkey_df.drop(columns=['ensemble'])
    param_names = lhkey_df.columns
    
    obs_df_all = pd.read_csv(obs_file)
    obs_df = obs_df_all[obs_df_all.pft == pft]
    
    emulators = emf.load_all_emulators(pft_id, emulator_dir, vars)
    
    #min_max_pars = pd.read_csv('/glade/u/home/afoster/FATES_Calibration/FATES_LH_min_max_crops.csv')
    #default_pars = DEFAULT_PARS[pft]
    #default_parvals = emf.make_default_values(default_pars, min_max_pars, FATES_INDEX[pft])
    default_parvals = None

    best_param_set, wave = find_best_parameters(num_waves, emulators, param_names, n_samp,
                                          obs_df, pft_id, out_dir, implausibility_tol,
                                          sens_tol, default_pars=default_parvals)

    if best_param_set is not None:
        best_param_set['wave'] = wave
    
    return best_param_set

def commandline_args():
    """Parse and return command-line arguments"""

    description = """
    Typical usage: python calibrate_emulate_sample --pft BETT

    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--pft",
        type=str,
        default=0,
        help="PFT ID to calibrate\n",
    )
    parser.add_argument(
        "--nsamp",
        type=int,
        default=100000,
        help="Number of samples to emulate\n",
    )
    parser.add_argument(
        "--sens_tol",
        type=float,
        default=0.1,
        help="Sensitivity tolerance\n",
    )
    parser.add_argument(
        "--num_waves",
        type=int,
        default=4,
        help="Number of emulated waves\n",
    )
    parser.add_argument(
        "--lhkey",
        type=str,
        default='/glade/work/afoster/FATES_calibration/parameter_files/fates_param_lh/fates_lh_key.csv',
        help="path to Latin Hypercube parameter key\n",
    )
    parser.add_argument(
        "--obs_file",
        type=str,
        default='/glade/work/afoster/FATES_calibration/mesh_files/dominant_pft_grid_update.csv',
        help="path to observations data frame\n",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1,
        help="Number of times to run calibration\n",
    )
    parser.add_argument(
        "--pft_var_file",
        type=str,
        default='/glade/u/home/afoster/FATES_Calibration/pft_vars_dompft_gs1.csv',
        help="path to file with which variables to calibrate for each pft\n",
    )

    args = parser.parse_args()

    return args

def main():
    
    emulator_dir = '/glade/u/home/afoster/FATES_Calibration/pft_output_gs1/emulators'
    top_dir = "/glade/u/home/afoster/FATES_Calibration/pft_output_gs1"
        
    args = commandline_args()
    
    pft_id = args.pft
    pft = list(FATES_PFT_IDS.keys())[list(FATES_PFT_IDS.values()).index(pft_id)]

    implaus_tol = IMPLAUS_TOL[pft]
    
    out_dir = os.path.join(top_dir, f"{pft_id}_outputs")
    sample_dir = os.path.join(out_dir, 'samples')
    
    var_dat = pd.read_csv(args.pft_var_file)
    vars_pft = var_dat[var_dat.pft == pft].vars.values.tolist()
    vars = [var.replace('_implausibility', '') for var in vars_pft]

    best_sets = []
    for _ in range(args.bootstraps):
        best_param_set = run_calibration(out_dir, pft, vars, emulator_dir, args.lhkey,
                                        args.obs_file, args.nsamp, implaus_tol, 
                                        args.sens_tol, args.num_waves)
        if best_param_set is not None:
            best_sets.append(best_param_set)
    
    rank = MPI.COMM_WORLD.rank
    file_name = f"param_vals_{str(rank)}.csv"
    best_params = pd.concat(best_sets)
    best_params.to_csv(os.path.join(sample_dir, file_name))
        
if __name__ == "__main__":
    main()


    