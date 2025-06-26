import numpy as np
import pandas as pd
import os
from esem.wrappers import wrap_data
from esem.data_processors import Flatten
from esem.model_adaptor import GPFlowModel
from esem.emulator import Emulator
from esem.utils import get_random_params
import pickle
import matplotlib.pyplot as plt

from fates_calibration.FATES_calibration_constants import OBS_MODEL_VARS, VAR_UNITS

def load_emulator(pickle_file, y_train, X_train):
    
    with open(pickle_file, "rb") as file:
        gpmodel = pickle.load(file)

    # wrap the loaded model
    wrapped_gpmodel = GPFlowModel(gpmodel)
    data = wrap_data(y_train, data_processors=[Flatten()])
    emulator = Emulator(wrapped_gpmodel, X_train, data)

    return emulator

def implausibility_metric(pred, obs, pred_var, obs_var):

    top = np.abs(pred - obs)
    bottom = np.sqrt(pred_var + obs_var)

    imp = top/bottom

    return imp

def sensitivity_analysis(emulators, param_names, pft_id, out_dir, wave=None,
                         update_vars=None, default_pars=None, plot_figs=False):
    
    sensitivity_dfs = []
    oaat_dfs = []
    for var, emulator in emulators.items():
        
        problem, fast_sample = create_fast_sample(param_names, update_vars)
        
        if default_pars is not None:
            fast_sample = update_sample(fast_sample, default_pars, param_names)
        
        sens_df = fourier_sensitivity(emulator, problem, fast_sample, update_vars)
        sens_df['var'] = var
        sensitivity_dfs.append(sens_df)
    
        if plot_figs:
            plot_fourier_sensitivity(sens_df, f'{var} for {pft_id} Grids')
            plt.savefig(f'{out_dir}/{var}_emulator_fourier_sensitivity.png',
                        bbox_inches='tight', dpi=300)
    
            df = plot_oaat_sens(param_names, emulator)
            plt.savefig(f'{out_dir}/{var}_emulator_oaat_sensitivity.png',
                            bbox_inches='tight', dpi=300)
            df['var'] = var
            oaat_dfs.append(df)
            
            
    em_sensitivity = pd.concat(sensitivity_dfs)
    if plot_figs:
        oaat_sensitivity = pd.concat(oaat_dfs)
    else:
        oaat_sensitivity = None
    if wave is not None:
        em_sensitivity['wave'] = wave

    return em_sensitivity, oaat_sensitivity

def sample_emulators(emulators, param_names, n_samp, obs_df, out_dir, pft_id,
                     update_vars=None, default_pars=None, plot_figs=False):
    
    # get a random sample
    sample = get_random_params(len(param_names), n_samp)
    
    # update the sample if we are updating
    if update_vars is not None:
        sample = update_sample(sample, update_vars, param_names)
    
    if default_pars is not None:
        sample = update_sample(sample, default_pars, param_names)
        
    sample_df = pd.DataFrame(sample)
    sample_df.columns = param_names

    for var, emulator in emulators.items():
        
        # predict sample
        pred_sampled, pred_sampled_var = emulator.predict(sample)

        # observational mean and variance
        obs_mean = obs_df[f'{OBS_MODEL_VARS[var]}'].mean()
        obs_var = obs_df[f'{OBS_MODEL_VARS[var]}_var'].mean()

        if plot_figs:
            plot_emulated_sample(pred_sampled, obs_mean, obs_var, pft_id, var,
                                 VAR_UNITS[var])
            plt.savefig(f'{out_dir}/{var}_emulated_sample.png',
                        bbox_inches='tight', dpi=300)

        # calculate implausibility metric
        implaus = implausibility_metric(pred_sampled, obs_mean, pred_sampled_var,
                                        obs_var)
        sample_df[f'{var}_implausibility'] = implaus

    return sample_df

def load_all_emulators(pft_id, emulator_dir, vars):

    y_train = pd.read_csv(os.path.join(emulator_dir, f"{pft_id}_y_train_data.csv"))
    X_train = pd.read_csv(os.path.join(emulator_dir, f"{pft_id}_X_train_data.csv"))
    
    emulators = {}
    for var in vars:
        y_train_df = y_train[y_train['var'] == 'GPP']
        y_train_var = np.array(y_train_df.y_train)
        
        X_train_var = X_train[X_train['var'] == 'GPP']
    
        pickle_file = os.path.join(emulator_dir, f"{pft_id}_{var}_emulator.pkl")
        
        emulators[var] = load_emulator(pickle_file, y_train_var, X_train_var)
        
    return emulators
    
def calculate_implaus_sum(df, col_list):
    
    implausiblity_sum = df[col_list].sum(axis=1)
    
    return implausiblity_sum

def subset_sample(sample_dat, col_list, implaus_tol):
    
    for col in col_list:
        sample_dat = sample_dat.where(sample_dat[col] < implaus_tol)
    
    sample_sub = sample_dat.dropna()
    
    return sample_sub

def make_default_values(pars, min_max_pars, pft):
    d = {}
    for par in pars:
        d[par] = [get_rescaled_default_parval(min_max_pars, par, pft)]
    df = pd.DataFrame(d)
    return df

def plot_parameter_hists(sample, min_max_pars):
    
    pars = [par for par in sample.columns if par != 'wave']
    
    plt.figure(figsize=[18, 16])
    for i, par in enumerate(pars):
        
        par_min_max = min_max_pars[min_max_pars.parameter == par]
        minpar = float(par_min_max[par_min_max.type == 'min']['val'].values[0])
        maxpar = float(par_min_max[par_min_max.type == 'max']['val'].values[0])
        defaultpar = float(par_min_max['default'].values[0])
        p = (defaultpar - minpar)/(maxpar - minpar)
        
        ax = plt.subplot(7, 5, i + 1)
        ax.hist(sample[par])
        ax.axvline(x=p, color='r', linestyle='-')
        ax.set_xlabel(par)
        ax.set_xlim(0, 1)
    plt.tight_layout()
    
def find_sensitive_parameters(sens_df, vars, sens_tol):
    
    sub_dat = sens_df[sens_df['var'].isin(vars)]
    sub_dat = sub_dat.where(sub_dat.S1 > sens_tol)
    sub_dat = sub_dat.dropna()
    
    sensitive_pars = sorted(np.unique(sub_dat.names))

    return sensitive_pars

def get_param_samples(sample_dir):
    
    files = sorted([os.path.join(sample_dir, file) for file in os.listdir(sample_dir) if file.endswith(".csv")])
    sample_dfs = []
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        sample_dfs.append(df)
    sample_df = pd.concat(sample_dfs)
    sample_df = sample_df.drop(columns=['wave'])

    return sample_df

def find_best_parameter_sets(sample):
    best_sample_index = sample[['implaus_sum']].idxmin()
    best_sample = sample.loc[best_sample_index, :]
    return best_sample

def get_rescaled_default_parval(min_max_pars, par, pft):
    
    par_min_max = min_max_pars[min_max_pars.parameter == par]
    par_min_max_pft = par_min_max[par_min_max.pft == str(pft)]
    minpar = [float(s) for s in par_min_max_pft[par_min_max_pft.type == 'min']['val'].values[0].split('_')]
    maxpar = [float(s) for s in par_min_max_pft[par_min_max_pft.type == 'max']['val'].values[0].split('_')]
    defaultpar = [float(s) for s in par_min_max_pft['default'].values[0].split('_')]
    top = [default - minp for default, minp in zip(defaultpar, minpar)]
    bottom = [maxp - minp for maxp, minp in zip(maxpar, minpar)]
    p = [t/b for t, b in zip(top, bottom)]
    if len(p) > 1:
        p = np.mean(p)
    else:
        p = p[0]
    
    return p