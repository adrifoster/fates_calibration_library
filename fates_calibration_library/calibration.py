"""Calibration Methods"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from esem.utils import get_random_params
import fates_calibration_library.emulator_functions as em
from fates_calibration_library.TFClass import TFEmulator
import fates_calibration_library.plotting_functions as plotting

def get_obs_mean_and_sd(df, variable):
    weighted_var = df["land_area"] * df[variable]
    weighted_sd = df["land_area"] * np.sqrt(df[f"{variable}_var"])
    total_land = df.land_area.sum()

    weighted_var_mean = weighted_var.sum() / total_land
    weighted_sd_mean = weighted_sd.sum() / total_land

    return weighted_var_mean, weighted_sd_mean

def load_emulator_and_obs_data(ensemble_config, pft_name, pft_id, emulator_dir, calibration_vars,
                              obs_config):
    
    # load observations
    obs = pd.read_csv(ensemble_config['obs_df'], index_col=[0])
    obs_pft = obs[obs.pft == pft_name]
    obs_pft = obs_pft[obs_pft.land_frac > 0.99]
    if pft_id != 'AC3G':
        obs_pft = obs_pft[obs_pft.pct_lake < 30]
    
    # load parameter sensitivity
    sens_df = pd.read_csv(ensemble_config['sens_df'], index_col=[0])
    sens_pft = sens_df[sens_df.pft == pft_id]
    
    emulators, targets, sds = prep_calibration_data(obs_pft, calibration_vars, obs_config, 
                                                    emulator_dir, pft_name, pft_id)
    
    return emulators, targets, sds, sens_pft

def prep_calibration_data(
    obs, calibration_vars, obs_config, emulator_dir, pft_name, pft_id
):

    # get observations for this pft
    obs_pft = obs[obs.pft == pft_name]

    targets = []
    sds = []
    emulators = []
    for variable in calibration_vars:

        # observations for this pft and variable
        obs_mean, obs_sd = get_obs_mean_and_sd(obs_pft, obs_config[variable]["var"])

        # convert to tf objects
        targets.append(obs_mean)
        sds.append(obs_sd)

        # load the emulator
        emulators.append(TFEmulator(emulator_dir, pft=pft_id, variable=variable))

    return emulators, targets, sds

def get_corr(sample_df, method='spearman'):
    corr = sample_df.corr(method=method)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan
    return corr

def get_final_params(var_choose, dat):
    var_median = dat[var_choose].median()
    var_diff = np.abs(dat[var_choose] - var_median)
    choose_index = np.argmin(var_diff)
    return dat.iloc[choose_index]

def marginal_mode_kde(data_1d):
    kde = gaussian_kde(data_1d)
    # Define a fine grid over the parameter range
    grid = np.linspace(data_1d.min(), data_1d.max(), 1000)
    kde_values = kde(grid)
    mode_index = np.argmax(kde_values)
    return grid[mode_index]

def subset_sample(sample_dat, col_list, implaus_tol):

    for col in col_list:
        sample_dat = sample_dat.where(sample_dat[col] < implaus_tol)

    sample_sub = sample_dat.dropna()

    return sample_sub

def calculate_implaus_sum(df, col_list):

    implausiblity_sum = df[col_list].sum(axis=1)

    return implausiblity_sum


def implausibility_metric(pred, obs, pred_var, obs_var):

    top = np.abs(pred - obs)
    bottom = np.sqrt(pred_var + obs_var)
    return top / bottom


def get_proportion_implausible(implaus, tol=3.0):
    return len(implaus[implaus <= tol])/len(implaus)*100.0

def find_sensitive_parameters(sens_df, calibration_vars, sens_tol=0.01):

    sub_dat = sens_df[sens_df["variable"].isin(calibration_vars)]
    sub_dat = sub_dat.where(sub_dat.S1 > sens_tol)
    sub_dat = sub_dat.dropna()

    sensitive_pars = sorted(np.unique(sub_dat.parameter))

    return sensitive_pars


def find_best_sample(df, calibration_vars, implaus_tol):

    # subset to only below some implausibility tolerance
    sub_df = subset_sample(
        df, [f"{var}_implaus" for var in calibration_vars], implaus_tol
    )
    sub_df["implaus_sum"] = calculate_implaus_sum(
        sub_df, [f"{var}_implaus" for var in calibration_vars]
    )
    sub_df = sub_df[sub_df.implaus_sum <= implaus_tol]

    # return best
    best_sample_index = sub_df[["implaus_sum"]].idxmin()

    return sub_df, sub_df.loc[best_sample_index, :]

def find_best_parameter_sets(sample):
    best_sample_index = sample[["implaus_sum"]].idxmin()
    best_sample = sample.loc[best_sample_index, :]
    return best_sample


def get_params_to_optimize(sens_df, param_names, num_params, sobol_threshold=0.01):

    sobol_indices = np.array(
        [sens_df[sens_df.parameter == p]["ST"].sum() for p in param_names]
    )
    optimize_mask = sobol_indices > sobol_threshold
    fixed_indices = np.where(np.logical_not(optimize_mask))[0]
    optimize_indices = np.arange(num_params)[optimize_mask]
    num_optimize = num_params - len(fixed_indices)

    return fixed_indices, optimize_indices, num_optimize

def misfit(X, emulator_array, targets, stdevs, fixed_indices, X_default_all, config):

    # loop over each emulator/target/stddev
    total_error = 0.0
    for emulator, target, stddev in zip(emulator_array, targets, stdevs):

        X_full = em.get_full_array(X, fixed_indices, X_default_all)
        y_pred, y_var = emulator(X_full.reshape(1, -1))

        # calculate loss
        z = config["loss_fn"](y_pred, target, stddev, y_var)
        total_error += z

    return total_error

def squared_z_loss(y_pred, target, stdev, y_var=None):
    z = np.abs((y_pred - target) / (stdev + 1e-8))
    return z**2

def implausibility_loss(y_pred, target, stdev, y_var):
    total_variance = y_var + stdev**2 + 1e-8
    z = np.abs(y_pred - target) / np.sqrt(total_variance)
    return z

def default_penalty_l1(X, X_default):
    return np.sum(np.abs(X - X_default))

def barrier_penalty(X):
    return np.mean(1.0 / (X + 1e-6) + 1.0 / (1.0 - X + 1e-6))


def run_optimization(
    emulators,
    targets,
    sds,
    fixed_indices,
    optimize_pars,
    X_default_all,
    num_optimize,
    config,
    param_update_config,
):

    x0 = np.random.rand(num_optimize)
    bounds = [(0, 1)] * len(x0)

    for parameter, par_bounds in param_update_config.items():
        if parameter in optimize_pars:
            idx = np.argwhere(optimize_pars == parameter)[0][0]
            bounds[idx] = (par_bounds["min_val"], par_bounds["max_val"])

    result = minimize(
        misfit,
        x0,
        args=(emulators, targets, sds, fixed_indices, X_default_all, config),
        bounds=bounds,
        method="L-BFGS-B",
        options={"ftol": config["tol"], "maxiter": config["maxiter"]},
    )

    return result

def run_batch_optimization(
    emulators,
    targets,
    sds,
    fixed_indices,
    X_default_all,
    num_optimize,
    param_names,
    optimize_indices,
    config,
    param_update_config,
    num_batch=100,
):

    optimize_pars = param_names[optimize_indices]

    all_results = []
    for i in range(num_batch):
        result = run_optimization(
            emulators,
            targets,
            sds,
            fixed_indices,
            optimize_pars,
            X_default_all,
            num_optimize,
            config,
            param_update_config,
        )
        if result["success"]:
            df = pd.DataFrame(
                {"parameter": param_names[optimize_indices], "values": result["x"]}
            )
            df["batch"] = i
            all_results.append(df)
    all_out = pd.concat(all_results)
    return all_out.pivot(index="batch", columns="parameter", values="values")

def get_param_samples(sample_dir):

    files = sorted(
        [
            os.path.join(sample_dir, file)
            for file in os.listdir(sample_dir)
            if file.endswith(".csv")
        ]
    )
    sample_dfs = []
    for file in files:
        df = pd.read_csv(file, index_col=[0])
        sample_dfs.append(df)
    sample_df = pd.concat(sample_dfs)
    sample_df = sample_df.drop(columns=["wave"])

    return sample_df

def get_default_pft_values(default_norm, pft):
    default_pft = default_norm[default_norm.pft == pft]
    default_pft = default_pft.drop(columns=["pft"])
    return default_pft.to_numpy().flatten()

def get_calibrated_distribution(param_dir):
    param_files = [
        os.path.join(param_dir, f) for f in os.listdir(param_dir) if f.endswith(".csv")
    ]
    dat_list = []
    for file in param_files:
        dat_list.append(pd.read_csv(file, index_col=[0]))
    dat = pd.concat(dat_list)
    return dat


def emulate_and_analyze(
    emulators: list,
    calibration_vars: list[str],
    targets: list[float],
    sds: list[float],
    param_names: list[str],
    update_pars: pd.DataFrame | None = None,
    sample_size: float = 1e6,
    is_ch4: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    sample = get_random_params(len(param_names), int(sample_size))
    idx_displar = np.where(param_names == "fates_turb_displar")
    idx_z0mr = np.where(param_names == "fates_turb_z0mr")
    idx_vcmax = np.where(param_names == "fates_leaf_vcmax25top")
    sample[:, idx_displar] = sample[:, idx_displar] * (0.761 - 0.248) + 0.248
    sample[:, idx_z0mr] = sample[:, idx_z0mr] * (0.75 - 0.25) + 0.25

    if is_ch4:
        sample[:, idx_vcmax] = (
            sample[:, idx_vcmax] * (0.333333333 - 0.083333333) + 0.083333333
        )

    if update_pars is not None:
        sample = em.update_sample(sample, update_pars, param_names)
    df = pd.DataFrame(sample)
    df.columns = param_names

    sens_dats = []
    for i, emulator in enumerate(emulators):

        sens_df = em.sensitivity_analysis(emulator, param_names, update_pars=update_pars)
        sens_df["variable"] = calibration_vars[i]
        sens_dats.append(sens_df)

        y_pred, y_var = em.chunked_emulation(emulator, sample)
        implaus = implausibility_metric(y_pred, targets[i], y_var, sds[i] ** 2)
        df[f"{calibration_vars[i]}_implaus"] = implaus

    all_sens_df = pd.concat(sens_dats)

    return df, all_sens_df

def run_history_matching(
    emulators,
    calibration_vars,
    targets,
    sds,
    param_names,
    implaus_tol=1.0,
    sens_tol=0.1,
    min_implaus=0.1,
    is_ch4=False,
):

    run_batch = True
    update_pars = None

    iteration = 0
    while run_batch:
        iteration += 1
        # run emulator step
        out_df, sens_df = emulate_and_analyze(
            emulators,
            calibration_vars,
            targets,
            sds,
            param_names,
            update_pars=update_pars,
            is_ch4=is_ch4,
        )

        # identify sensitive parameters
        sensitive_pars = find_sensitive_parameters(
            sens_df, calibration_vars, sens_tol=sens_tol
        )

        # find the best sample (lowest implausibility)
        sub_df, best_sample = find_best_sample(out_df, calibration_vars, implaus_tol)

        # calculate the spread of the remaining samples
        if len(sub_df) > 0:
            implaus_range = sub_df.implaus_sum.max() - sub_df.implaus_sum.min()
        else:
            break

        # decision gate
        if (implaus_range > min_implaus) or (iteration == 1):

            if not sensitive_pars:
                run_batch = False
                continue

            # extract the specific values for these sensitive parameters
            param_best = best_sample[sensitive_pars]

            if isinstance(param_best, pd.Series):
                param_best = param_best.to_frame().T

            # update the tracking DataFrame
            if update_pars is None:
                update_pars = param_best
            else:
                update_pars = pd.concat(
                    [
                        update_pars.reset_index(drop=True),
                        param_best.reset_index(drop=True),
                    ],
                    axis=1,
                )

        else:
            run_batch = False

    return update_pars
    
def subset_calibrated_ensemble(df, param_names, X_default_all, calibration_vars, emulators,
                               targets, sds, pft_id, obs_config, imp_tol=1, plot_output=False):
    
    fixed_indices = np.where(~np.isin(param_names.values, df.columns.values))[0]
    
    sample = np.asarray(df)
    result = np.array([
        em.get_full_array(sample[i], fixed_indices, X_default_all)
        for i in range(sample.shape[0])
    ])

    for i, variable in enumerate(calibration_vars):
        y_pred, y_var = emulators[i](result)
        implaus = implausibility_metric(y_pred.numpy().flatten(), targets[i],
                                           y_var.numpy().flatten(), sds[i]**2)
        df[f"{variable}_implaus"] = implaus
        if plot_output:
            plotting.plot_implausibility_histogram(implaus, tol=imp_tol)
            plotting.plot_emulated_sample(y_pred.numpy().flatten(), targets[i], sds[i],
                                    pft_id, calibration_vars[i],
                                    obs_config[variable]['global_units'])

    col_list = [f"{f}_implaus" for f in calibration_vars]
    df_sub = subset_sample(df, col_list, imp_tol)
    df_sub['sum_implaus'] = calculate_implaus_sum(df_sub, col_list)

    df_out = df_sub.where(df_sub['sum_implaus'] < imp_tol)
    df_out = df_out.dropna()
    df_out = df_out.drop(columns=np.append('sum_implaus', col_list))

    return df_out