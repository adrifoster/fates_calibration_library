import os
import xarray as xr
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from esem import gp_model
import pickle
import gpflow

def split_dataset(var: xr.DataArray, params: pd.DataFrame, n_test: int):

    # target variable (excluding default [0])
    Y = var[1:].values

    # test and training parameters
    X_test, X_train = params.iloc[:n_test].copy(), params.iloc[n_test:].copy()

    # test and training output
    y_test, y_train = Y[:n_test], Y[n_test:]

    return X_test, X_train, y_test, y_train
  
def test_emulator(emulator, params, test, var):

    # predict test points with emulator
    pred, pred_var = emulator.predict(params)
    df = {f'{var}_test': test,
          f'{var}_pred': pred,
          f'{var}_var': pred_var}

    return pd.DataFrame(df)
  
def make_emulator(num_params, X_train, y_train):

  # create kernel
  kernel_linear = gpflow.kernels.Linear(active_dims=range(num_params),
                                        variance=1)
  kernel_matern32 = gpflow.kernels.Matern32(active_dims=range(num_params),
                                            variance=1,
                                            lengthscales=np.tile(1, num_params))
  kernel = kernel_linear + kernel_matern32

  # define emulator model and train
  emulator = gp_model(np.array(X_train), np.array(y_train), kernel=kernel)
  emulator.train()

  return emulator

def plot_emulator_validation(var_preds, var, units, pft):

    sqrt = np.sqrt(var_preds[f"{var}_var"])
    rms = root_mean_squared_error(var_preds[f"{var}_test"], var_preds[f"{var}_pred"])

    plt.figure()
    plt.scatter(var_preds[f"{var}_test"], var_preds[f"{var}_pred"])
    plt.plot([min(var_preds[f"{var}_test"]), max(var_preds[f"{var}_test"])],
             [min(var_preds[f"{var}_test"]), max(var_preds[f"{var}_test"])],
             c='k', linestyle='--', label='1:1 line')
    plt.errorbar(var_preds[f"{var}_test"], var_preds[f"{var}_pred"],
                 yerr=2*sqrt, fmt="o")
    plt.text(min(var_preds[f"{var}_test"]), max(var_preds[f"{var}_test"]), 
             'RMSE = {}'.format(np.round(rms, 3)))
    plt.xlabel(f'FATES {pft} mean annual mean {var} ({units})')
    plt.ylabel(f'Emulated {pft} mean annual mean {var} ({units})')
    plt.legend(loc='lower right')
    
def train_all_emulators(ds, vars, params, n_test, num_params, emulator_dir, fig_dir,
                        pft_id):

    validation_dfs = []
    X_train_dfs = []
    y_train_dfs = []
    for var in vars:
                
        # get average across gridcells
        average_var = weighted_mean(ds, var)
        
        # split ensemble dataset into testing and training
        X_test, X_train, y_test, y_train = split_dataset(average_var, params, n_test)
        
        # train emulator
        em = make_emulator(num_params, X_train, y_train)
        
        # write to file
        emulator_filename = os.path.join(emulator_dir, f"{pft_id}_{var}_emulator.pkl")
        with open(emulator_filename, "wb") as f:
            pickle.dump(em.model.model, f)

        # validate emulator 
        df_validation = test_emulator(em, X_test, y_test, var)
        df_validation['var'] = var
        validation_dfs.append(df_validation)
        
        # save the training data as well
        y_df = pd.DataFrame({'y_train': y_train})
        y_df['var'] = var
        X_train['var'] = var
        X_train_dfs.append(X_train)
        y_train_dfs.append(y_df)

        plot_emulator_validation(df_validation, var, VAR_UNITS[var], pft_id)
        plt.savefig(f'{fig_dir}/{pft_id}_{var}_emulator_validation.png',
                    bbox_inches='tight', dpi=300)

    em_validation = pd.concat(validation_dfs)
    X_trains = pd.concat(X_train_dfs)
    y_trains = pd.concat(y_train_dfs)
    
    em_validation.to_csv(f'{emulator_dir}/em_validation_{pft_id}.csv')
    X_trains.to_csv(f'{emulator_dir}/{pft_id}_X_train_data.csv')
    y_trains.to_csv(f'{emulator_dir}/{pft_id}_y_train_data.csv')

    return X_train, y_train

def train(pft, land_mask_file, mesh_file, ensemble_file, vars, lhckey,
         n_test, emulator_dir, fig_dir, ds0_file):
    
    lhkey_df = pd.read_csv(lhckey)
    params = lhkey_df.drop(columns=['ensemble'])
    num_params = len(params.columns)
    
    pft_grids = get_pft_grids(land_mask_file, mesh_file, FATES_INDEX[pft])
    ensemble = get_pft_ensemble(ensemble_file, pft_grids, ds0_file)
    
    train_all_emulators(ensemble, vars, params, n_test, num_params, emulator_dir,
                        fig_dir, FATES_PFT_IDS[pft])
    
    