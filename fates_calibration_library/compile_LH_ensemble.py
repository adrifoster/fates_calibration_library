import os
import numpy as np
import xarray as xr

def get_ensemble(files):
    # read in dataset and attach other info
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='ensemble',
                           parallel=True, chunks = {'time': 60, 'ensemble': 250, 'gridcell': 200})
    return ds
  
def annual_mean(da, cf):

    days_per_month = da['time.daysinmonth']
    ann_mean = cf*(days_per_month*da).groupby('time.year').sum()
    ann_mean.name = da.name

    return ann_mean
  
def calculate_vars(ds):

    grid1d_lat = ds.grid1d_lat
    grid1d_lon = ds.grid1d_lon
    
    # GPP in kg/m2/yr
    gpp_ens = annual_mean(ds.GPP, 24*60*60).mean(dim='year')

    # LAI [m2/m2]
    lai_ens = annual_mean(ds.LAI, 1/365).mean(dim='year')

    # ET in W/m2
    lh_ens = annual_mean(ds.EFLX_LH_TOT, 1/365).mean(dim='year')

    # SH in W/m2
    sh_ens = annual_mean(ds.FSH, 1/365).mean(dim='year')

    # evaporative fraction [0-1]
    ef_ens = annual_mean(ds.EF, 1/365).mean(dim='year')

    # surface soil moisture
    sw_ens = annual_mean(ds.SOILWATER_10CM, 1/365).mean(dim='year')

    # albedo [0-1]
    alb_ens = annual_mean(ds.ASA, 1/365).mean(dim='year')

    #surface upward sw radiation
    fsr_ens = annual_mean(ds.FSR, 1/365).mean(dim='year')

    # surface net sw radiation
    fsa_ens = annual_mean(ds.FSA, 1/365).mean(dim='year')

    # surface upward longwave radiation
    fire_ens = annual_mean(ds.FIRE, 1/365).mean(dim='year')

    # surface net longwave radiation
    rlns_ens = annual_mean(ds.RLNS, 1/365).mean(dim='year')

    # surface net radiation
    rn_ens = annual_mean(ds.RN, 1/365).mean(dim='year')

    #BTRAN
    btran_ens = annual_mean(ds.BTRANMN, 1/365).mean(dim='year')
    


    ensemble_ds = xr.merge([gpp_ens, lai_ens, lh_ens, sh_ens, ef_ens,
                            sw_ens, alb_ens, fsr_ens,
                           fsa_ens, fire_ens, rlns_ens, rn_ens, btran_ens, grid1d_lat,
                           grid1d_lon])

    return ensemble_ds
  
def compile_ensemble(topdir, out_file):
    
    files = sorted([os.path.join(topdir, file) for file in os.listdir(topdir)])
    ds = get_ensemble(files)
  
    annual_vals = calculate_vars(ds)
    annual_vals.to_netcdf(out_file)

    


