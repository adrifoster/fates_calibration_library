import os
import glob
from datetime import date
import functools

import xarray as xr
  
def postprocess(top_dir, histdir, data_vars):

    files = sorted(glob.glob(os.path.join(top_dir, histdir, 'lnd', 'hist/') +
                             "*clm2.h0*.nc"))
    flen = len(files)

    if flen < 12:

        return None

    else:

        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time',
                               preprocess=functools.partial(preprocess,
                                                            data_vars=data_vars),
                               parallel=True, autoclose=True)

        # fix time bug
        # ds['time'] = xr.cftime_range(str(2005), periods=len(ds.time), freq='MS')
        # ds = ds.sel(time=slice("2055-01-01", "2064-12-31"))
        # ds['time'] = xr.cftime_range(str(2005), periods=12*10, freq='MS')
        
        ds['time'] = xr.cftime_range(str(2000), periods=len(ds.time), freq='MS')
        ds = ds.sel(time=slice("2060-01-01", "2074-12-31"))
        ds['time'] = xr.cftime_range(str(2000), periods=len(ds.time), freq='MS')
        # ds['time'] = xr.cftime_range(str(2005), periods=len(ds.time), freq='MS')
        # ds = ds.sel(time=slice("2055-01-01", "2150-12-31"))
        # ds['time'] = xr.cftime_range(str(2005), periods=len(ds.time), freq='MS')

        # calculate some variables

        ds['GPP'] = ds['FATES_GPP']*ds['FATES_FRACTION']  # kg m-2 s-1
        ds['GPP'].attrs['units'] = ds['FATES_GPP'].attrs['units']
        ds['GPP'].attrs['long_name'] = ds['FATES_GPP'].attrs['long_name']

        ds['LAI'] = ds['FATES_LAI']*ds['FATES_FRACTION']  # m m-2 
        ds['LAI'].attrs['units'] = ds['FATES_LAI'].attrs['units']
        ds['LAI'].attrs['long_name'] = ds['FATES_LAI'].attrs['long_name']

        sh = ds.FSH
        le = ds.EFLX_LH_TOT
        energy_threshold = 20
        
        sh = sh.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
        le = le.where((sh > 0) & (le > 0) & ((le + sh) > energy_threshold))
        ds['EF'] = le/(le + sh)
        ds['EF'].attrs['units'] = 'unitless'
        ds['EF'].attrs['long_name'] = 'Evaporative fraction'

        ds['ASA'] = ds.FSR/ds.FSDS.where(ds.FSDS > 0)
        ds['ASA'].attrs['units'] = 'unitless'
        ds['ASA'].attrs['long_name'] = 'All sky albedo'

        ds['RLNS'] = ds.FLDS - ds.FIRE
        ds['RLNS'].attrs['units'] = 'W m-2'
        ds['RLNS'].attrs['long_name'] = 'surface net longwave radiation'

        ds['RN'] = ds.FLDS - ds.FIRE + ds.FSDS - ds.FSR
        ds['RN'].attrs['units'] = 'W m-2'
        ds['RN'].attrs['long_name'] = 'surface net radiation'

        ds['Temp'] = ds.TSA - 273.15
        ds['Temp'].attrs['units'] = 'degrees C'
        ds['Temp'].attrs['long_name'] = ds['TSA'].attrs['long_name']

        ds0 = xr.open_dataset(files[0])
        extras = ['grid1d_lat', 'grid1d_lon']
        for extra in extras:
            ds[extra] = ds0[extra]

        ds['ensemble'] = histdir.split('_')[-1]

        ds.attrs['Date'] = str(date.today())
        ds.attrs['Author'] = 'afoster@ucar.edu'
        ds.attrs['Original'] = files[0]

        return ds

def postprocess_files(top_dir, postp_dir):
    
    data_vars = ['FATES_FRACTION', 'FATES_GPP', 'FATES_LAI', 'QVEGE', 'QVEGT',
         'EFLX_LH_TOT', 'FSH', 'QRUNOFF', 'SOILWATER_10CM','TBOT',
         'FSR', 'FSDS', 'FSA', 'FIRE', 'FLDS', 'RAIN', 'BTRANMN',
         'SNOW', 'H2OSNO', 'SNOWDP', 'TSA', 'landfrac', 'area']

    if not os.path.isdir(postp_dir):
        os.mkdir(postp_dir)
    
    dirs = sorted(os.listdir(top_dir))
    
    bad_ensembles = []
    good_ensembles = []
    for histdir in dirs:
        
        ensemble = histdir.split('_')[-1]
        out_file = os.path.join(postp_dir, f"{histdir}.nc")
        
        if not os.path.isfile(out_file):
            
            ds = postprocess(top_dir, histdir, data_vars)
            if ds is not None:
                ds.to_netcdf(out_file)
                good_ensembles.append(ensemble)
            else:
                bad_ensembles.append(ensemble)
    
    if len(bad_ensembles) > 0:
        with open('ensembles_good.txt', 'w') as f:
            for line in good_ensembles:
                f.write(f"{line}\n")
    else:
       print("All ensembles passed!")