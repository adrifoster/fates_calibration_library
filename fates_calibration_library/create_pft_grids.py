import os
import xarray as xr
import pandas as pd
from functools import reduce
import numpy as np

from FATES_CLM_surdat import get_surdat
from FATES_CLM_surdat import extract_dominant_grids, extract_biome, extract_co_dominant_grids
from FATES_calibration_constants import ILAMB_MODELS, FATES_INDEX
from fates_calibration_library.ILAMB_functions import extract_obs, filter_df

def get_actual_lat_lon(point, lats, lons):

    lat_out = lats[np.argmin(abs(point[0] - lats))]
    lon_out = lons[np.argmin(abs(point[1] - lons))]

    return (lat_out, lon_out)

def main(surdat_file, biome, lai, dom_threshold, lai_threshold, uncertainty_threshold,
         filter_vars, dom_pft_file=None, co_dom=False):
    
    surdat = get_surdat(surdat_file)
    
    if co_dom:
        if dom_pft_file is not None:
            dom_pfts = np.unique(pd.read_csv(dom_pft_file).pft.values)
            grid  = extract_co_dominant_grids(surdat, dom_threshold, lai, lai_threshold, dom_pfts)
        else:
            print("Must supply a dominant pft file")
            quit
    else:        
        grid = extract_dominant_grids(surdat, dom_threshold, lai, lai_threshold)
    
    lats = grid.lats.values
    lons = grid.lons.values
    pfts = grid.pft.values
    
    all_df = []
    for var, models in ILAMB_MODELS.items():
        all_df.append(extract_obs(ilamb_obs, var.lower(), models, lats, lons, pfts))
    all_df.append(extract_biome(biome, lats, lons, pfts))
    out_df = reduce(lambda x, y: pd.merge(x, y, on=['lat', 'lon', 'pft']), all_df)
    
    filtered_df = filter_df(out_df, filter_vars, uncertainty_threshold)
    
    # now make the grid
    lats = filtered_df.lat.values
    lons = filtered_df.lon.values
    pfts = filtered_df.pft.values
    surdat_lats = surdat.lat.values
    surdat_lons = surdat.lon.values
    points = [(p1, p2) for p1, p2 in zip(lats, lons)]
    
    # grab points and merge together
    surdat_list = []
    for i, point in enumerate(points):
        actual_point = get_actual_lat_lon(point, surdat_lats, surdat_lons)
        sub_surdat = surdat.where(surdat['lat'] == actual_point[0]).where(surdat['lon'] == actual_point[1])
        sub_surdat["pft"] = xr.full_like(sub_surdat.LANDFRAC_PFT, FATES_INDEX[pfts[i]])
        sub_surdat = sub_surdat.where(surdat['lat'] == actual_point[0]).where(surdat['lon'] == actual_point[1])
        surdat_list.append(sub_surdat)
    surdat_out = xr.merge(surdat_list)

    cells = surdat_out['LANDFRAC_PFT'].values
    num_gridcells = len(cells[~np.isnan(cells)])

    surdat_out['landmask'] = xr.where(~np.isnan(surdat_out["LANDFRAC_PFT"]), 1.0, 0.0)
    surdat_out['mod_lnd_props'] = surdat_out.landmask
    surdat_out = surdat_out.drop_vars(['LANDFRAC_PFT', 'PCT_NAT_PFT', 'PCT_CFT', 'natpft', 'cft'])

    print(f"Made a surface dataset with {num_gridcells} gridcells")

    return surdat_out, filtered_df


if __name__ == "__main__":
  
    # surface file
    surdat_dir = "/glade/campaign/cesm/cesmdata/inputdata/lnd/clm2/surfdata_esmf/ctsm5.3.0/"
    surdat_2deg = os.path.join(surdat_dir, "surfdata_1.9x2.5_hist_2000_16pfts_c240908.nc")
    
    # observations
    obs_dir = '/glade/u/home/afoster/FATES_Calibration/observations'
    
    # ilamb MODIS LAI
    ilamb_obs = xr.open_dataset(os.path.join(obs_dir, 'ILAMB_obs.nc'))
    modis_lai = ilamb_obs.sel(model='MODIS').lai
    
    # whittaker biome
    biome = xr.open_dataset(os.path.join(obs_dir, 'biome_2deg.nc'))
    
    # output directory
    out_dir = '/glade/work/afoster/FATES_calibration/mesh_files'
    
    # need encoding for ncks to work
    encoding = {'lat': {'_FillValue': False},
                'landmask': {'_FillValue': False},
                'pft': {'_FillValue': False}}
    
    # thresholds
    dom_threshold = 75.0
    lai_threshold = 0.1
    uncert_threshold = 6.0
    
    # variables to filter on
    filter_vars = ['le', 'sh', 'gpp', 'ef']
    
    # create the grid
    grid, grid_df = main(surdat_2deg, biome, modis_lai, dom_threshold, lai_threshold,
                         uncert_threshold, filter_vars)
    
    ## write out files
    dom_file_out = os.path.join(out_dir, 'dominant_pft_grid_fatesctsm6.nc')
    grid.to_netcdf(dom_file_out, encoding=encoding)
    dom_pft_file = os.path.join(out_dir, 'dominant_pft_grid_fatesctsm6.csv')
    grid_df.to_csv(dom_pft_file)
    
    # create the grid
    co_grid, co_grid_df = main(surdat_2deg, biome, modis_lai, dom_threshold, lai_threshold,
                         uncert_threshold, filter_vars, dom_pft_file=dom_pft_file, co_dom=True)
    
    ## write out files
    co_dom_file_out = os.path.join(out_dir, 'co_dominant_pft_grid_fatesctsm6.nc')
    co_grid.to_netcdf(co_dom_file_out, encoding=encoding)
    co_grid_df.to_csv(os.path.join(out_dir, 'co_dominant_pft_grid_fatesctsm6.csv'))