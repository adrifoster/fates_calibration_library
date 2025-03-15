import os
import xarray as xr
import numpy as np
import pandas as pd
from FATES_calibration_constants import FATES_CLM_INDEX

DOM_THRESH =   {"broadleaf_evergreen_tropical_tree": 75,
                "needleleaf_evergreen_extratrop_tree": 75,
                "needleleaf_colddecid_extratrop_tree": 75,
                "broadleaf_evergreen_extratrop_tree": 30,
                "broadleaf_hydrodecid_tropical_tree": 30,
                "broadleaf_colddecid_extratrop_tree": 30,
                "broadleaf_evergreen_extratrop_shrub": 30,
                "broadleaf_hydrodecid_extratrop_shrub": 60,
                "broadleaf_colddecid_extratrop_shrub": 60,
                "broadleaf_evergreen_arctic_shrub": 60,
                "broadleaf_colddecid_arctic_shrub": 60,
                "arctic_c3_grass": 75,
                "cool_c3_grass": 75,
                "c4_grass": 75,
                "c3_crop": 60,
                "c3_irrigated": 30}
CFT_MIN = 15

def get_surdat(surdat_file: str) -> xr.Dataset:
    """Reads a CLM surface dataset and calculates the percent PFT and CFT

    Args:
        surdat_file (str): full path to surface dataset

    Returns:
        xr.Dataset: Dataset with just PCT_NAT_PFT, PCT_CFT, and LANDFRAC
    """
    
    surdat = xr.open_dataset(surdat_file)

    # set actual latitude and longitude values as the dimension values
    surdat['lat'] = xr.DataArray(np.unique(surdat.LATIXY.values), dims='lsmlat')
    surdat['lon'] = xr.DataArray(np.unique(surdat.LONGXY.values), dims='lsmlon')
    surdat = surdat.swap_dims({'lsmlat': 'lat', 'lsmlon': 'lon'})

    surdat['PCT_NAT_PFT_UPDATED'] = (surdat.PCT_NAT_PFT*(surdat.PCT_NATVEG/100.0))/(surdat.PCT_NATVEG + surdat.PCT_CROP)*100.0
    surdat['PCT_CFT_UPDATED'] = (surdat.PCT_CFT*(surdat.PCT_CROP/100.0))/(surdat.PCT_NATVEG + surdat.PCT_CROP)*100.0
    
    surdat['PCT_NAT_PFT'] = surdat.PCT_NAT_PFT_UPDATED
    surdat['PCT_CFT'] = surdat.PCT_CFT_UPDATED

    surdat['PCT_NAT_PFT'] = surdat.PCT_NAT_PFT.where(surdat['natpft'] > 0, 0.0)
    
    sum_veg = surdat.PCT_NAT_PFT.sum(dim='natpft') + surdat.PCT_CFT.sum(dim='cft')
    surdat['PCT_NAT_PFT'] = surdat['PCT_NAT_PFT']/sum_veg*100.0
    surdat['PCT_CFT'] = surdat['PCT_CFT']/sum_veg*100.0

    out = surdat[['PCT_NAT_PFT', 'PCT_CFT', 'LANDFRAC_PFT']]

    return out

def get_dom_pft_landmask(surdat: xr.Dataset, pft_indices, dominance_threshold, lai, lai_threshold):
   
    surdat = surdat.where(surdat.LANDFRAC_PFT >= 0.9999)
    
    # subset surface data by pft
    # we allow more than one so that we can combine pfts (for FATES - CLM compatibility)
    if pft_indices[0] < CFT_MIN:
        pftvar = 'natpft'
        pftpctvar = 'PCT_NAT_PFT'
    else:
        pftvar = 'cft'
        pftpctvar = 'PCT_CFT'

    pft_dat = surdat.where(surdat[pftvar].isin(pft_indices))
    combined = pft_dat[pftpctvar].sum(dim=pftvar).to_dataset(name=pftpctvar)

    # add lai and subset out too little lai
    lai_interp = lai.interp(lat=combined['lat'], lon=combined['lon'])
    surdat_lai = xr.merge([combined, lai_interp])
    surdat_lai = surdat_lai.where(surdat_lai.lai > lai_threshold)
    
    # get rid of below a certain PFT percentage
    dom_pft = surdat_lai.where(surdat_lai[pftpctvar] > dominance_threshold)

    # get number of gridcells
    cells = dom_pft[pftpctvar].values
    num_gridcells = len(cells[~np.isnan(cells)])

    # create the landmask
    dom_pft['landmask'] = xr.where(~np.isnan(dom_pft[pftpctvar]), 1.0, 0.0)
    dom_pft['mod_lnd_props'] = dom_pft.landmask
    dom_pft = dom_pft.drop_vars(['mask', 'model'])

    return num_gridcells, dom_pft

def get_lats_lons(dom_pft, pft_indices):
    
    # we allow more than one so that we can combine pfts (for FATES - CLM compatibility)
    if pft_indices[0] < CFT_MIN:
        pftpctvar = 'PCT_NAT_PFT'
    else:
        pftpctvar = 'PCT_CFT'
    
    all_lats = dom_pft.lat.values
    all_lons = dom_pft.lon.values
    indices = np.argwhere(np.array(~np.isnan(dom_pft[pftpctvar])))
    
    lats = []
    lons = []
    for coord in indices:
        lats.append(all_lats[coord[0]])
        lons.append(all_lons[coord[1]])
    
    return lats, lons
    
def get_coords(mesh_file: str) -> tuple[np.ndarray, int]:
    """Gets the latitude and longitudes from a mesh file

    Args:
        mesh_file (str): full path to mesh file

    Returns:
        tuple[np.ndarray, int]: array of latitudes and longitudes as well as the 
        total number of gridcells
    """
    
    mesh = xr.open_dataset(mesh_file)
    
    # only get where we are actually simulating
    this_mesh = mesh.where(mesh.elementMask == 1, drop=True)
    
    # return center coordinates
    centerCoords = this_mesh.centerCoords.values
    
    # count up number of grids
    grids = this_mesh.elementCount

    return centerCoords, grids

def extract_dominant_grids(surdat, dom_threshold, lai, lai_threshold):
    all_grids = []
    for pft in FATES_CLM_INDEX.keys():
        if pft != "broadleaf_hydrodecid_extratrop_shrub":
        
            pft_indices = FATES_CLM_INDEX[pft]
            
            num_gridcells, dompft = get_dom_pft_landmask(surdat, pft_indices,
                                                         dom_threshold, lai, lai_threshold)
            if num_gridcells >= 20:
                lats, lons = get_lats_lons(dompft, pft_indices)
                df = pd.DataFrame({'lats': lats, 'lons': lons})
                df['pft'] = pft
                all_grids.append(df)
                
    grids = pd.concat(all_grids)

    return grids

def extract_co_dominant_grids(surdat, dom_threshold, lai, lai_threshold, dom_pfts):

    # first get the dominant pft grid
    dom_ids = [FATES_CLM_INDEX[pft] for pft in dom_pfts]
    dom_list = [x for xs in dom_ids for x in xs]
    dom_dat = surdat.where(surdat['natpft'].isin(dom_list))
    dom_dat = dom_dat['PCT_NAT_PFT'].sum(dim='natpft').to_dataset(name='dominant_pct')
    
    non_dom_pfts = [pft for pft in FATES_CLM_INDEX.keys()
                    if pft not in dom_pfts and pft != 'not_vegetated' and pft != "broadleaf_hydrodecid_extratrop_shrub"]
    all_grids = []
    for pft in non_dom_pfts:
        
        pft_indices = FATES_CLM_INDEX[pft]
        if pft_indices[0] < CFT_MIN:
            pftvar = 'natpft'
            pctvar = 'PCT_NAT_PFT'
        else:
            pftvar = 'cft'
            pctvar = 'PCT_CFT'
        
        pft_dat = surdat.where(surdat[pftvar].isin(pft_indices))
        pft_dat = pft_dat[pctvar].sum(dim=pftvar).to_dataset(name=pctvar)
        pft_dat = pft_dat.where(pft_dat[pctvar] > DOM_THRESH[pft])

        lai_interp = lai.interp(lat=pft_dat['lat'], lon=pft_dat['lon'])
        
        combined = xr.merge([pft_dat, dom_dat, lai_interp])
        
        combined['pct_sum'] = combined[pctvar] + combined['dominant_pct']
        
        combined = combined.where(combined.lai > lai_threshold)
        combined = combined.where(combined.pct_sum > dom_threshold)

        # get number of gridcells
        cells = combined[pctvar].values
        num_gridcells = len(cells[~np.isnan(cells)])

        if num_gridcells >= 10:
            lats, lons = get_lats_lons(combined, pft_indices)
            df = pd.DataFrame({'lats': lats, 'lons': lons})
            df['pft'] = pft
            all_grids.append(df)

    grids = pd.concat(all_grids)

    return grids

def write_landmask(ds, out_dir, tag):

    file_out = os.path.join(out_dir, f"{tag}_grid.nc")

    # need encoding for ncks to work
    encoding = {'lat': {'_FillValue': False},
                'lon': {'_FillValue': False},
                'landmask': {'_FillValue': False}}
    ds.to_netcdf(file_out, encoding=encoding)
    
def extract_biome(biome, lats, lons, pfts):
    
    # grab biome lat/lons
    biome_lats = biome['lat']
    biome_lons = biome['lon']

    # extract surdat and biome information at the chosen gridcells
    biome_extract = np.zeros(len(lats))
    for i in range(len(lats)):
            
            nearest_index_lat = np.abs(biome_lats - lats[i]).argmin()
            nearest_index_lon = np.abs(biome_lons - lons[i]).argmin()
            
            # grab data at correct lat/lon
            biome_extract[i] = biome['biome'][nearest_index_lat, nearest_index_lon]
    
    obs_df = pd.DataFrame({'lat': lats, 'lon': lons, 'pft': pfts,
                           'biome': biome_extract})
    
    return obs_df

def get_actual_lat_lon(point, lats, lons):

    lat_out = lats[np.argmin(abs(point[0] - lats))]
    lon_out = lons[np.argmin(abs(point[1] - lons))]

    return (lat_out, lon_out)


