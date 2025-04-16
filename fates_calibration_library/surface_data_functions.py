"""Functions to process CTSM surface datasets
"""
import os
import xarray as xr
import numpy as np
import pandas as pd

from fates_calibration_library.plotting_functions import plot_pft_percent

def get_surdat(surdat_file: str) -> xr.Dataset:
    """Reads a CLM surface dataset and calculates the percent PFT and CFT

    Args:
        surdat_file (str): full path to surface dataset

    Returns:
        xr.Dataset: Dataset with just PCT_NAT_PFT, PCT_CFT, and LANDFRAC_PFT
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

def get_pft_vars(pft_indices: list[int], cft_min: int=15) -> tuple[str, str]:
    """Return the variables and dimensions for a surface dataset depending on input pft indices

    Args:
        pft_indices (list[int]): list of pft indices
        cft_min (int, optional): minimum index for crop functional types. Defaults to 15.

    Raises:
        ValueError: A mix of crop and natural PFTs is not allowd

    Returns:
        tuple[str, str]: index variable name, variable name
    """
    if all(pft < cft_min for pft in pft_indices):
        # all are natural PFTs
        return 'natpft', 'PCT_NAT_PFT'
    elif all(pft >= cft_min for pft in pft_indices):
        # all are crop PFTs
        return 'cft', 'PCT_CFT'
    else:
        raise ValueError("Mixed PFT types in pft_indices: contains both natural and crop PFTs.")
    
def get_dom_pft_landmask(surdat: xr.Dataset, lai: xr.DataArray, pft_indices: list[int], 
                         config: dict=None, dom_dat: xr.Dataset=None) -> tuple[int, xr.Dataset]:
    """Subsets an input surface dataset based on pft indices and lai and dominance thresholds

    Args:
        surdat (xr.Dataset): _description_
        lai (xr.DataArray): _description_
        pft_indices (list[int]): _description_
        config (dict, optional): Dictionary containing optional filtering parameters. Defaults to None.
            - 'dominance_threshold' (float, optional): dominance threshold to select gridcells by. Defaults to 75.0.
            - 'lai_threshold' (str, optional): lai threshold to filter by. Defaults to 0.1.
            - 'cft_min' (float, optional): minimum PFT index for crop functional types. Defaults to 15.
            - 'overall_dom_thresh' (float, optional): dominance threshold to select co-dominant grids. Defaults to 75.0
        dom_dat (xr.Datset): dataset with percent cover data for "dominant pfts"
    Returns:
       tuple[int, xr.Dataset]: number of gridcells, output surface dataset masked to just gridcells for select pft_indices and thresholds
    """
    
    # assign default values if config is None
    if config is None:
        config = {}
    dominance_threshold = config.get("dominance_threshold", 75.0)
    lai_threshold = config.get("lai_threshold", 0.1)
    cft_min = config.get("cft_min", 15)
    overall_dom_thresh = config.get("overal_dom_thresh", 75.0)
    
    # depending on if crop or nat-pft, the variables have different names
    pftvar, pftpctvar = get_pft_vars(pft_indices, cft_min)
   
    # subset surface data by pft
    # we allow more than one so that we can combine pfts (for FATES - CLM compatibility)
    pft_dat = surdat.where(surdat[pftvar].isin(pft_indices))
    combined = pft_dat[pftpctvar].sum(dim=pftvar).to_dataset(name=pftpctvar)

    # add lai and subset out too little lai
    lai_interp = lai.interp(lat=combined['lat'], lon=combined['lon'])
    surdat_lai = xr.merge([combined, lai_interp])
    surdat_lai = surdat_lai.where(surdat_lai.lai > lai_threshold)
    
    # get rid of below a certain PFT percentage
    dom_pft = surdat_lai.where(surdat_lai[pftpctvar] > dominance_threshold)
    
    # filter based on 
    if dom_dat:
        dom_pft = xr.merge([dom_pft, dom_dat])
        dom_pft['pct_sum'] = dom_pft[pftpctvar] + dom_pft['dominant_pct']
        dom_pft = dom_pft.where(dom_pft.pct_sum > overall_dom_thresh)

    # get number of gridcells
    cells = dom_pft[pftpctvar].values
    num_gridcells = len(cells[~np.isnan(cells)])

    # create the landmask
    dom_pft['landmask'] = xr.where(~np.isnan(dom_pft[pftpctvar]), 1.0, 0.0)
    dom_pft['mod_lnd_props'] = dom_pft.landmask
    dom_pft = dom_pft.drop_vars(['lai'])

    return num_gridcells, dom_pft

def get_lats_lons(ds: xr.Dataset, pft_indices: list[int], var) -> tuple[list[float], list[float]]:
    """Returns the latitudes and longitudes associated with a pft grid

    Args:
        ds (xr.Dataset): input pft grid
        pft_indices (list[int]): list of pft indices associated with this grid
        var (float): variable to grab for dataset

    Returns:
        tuple[list[float], list[float]]: list of latitudes and longitudes
    """

    all_lats = ds.lat.values
    all_lons = ds.lon.values
    indices = np.argwhere(np.array(~np.isnan(ds[var])))
    
    lats = []
    lons = []
    for coord in indices:
        lats.append(all_lats[coord[0]])
        lons.append(all_lons[coord[1]])
    
    return lats, lons

def extract_dominant_grids(surdat: xr.Dataset, lai: xr.Dataset, mapping_dict: dict, 
                           min_grid_cells: int=20, config: dict=None) -> pd.DataFrame:
    """Creates a pandas dataframe of latitudes and longitudes associated with dominant pft grids
    for an input dictionary of PFTs

    Args:
        surdat (xr.Dataset): surface dataset
        lai (xr.DataArray): lai dataset for subsetting
        mapping_dict (dict): dictionary mapping PFT name to its index on a CLM surface dataset
        min_grid_cells(int, optional): minimum number of gridcells to include in "dominant" grid. Defaults to 20.
        config (dict, optional): Dictionary containing optional filtering parameters. Defaults to None.
            - 'dominance_threshold' (float, optional): dominance threshold to select gridcells by. Defaults to 75.0.
            - 'lai_threshold' (str, optional): lai threshold to filter by. Defaults to 0.1.
            - 'cft_min' (float, optional): minimum PFT index for crop functional types. Defaults to 15.
            - 'plot_figs' (bool, optional): whether or not to plot figures. Defaults to True.
    Returns:
       tuple[pd.DataFrame]: output dataframe of grids
    """
    
    # assign default values if config is None
    if config is None:
        config = {}
    
    # loop through pfts and extract dominant pft grid
    all_grids = []
    for pft, indices in mapping_dict.items():
        
        if len(indices) > 0:
            _, var = get_pft_vars(indices, config.get("cft_min", 15))
            num_gridcells, dompft = get_dom_pft_landmask(surdat, lai, indices, config)
            
        # if enough gridcells, pull out latitudes and longitudes
        if num_gridcells >= min_grid_cells:
            if config.get("plot_figs", True):
                plot_pft_percent(dompft, var, pft)
            lats, lons = get_lats_lons(dompft, indices, var)
            df = pd.DataFrame({'lats': lats, 'lons': lons})
            df['pft'] = pft
            all_grids.append(df)
            
    return pd.concat(all_grids)

def extract_biome(biome: xr.Dataset, lats: list[float], lons: list[float], pfts: list[str]) -> pd.DataFrame:
    """Extract Whittaker biomes for a list of observations

    Args:
        biome (xr.Dataset): Whittaker biome
        lats (list[float]): list of latitudes
        lons (list[float]): list of longitudes
        pfts (list[str]): list of pfts

    Returns:
        pd.DataFrame: output dataset
    """
    
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

def create_mask_ds(original_surdat: xr.Dataset, grid_df: pd.DataFrame, 
                   pft_names: list[str]) -> xr.Dataset:
    """Creates an xarray dataset that can be used to create a mesh file for running a "sparse" grid

    Args:
        original_surdat (xr.Dataset): original surface dataset
        grid_df (pd.DataFrame): dataframe with latitudes and longitudes
        pft_names (list[str]): names of pfts

    Returns:
        xr.Dataset: output dataset
    """
    
    surdat_lats = original_surdat.lat.values
    surdat_lons = original_surdat.lon.values
    points = [(p1, p2) for p1, p2 in zip(grid_df.lat.values, grid_df.lon.values)]
    pfts = grid_df.pft.values
    
    # grab points and merge together
    surdat_list = []
    for i, point in enumerate(points):
        actual_point = find_closest_point(point, surdat_lats, surdat_lons)
        sub_surdat = original_surdat.where(original_surdat['lat'] == actual_point[0]).where(original_surdat['lon'] == actual_point[1])
        sub_surdat["pft"] = xr.full_like(sub_surdat.LANDFRAC_PFT, pft_names.index(pfts[i])+1)
        sub_surdat = sub_surdat.where(original_surdat['lat'] == actual_point[0]).where(original_surdat['lon'] == actual_point[1])
        surdat_list.append(sub_surdat)
    surdat_out = xr.merge(surdat_list)

    cells = surdat_out['LANDFRAC_PFT'].values
    num_gridcells = len(cells[~np.isnan(cells)])

    surdat_out['landmask'] = xr.where(~np.isnan(surdat_out["LANDFRAC_PFT"]), 1.0, 0.0)
    surdat_out['mod_lnd_props'] = surdat_out.landmask
    surdat_out = surdat_out.drop_vars(['LANDFRAC_PFT', 'PCT_NAT_PFT', 'PCT_CFT', 'natpft', 'cft'])
    
    return num_gridcells, surdat_out

def find_closest_point(point: tuple[float, float], lats: list[float], 
                       lons: list[float]) -> tuple[float, float]:
    """Returns latitude and longitude in list closest to input point

    Args:
        point (tuple[float, float]): input point
        lats (list[float]): list of latitudes
        lons (list[float]): list of longitudes

    Returns:
        tuple[float, float]: output point
    """

    lat_out = lats[np.argmin(abs(point[0] - lats))]
    lon_out = lons[np.argmin(abs(point[1] - lons))]

    return (lat_out, lon_out)

def get_dom_surdat(surdat: xr.Dataset, dom_pfts: list[str], mapping_dict: dict) -> xr.Dataset:
    """Generates a grid with only dominant pfts

    Args:
        surdat (xr.Dataset): surface dataset
        dom_pfts (list[str]): list of dominant pft names
        mapping_dict (dict): dictionary to map FATES pft name to CLM PFT index

    Returns:
        xr.Dataset: output dataset
    """
    
    dom_ids = [mapping_dict[pft] for pft in dom_pfts]
    dom_list = [x for xs in dom_ids for x in xs]
    
    dom_dat = surdat.where(surdat['natpft'].isin(dom_list))
    dom_dat = dom_dat['PCT_NAT_PFT'].sum(dim='natpft').to_dataset(name='dominant_pct')
    
    return dom_dat

def extract_co_dominant_grids(surdat: xr.Dataset, lai: xr.DataArray, dom_pfts: list[str], 
                              mapping_dict: dict, dom_thresh: dict, min_grid_cells: int=10, 
                              config: dict=None):
    
    # assign default values if config is None
    if config is None:
        config = {}
    
    # first get the dominant pft grid
    dom_dat = get_dom_surdat(surdat, dom_pfts, mapping_dict)
    
    # get non-dominant PFTs
    nondom_pfts = [pft for pft, indices in mapping_dict.items()
               if pft not in dom_pfts and pft not in ['not_vegetated', 'broadleaf_evergreen_arctic_shrub'] and indices]
    
    # loop through pfts and extract c-dominant pft grid
    all_grids = []
    for pft in nondom_pfts:
        pft_indices = mapping_dict[pft]
        
        _, var = get_pft_vars(pft_indices, config.get("cft_min", 15))
        config['dominance_threshold'] = dom_thresh[pft]
        num_gridcells, dompft = get_dom_pft_landmask(surdat, lai, pft_indices, 
                                                     config=config, dom_dat=dom_dat)
        
        if num_gridcells >= min_grid_cells:
            if config.get("plot_figs", True):
                plot_pft_percent(dompft, var, pft)
            lats, lons = get_lats_lons(dompft, pft_indices, var)
            df = pd.DataFrame({'lats': lats, 'lons': lons})
            df['pft'] = pft
            all_grids.append(df)

    grids = pd.concat(all_grids)

    return grids


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

def write_landmask(ds: xr.Dataset, out_dir: str, tag: str):
    """Writes a landmask with the correct encoding to a directory

    Args:
        ds (xr.Dataset): landmask dataset
        out_dir (str): output directory
        tag (str): tag for naming
    """

    file_out = os.path.join(out_dir, f"{tag}_grid.nc")

    # need encoding for ncks to work
    encoding = {'lat': {'_FillValue': False},
                'landmask': {'_FillValue': False},
                'pft': {'_FillValue': False}}
    
    ds.to_netcdf(file_out, encoding=encoding)
    



