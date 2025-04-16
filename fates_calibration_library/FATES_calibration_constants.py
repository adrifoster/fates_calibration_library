"""Holds constants for FATES calibration workflow
"""

#mapping from FATES pft name to CLM PFT index
FATES_CLM_INDEX = {"not_vegetated": [0],
                "broadleaf_evergreen_tropical_tree": [4],
                "needleleaf_evergreen_extratrop_tree": [1, 2],
                "needleleaf_colddecid_extratrop_tree": [3],
                "broadleaf_evergreen_extratrop_tree": [5],
                "broadleaf_hydrodecid_tropical_tree": [6],
                "broadleaf_colddecid_extratrop_tree": [7, 8],
                "broadleaf_evergreen_extratrop_shrub": [9],
                "broadleaf_hydrodecid_extratrop_shrub": [],
                "broadleaf_colddecid_extratrop_shrub": [10, 11],
                "broadleaf_evergreen_arctic_shrub": [11],
                "broadleaf_colddecid_arctic_shrub": [11],
                "arctic_c3_grass": [12],
                "cool_c3_grass": [13],
                "c4_grass": [14],
                "c3_crop": [15],
                "c3_irrigated": [16]}

FATES_INDEX = {"broadleaf_evergreen_tropical_tree": 1,
                "needleleaf_evergreen_extratrop_tree": 2,
                "needleleaf_colddecid_extratrop_tree": 3,
                "broadleaf_evergreen_extratrop_tree": 4,
                "broadleaf_hydrodecid_tropical_tree": 5,
                "broadleaf_colddecid_extratrop_tree": 6,
                "broadleaf_evergreen_extratrop_shrub": 7,
                "broadleaf_hydrodecid_extratrop_shrub": 8,
                "broadleaf_colddecid_extratrop_shrub": 9,
                "broadleaf_evergreen_arctic_shrub": 10,
                "broadleaf_colddecid_arctic_shrub": 11,
                "arctic_c3_grass": 12,
                "cool_c3_grass": 13,
                "c4_grass": 14,
                "c3_crop": 15,
                "c3_irrigated": 16}

IMPLAUS_TOL = {
    'broadleaf_evergreen_tropical_tree': 1.0,
    'needleleaf_colddecid_extratrop_tree': 1.0,
    'needleleaf_evergreen_extratrop_tree': 1.0,
    'arctic_c3_grass': 3.0,
    'cool_c3_grass': 3.0,
    'c4_grass': 3.0,
    'broadleaf_evergreen_extratrop_tree': 1.0,
    'broadleaf_hydrodecid_tropical_tree': 1.0,
    'broadleaf_colddecid_extratrop_tree': 1.0,
    'broadleaf_colddecid_extratrop_shrub': 1.0,
    'c3_crop': 1.0,
    'c3_irrigated': 1.0
}

# FATES pft names and their IDs
FATES_PFT_IDS = {"broadleaf_evergreen_tropical_tree": "BETT",
            "needleleaf_evergreen_extratrop_tree": "NEET",
            "needleleaf_colddecid_extratrop_tree": "NCET",
            "broadleaf_evergreen_extratrop_tree": "BEET",
            "broadleaf_hydrodecid_tropical_tree": "BDTT",
            "broadleaf_colddecid_extratrop_tree": "BDET",
            "broadleaf_evergreen_extratrop_shrub": "BEES",
            "broadleaf_colddecid_extratrop_shrub": "BCES",
            "broadleaf_evergreen_arctic_shrub": "BEAS",
            "broadleaf_colddecid_arctic_shrub": "BCAS",
            "arctic_c3_grass": "AC3G",
            "cool_c3_grass": "C3G",
            "c4_grass": "C4G",
            "c3_crop": "C3C",
            "c3_irrigated": "C3CI"}
