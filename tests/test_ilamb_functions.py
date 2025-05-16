import os
from fates_calibration_library.ilamb_functions import (
    construct_output_filename
)

def test_construct_output_filename():
    
    config = {"regrid_dir": "/tmp/test", "regrid_tag": "testgrid"}
    attrs = {"model": "CLM5", "out_var": "gpp"}
    
    expected_regrid = "/tmp/test/CLM5_GPP_testgrid.nc"
    expected_noregrid = "/tmp/test/CLM5_GPP.nc"
    
    assert construct_output_filename(config, attrs, True) == expected_regrid
    
    assert construct_output_filename(config, attrs, False) == expected_noregrid
