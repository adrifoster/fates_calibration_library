"""Tests for ilamb_functions"""

from fates_calibration_library.ilamb_functions import construct_output_filename


def test_construct_output_filename():
    """Tests basic usage of constuct_output_filename for a regrid and non-regridded file"""

    config = {
        "out_dir": "/tmp/test2",
        "regrid_dir": "/tmp/test1",
        "regrid_tag": "testgrid",
    }
    attrs = {"model": "CLM5", "out_var": "gpp"}

    expected_regrid = "/tmp/test1/CLM5_GPP_testgrid.nc"
    expected_noregrid = "/tmp/test2/CLM5_GPP.nc"

    assert construct_output_filename(config, attrs, True) == expected_regrid

    assert construct_output_filename(config, attrs, False) == expected_noregrid
