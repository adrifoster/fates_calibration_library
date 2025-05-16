import pytest

@pytest.fixture
def config_dict():
    return {
        "regrid_dir": "/tmp/test_output",
        "regrid_tag": "v1",
        "clobber": False
    }

@pytest.fixture
def ilamb_attributes():
    return {
        "model": "CLM5",
        "out_var": "gpp",
        "in_var": "gpp"
    }