"""Fixtures that can be used among tests"""
import pytest


@pytest.fixture
def config_dict() -> dict:
    """A config_dict fixture

    Returns:
        dict: a test config_dict
    """
    return {"regrid_dir": "/tmp/test_output", "regrid_tag": "v1", "clobber": False}


@pytest.fixture
def ilamb_attributes() -> dict:
    """An ilamb_attributes fixture

    Returns:
        dict: a test ilamb_attributes dictionary
    """
    return {"model": "CLM5", "out_var": "gpp", "in_var": "gpp"}
