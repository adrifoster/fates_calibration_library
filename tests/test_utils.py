"""Tests utils.py"""

from fates_calibration_library.utils import should_skip_file


def test_should_skip_file(tmp_path: str):
    """Tests should_skip_file function

    Args:
        tmp_path (str): input temporary path
    """
    file_path = tmp_path / "testfile.nc"
    file_path.write_text("dummy content")

    # test when file exists and clobber is False
    assert should_skip_file(str(file_path), clobber=False) is True

    # test when file exists and clobber is True
    assert should_skip_file(str(file_path), clobber=True) is False

    # test when file doesn't exist
    missing_path = tmp_path / "missing.nc"
    assert should_skip_file(str(missing_path), clobber=False) is False
