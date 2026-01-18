# FATES Calibration Library

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18285650.svg)](https://doi.org/10.5281/zenodo.18285650)

This repository contains the analysis code, notebooks, and local Python library used for calibrating FATES. 
It is designed to assist in generating parameter files, post-processing outputs, and 
emulation and calibration.

## Project Structure

| Folder/File | Description |
|---|---|
| `configs/` | Configuration files required for analysis. | 
| `data/` | Contains an observational data file used in the study. | 
| `fates_calibration_library/` | Local Python library containing core logic.| 
| `notebooks/` | Jupyter Notebooks
| `tests/` | Python test directory.| 
| `scripts/` | Scripts for creating jobs, submitting ensembles, and generating mesh files.| 
| `setup.py` | Build configuration to install `fates_calibration_library` as a local package. | 
| `pytest.ini` | Test configuration. | 

## Citation

If you use any of these scripts, code, or notebooks, please cite this repository:

Adrianna Foster. (2026). adrifoster/fates_calibration_library: Manuscript Submission - JAMES 2026 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18285651

## Contact

For questions regarding the code or data, please contact Adrianna Foster at afoster@ucar.edu.