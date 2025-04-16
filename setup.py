"""Setup"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fates_calibration_library",
    version="0.1.0",
    description="A collection of methods to assist in FATES calibration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adrianna Foster",
    author_email="afoster@ucar.edu",
    url="https://github.com/adrifoster/fates_calibration_library",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "bottleneck==1.3.7",
        "cartopy==0.22.0",
        "cftime==1.6.3",
        "geopandas==0.14.2",
        "matplotlib==3.8.2",
        "mpi4py",
        "nc-time-axis==1.4.1",
        "pandas==2.2.0",
        "scipy==1.12.0",
        "scikit-learn==1.4.0",
        "seaborn==0.13.2",
        "shapely==2.0.2",
        "gpflow==2.5.2",
        "xarray==2024.1.1",
        "xesmf==0.8.2",
        "xlrd",
        "salib==1.4.7",
        "numpy==1.24.3",
        "esem==1.1.0",
        "tensorflow==2.12.1",
        "tensorboard==2.12.3",
        "tensorflow-estimator==2.12.0",
        "tensorflow-probability==0.20.1",
        "tensorflow-io-gcs-filesystem==0.37.1",
    ],
    python_requires=">=3.11",
)
