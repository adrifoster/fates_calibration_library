"""
ParamEnsemble class - responsible for generating the entire ensemble
"""

from __future__ import annotations
from typing import Any, Optional
import copy
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import qmc
import xarray as xr

from .posterior import PosteriorConfig
from .parameter import Parameter
from .param_spec import DimIndex
from .scaler import DefaultScaler


class ParamEnsemble(ABC):
    """Abstract base class for the parameter ensemble class"""

    def __init__(
        self,
        param_data_file: Path,
        ensemble_dir: Path,
        file_prefix: str,
        param_list: Optional[list[str]] = None,
    ):
        
        ## TODO: re-order main correctly and then also enforce the order
        ## of the create_ensemble_member for scale_from_root
        main, pft_sheets = _read_param_list(param_data_file)

        # subset to only a list of parameters if supplied
        if param_list is not None:
            main = main[main.parameter_name.isin(param_list)].copy()

        self.ensemble_dir = Path(ensemble_dir)
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)
        self.file_prefix = file_prefix
        self.main = main
        self.pft_sheets = pft_sheets
        self.params = [
            Parameter.from_row(
                row, pft_sheet=self.pft_sheets.get(row["parameter_name"])
            )
            for _, row in self.main.iterrows()
        ]
        self.num_params = len(self.params)
        self.scaler = DefaultScaler()
        self._posterior_configs: dict[str, PosteriorConfig] = {}

    @abstractmethod
    def create_samples(self) -> list[dict[Parameter, Any]]:
        """Create samples from the list of parameters

        Returns:
            list[dict[Parameter, Any]]: list of dictionaries of Parameter and value to write
        """

    @abstractmethod
    def create_ensemble_member(
        self, sample: dict[Parameter, Any], default_ds: xr.Dataset
    ) -> xr.Dataset:
        """Create one member of the ensemble

        Args:
            sample (dict[Parameter, Any]): dictionary of Parameter and value to write
            default_ds (xr.Dataset): Default parameter dataset. Used as base of updated
            files.
        Returns:
            xr.Dataset: one member of the ensemble with updated values from default
        """
    
    @abstractmethod
    def create_ensemble_key(self, samples: list[dict[Parameter, Any]]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[dict[Parameter, Any]]): list of dictionaries of Parameter and value to write

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """

    def create_ensemble(self, default_ds: xr.Dataset):
        """Create and write out all ensemble parameter files

        Args:
            default_ds (xr.Dataset): default parameter dataset. Used as base for all
            ensemble parameter files.
        """
        samples = self.create_samples()
        for i, sample in enumerate(samples):
            ds = self.create_ensemble_member(sample, default_ds)
            file_name = f"{self.file_prefix}_{_generate_suffix(i)}.nc"
            ds.to_netcdf(self.ensemble_dir / file_name)
            ds.close()
        
        ensemble_key = self.create_ensemble_key(samples)
        ensemble_key.to_csv(self.ensemble_dir / f"{self.file_prefix}_key.csv")
        
        _write_ensemble_list(self.ensemble_dir, self.file_prefix,
                             list(ensemble_key.ensemble.values))

    def expand(
        self,
        default_ds: xr.Dataset,
        fixed_indices: Optional[dict[str, list[int]]] = None,
    ) -> list[Parameter]:
        """Expand list of Parameter objects into one Parameter per active index.

        Args:
            default_ds (xr.Dataset): Default parameter dataset. Used to determine the
                full set of valid indices for each dimension, and to validate
                active_indices.
            fixed_indices (Optional[dict[str, list[int]]], optional): Mapping of dimension
                name to 0-based indices to hold at default. These are never expanded into
                specs. If None, no indices are fixed and all are expanded over.

        Returns:
            list[Parameter]: Expanded Parameter list. Unexpanded Parameters are returned
            unchanged. Expanded Parameters are shallow copies with active_index set to a
            DimIndex.

        Raises:
            ValueError
                If fixed_indices references unknown dimensions or out-of-range indices.
            ValueError
                If a spec with expand_by_index=True has no free_dims.
        """
        fixed = fixed_indices or {}
        full_index_map = _build_full_index_map(default_ds)
        _validate_fixed(fixed, full_index_map)

        result = []
        for param in self.params:
            result.extend(self.expand_param(param, fixed, full_index_map))
        return result

    @staticmethod
    def expand_param(
        param: Parameter,
        fixed: dict[str, list[int]],
        full_index_map: dict[str, list[int]],
    ) -> list[Parameter]:
        """Return one expanded copy of Paremeter per active index of free_dims[0].

        Args:
            param (Parameter): parameter to expand
            fixed (dict[str, list[int]]): mapping of dim to indices of indices to fix
            full_index_map (dict[str, list[int]]): full available dim: indes mapping

        Raises:
            ValueError: no free dims to expand over
            ValueError: dimension not fouond in default_ds

        Returns:
            list[Parameter]: expanded copy of Parameters
        """
        if not param.spec.free_dims:
            raise ValueError(
                f"Parameter '{param.spec.name}' has no " "free_dims to expand over."
            )

        # expand over the first free dimension
        expand_dim = param.spec.free_dims[0]

        if expand_dim not in full_index_map:
            # dimension exists on the spec but not in default_ds — shouldn't
            # happen if the netCDF file and spreadsheet are consistent
            raise ValueError(
                f"Parameter '{param.spec.name}': free dimension '{expand_dim}' not "
                f"found in default_ds. Available dimensions: {sorted(full_index_map)}"
            )

        fixed_for_dim = fixed.get(expand_dim, [])
        active = [i for i in full_index_map[expand_dim] if i not in fixed_for_dim]

        expanded = []
        for idx in active:
            # here clone.spec points to the same ParamSpec as original,
            # which is intentional
            clone = copy.copy(param)
            clone.active_index = DimIndex(dim=expand_dim, index=idx)
            expanded.append(clone)

        return expanded
    
    def attach_posteriors(self, yaml_path: Path) -> None:
        """Load posterior configs from YAML

        Args:
            yaml_path (Path): Path to posterior_sources.yaml.

        Raises:
            ValueError: If a config entry has no matching Parameter
        """
        configs = PosteriorConfig.from_yaml(yaml_path)
        posterior_params = [p for p in self.params if p.spec.strategy == "posterior"]
        for param in posterior_params:
            if param.spec.name not in configs:
                raise ValueError(
                    f"parameter '{param.spec.name}' has strategy='posterior' but no "
                    "entry in posterior_sources.yaml."
                )
        self._posterior_configs = configs
        


class LatinHypercubeEnsemble(ParamEnsemble):
    """Concrete class for a Latin Hypercube ensemble"""

    def __init__(
        self,
        param_data_file: Path,
        ensemble_dir: Path,
        file_prefix: str,
        n_samples: int,
        param_list: Optional[list[str]] = None,
        prebuilt: Optional[np.ndarray] = None,
    ):
        super().__init__(
            param_data_file, ensemble_dir, file_prefix, param_list=param_list
        )
        self.n_samples = n_samples
        self.prebuilt = prebuilt

    def create_samples(self, default_ds) -> list[dict[Parameter, Any]]:
        """Create samples from the list of parameters

        Returns:
            list[dict[Parameter, Any]]: list of dictionaries of Parameter and value to
            write
        """

        # build latin hypercube
        latin_hypercube = self.build_lh(len(self.params), self.prebuilt)
                
        # set up and check posterior params
        posterior_params = [p for p in self.params if p.spec.strategy == "posterior"]
        if len(posterior_params) > 0:
            if not self._posterior_configs:
                raise RuntimeError(
                        f"Parameter ensemble does not have any posterior sources yet -"
                        "run ParamEnsemble.attach_configs('yaml_path')"
                    )
            for param in posterior_params:
                if param.spec.name not in self._posterior_configs:
                    raise RuntimeError(
                        f"parameter '{param.spec.name}' has strategy='posterior' but is not in"
                        "posterior_configs - check input yaml and re-run attach_configs"
                    )
            for name, config in self._posterior_configs.items():
                config.prepare(self.n_samples)
        
        
        # draw samples
        samples = []
        for i in range(self.n_samples):
            sample: dict[Parameter, Any] = {}

            for j, param in enumerate(self.params):
                lh_value = latin_hypercube[i, j]
                
                if param.spec.strategy == 'uniform':
                    sample[param] = float(lh_value)
                elif param.spec.strategy == 'posterior':
                    
                    ## FIX THIS
                    array_index = (
                         param.active_index.index if param.active_index is not None else None
                        )
                    free_dim = param.spec.free_dims[0] if param.spec.free_dims else None
                    n_indices = default_ds.sizes.get(free_dim, 1)
                    sample[param] = self._posterior_configs[param.spec.name].draw(lh_value, array_index, n_indices)

            samples.append(sample)

        return samples

    def create_ensemble_member(
        self, sample: dict[Parameter, Any], default_ds: xr.Dataset
    ) -> xr.Dataset:
        """Create one member of the ensemble

        Args:
            sample (dict[Parameter, Any]): dictionary of Parameter and value to write
            default_ds (xr.Dataset): Default parameter dataset. Used as base of updated
            files.
        Returns:
            xr.Dataset: one member of the ensemble with updated values from default
        """
        ds = default_ds.copy(deep=False)
        for param, sample_value in sample.items():
            default_val = param.get_default(default_ds)
            if param.spec.strategy == "uniform":
                value = self.scaler.scale(param.spec, sample_value, default_val)
            elif param.spec.strategy == "posterior":
                # passing on this for now
                continue
            param.set_value(ds, default_ds, value)

        return ds

    def create_ensemble_key(self, samples: list[dict[Parameter, Any]]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[dict[Parameter, Any]]): list of dictionaries of Parameter and value to write

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """
        param_dfs = []
        for i, sample in enumerate(samples):
            parameter_names = []
            sample_values = []
            for param, sample_value in sample.items():
                parameter_names.append(param.spec.name)
                sample_values.append(sample_value)
            df = pd.DataFrame({"parameter": parameter_names, "value": sample_values})
            df["ensemble"] = f"{self.file_prefix}_{_generate_suffix(i)}"
            param_dfs.append(df)
        param_df = pd.concat(param_dfs, ignore_index=True)
        return (
            param_df.pivot(index="ensemble", columns="parameter", values="value")
            .reset_index()
            .rename_axis(None, axis=1)
        )

    def build_lh(
        self, n_lh_dims: int, prebuilt: np.ndarray | None = None
    ) -> np.ndarray:
        """Create a Latin Hypercube, or validate a pre-built one

        Args:
            n_lh_dims (int): number of dimensions for the array (i.e. number of params)
            prebuilt (np.ndarray | None, optional): Optional pre-built hypercube.
            Defaults to None.

        Raises:
            ValueError: Supplied pre-built Latin Hypercube dimensions do not match setup

        Returns:
            np.ndarray: output Latin Hypercube array
        """
        if self.num_params == 0:
            return np.empty((self.n_samples, 0))

        # validate pre-built LH
        if prebuilt is not None:
            if prebuilt.shape != (self.n_samples, n_lh_dims):
                raise ValueError(
                    f"Pre-built LH sample has shape {prebuilt.shape}, "
                    f"expected ({self.n_samples}, {n_lh_dims})."
                )
            return prebuilt

        # otherwise generate one
        return qmc.LatinHypercube(d=n_lh_dims).random(n=self.n_samples)


def _read_param_list(
    param_data_file: Path,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Read in the excel file that sets up all the parameters

    Args:
        param_data_file (Path): path to excel file

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: main dataframe, dictionary of
        sheets with pft-specific parameter values
    """

    xl = pd.ExcelFile(param_data_file, engine="xlrd")
    main = pd.read_excel(xl, sheet_name="main")

    pft_sheets = {}
    for sheet in xl.sheet_names:
        if sheet != "main":
            pft_sheets[f"fates_{sheet}"] = pd.read_excel(xl, sheet_name=sheet)
    return main, pft_sheets


def _build_full_index_map(default_ds: xr.Dataset) -> dict[str, list[int]]:
    """Build a map of all dimension names to all valid 0-based indices.

    Args:
        default_ds (xr.Dataset): input default parameter dataset

    Returns:
        dict[str, list[int]]: output dictionary mapping
    """
    return {dim: list(range(default_ds.sizes[dim])) for dim in default_ds.dims}


def _validate_fixed(
    fixed: dict[str, list[int]],
    full_index_map: dict[str, list[int]],
):
    """Raise if fixed_indices references unknown dims or out-of-range indices.

    Args:
        fixed (dict[str, list[int]]): input mapping of dim to indices of fixed indices
        full_index_map (dict[str, list[int]]): full available mapping of dim to indices

    Raises:
        ValueError: fixed_indices has dimension which does not exist in default_ds
        ValueError: out of range index
    """
    for dim, idxs in fixed.items():
        if dim not in full_index_map:
            raise ValueError(
                f"fixed_indices contains dimension '{dim}' which does not "
                f"exist in default_ds. Available dimensions: {sorted(full_index_map)}"
            )
        valid = full_index_map[dim]
        invalid = [i for i in idxs if i not in valid]
        if invalid:
            raise ValueError(
                f"fixed_indices['{dim}'] contains out-of-range indices {invalid}. "
                f"Valid range for '{dim}' is 0–{len(valid) - 1}."
            )


def _generate_suffix(ensemble_number: int, pad_length: int = 3) -> str:
    """Generate a suffix for an ensemble member

    Args:
        ensemble_number (int): ensemble number
        pad_length (int, optional): pad length. Defaults to 3.

    Returns:
        str: output string
    """
    return str(ensemble_number).zfill(pad_length)


def _write_ensemble_list(out_dir: Path, file_prefix: str, ensembles: list[str]):
    """Writes out a list of ensemble members to supply to the run_ens script

    Args:
        out_dir (Path): output directory to write file to
        file_prefix (str): ensemble list file prefix
        ensembles (list[str]): list of ensembles
    """
    with open(out_dir / f"{file_prefix}.txt", "w", encoding="utf-8") as f:
        for ens in ensembles:
            f.write(f"{ens}\n")