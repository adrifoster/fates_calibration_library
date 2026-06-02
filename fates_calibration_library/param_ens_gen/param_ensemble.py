"""
ParamEnsemble class - responsible for generating the entire ensemble
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import yaml
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy.stats import qmc
import xarray as xr

from .parameter import Parameter
from .ensemble_config import EnsembleConfig, LatinHypercubeConfig, OneAtATimeConfig
from .sort_params import sort_params
from .utils import read_param_list

@dataclass
class ParameterSample:
    """A single parameter paired with its normalized sample value.
    
    Attributes
    ----------
    parameter: Parameter
        The parameter being samples.
    normalized_value: float
        A value in [0, 1]. For uniform parameters this is passed to a scaler to produce
        an actual value. For posterior parameters this is used as a quantile index into
        the posterior distribution.
    """
    parameter: Parameter
    normalized_value: float
    
@dataclass
class EnsembleMemberSample:
    """All parameter samples for one ensemble member
    
    Attributes
    ----------
    parameter_samples : list[ParameterSample]
        One ParameterSample per parameter being varied in this ensemble member.
    """
    parameter_samples: list[ParameterSample]
    
    def __iter__(self):
        """Iterate over the ParameterSamples in this member"""
        return iter(self.parameter_samples)
    
    def __len__(self):
        """Return the number of ParameterSamples in this member"""
        return len(self.parameter_samples)

class ParamEnsemble(ABC):
    """Abstract base class for the parameter ensemble class

    Attributes
    ----------
    ensemble_dir : Path
        Path to where parameter files will be written out
    file_prefix: str
        Prefix for output filenames, e.g. 'my_ensemble' produces 'my_ensemble_0001.nc', etc.
    default_ds: xr.DataFrame
        Default parameter dataset. Used as the base for all ensemble
        members and for parameter validation at construction time.
    params: list[Parameter]
        list of Parameter objects to sample
    num_params: int
        number of parameters
    fixed_indices: dict[str, list[int]] | None
        Mapping of dimension name to 0-based indices to hold at
        their default values across all ensemble members. For example,
        ``{'fates_pft': [7, 8, 9]}`` fixes PFTs 8, 9, and 10 (0-based).
        If None, all indices are free.
    """

    _registry: dict[str, type[ParamEnsemble]] = {}

    def __init_subclass__(cls, ensemble_type: str, **kwargs):
        super().__init_subclass__(**kwargs)
        ParamEnsemble._registry[ensemble_type] = cls

    def __init__(
        self,
        config: EnsembleConfig,
    ):

        main, pft_sheets = read_param_list(config.param_data_file)

        # subset to only a list of parameters if supplied
        if config.param_list is not None:
            main = main[main.parameter_name.isin(config.param_list)].copy()

        # create output directory if it doesn't exit yet
        self.ensemble_dir = Path(config.ensemble_dir)
        self.ensemble_dir.mkdir(parents=True, exist_ok=True)

        # set attributes
        self.file_prefix = config.file_prefix
        self.default_ds = xr.open_dataset(config.default_param_file)
        
        if config.posterior_sources:
            with open(config.posterior_sources, "r", encoding="utf-8") as f:
                posterior_config = yaml.safe_load(f)

        # create sorted list of parameter objects
        # this automatically checks and sorts order to make sure anything
        # that depends on another parameter is either not being modified or
        # gets written first
        self.params = sort_params(
            [
                Parameter.from_row(
                    row,
                    pft_sheet=pft_sheets.get(row["parameter_name"]),
                    default_ds=self.default_ds,
                    posterior_config=posterior_config.get(row["parameter_name"]) if config.posterior_sources else None,
                )
                for _, row in main.iterrows()
            ]
        )
        self.num_params = len(self.params)

        self.fixed_indices: dict[str, list[int]] = config.fixed_indices or {}
        if self.fixed_indices:
            _validate_fixed_indices(self.fixed_indices, self.default_ds)


    @classmethod
    def from_dict(
        cls,
        config: dict,
    ) -> ParamEnsemble:
        """Construct the correct ParamEnsemble subclass from an input configuration dict.

        The dict must contain an ``'ensemble_type'`` key whose value matches
        a registered subclass (e.g. ``'LatinHypercube'``). All other keys
        are passed to the corresponding config dataclass.

        Args:
            config (dict): Configuration dictionary. Example::

            {
                'ensemble_type': 'LatinHypercube',
                'param_data_file': 'params.xlsx',
                'ensemble_dir': 'output/',
                'file_prefix': 'my_run',
                'default_ds': ds,
                'n_samples': 200,
                'fixed_indices': {'fates_pft': [7, 8, 9]},
                'posterior_sources': 'posteriors.yaml',

            }
        Raises:
            ValueError
                If ``'ensemble_type'`` is missing or not a registered type.
            TypeError
                If the config dict contains keys not recognised by the config
                dataclass (misspelled or unsupported options).
        Returns:
            ParamEnsemble: A fully constructed ensemble subclass instance.
        """
        config = config.copy()

        ensemble_type = config.pop("ensemble_type", None)
        if ensemble_type is None:
            raise ValueError(
                "'ensemble_type' is required in the config dict. "
                f"Valid types: {sorted(cls._registry)}"
            )

        subclass = cls._registry.get(ensemble_type)
        if subclass is None:
            raise ValueError(
                f"Unknown ensemble_type '{ensemble_type}'. "
                f"Valid types: {sorted(cls._registry)}"
            )
        return subclass.from_config(config)
    
    
    def create_ensemble(self):
        """Create and write out all ensemble parameter files

        Args:
            ensemble parameter files.
        """
        samples = self.create_samples()
        for i, sample in enumerate(samples):
            ds = self.create_ensemble_member(sample)
            file_name = f"{self.file_prefix}_{_generate_suffix(i)}.nc"
            ds.to_netcdf(self.ensemble_dir / file_name)
            ds.close()

        ensemble_key = self.create_ensemble_key(samples)
        ensemble_key.to_csv(self.ensemble_dir / f"{self.file_prefix}_key.csv")

        _write_ensemble_list(
            self.ensemble_dir,
            self.file_prefix,
            list(ensemble_key.ensemble.values),
        )


    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> ParamEnsemble:
        """Construct this subclass from a config dict (ensemble_type already removed).

        Implementations should construct the appropriate config dataclass
        from the dict, then pass it to the constructor:

        cfg = LatinHypercubeConfig(**config)
        return cls(cfg)

        Args:
            config (dict): Config dict with 'ensemble_type' already popped.

        Returns:
            ParamEnsemble: A fully constructed ensemble subclass instance.
        """

    @abstractmethod
    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member, 
            each containing one ParameterSample per parameter.
        """

    @abstractmethod
    def create_ensemble_member(self, sample: EnsembleMemberSample) -> xr.Dataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member. 
                Contains one ParameterSample per parameter.
        Returns:
            xr.Dataset: one member of the ensemble with updated values from default
        """

    @abstractmethod
    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble 
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """

class LatinHypercubeEnsemble(ParamEnsemble, ensemble_type="LatinHypercube"):
    """Concrete class for the Latin Hypercube ensemble class

    Parameters
    ----------
    n_samples : int
        number of ensemble members to generate. Each member corresponds
        to one row of the Latin Hypercube sample matrix.
    prebuilt: np.ndarray | None
        Optional pre-built Latin Hypercube sample matrix of shape
        (n_samples, n_params). If supplied, this matrix is used directly
        instead of generating a new one. Useful for reproducibility or
        when the LH matrix was generated externally.
    posterior_configs: dict[str, PosteriorConfig]
        PosteriorConfig objects for pulling from a distribution

    """

    def __init__(
        self,
        config: LatinHypercubeConfig,
    ):
        super().__init__(config)
        self.n_samples = config.ensemble_members
        self.prebuilt = config.prebuilt

    @classmethod
    def from_config(cls, config: dict) -> LatinHypercubeEnsemble:
        """Construct from a plain dict (ensemble_type already removed)

        Args:
            config (dict): Must contain all required ``LatinHypercubeConfig`` fields.
            Unrecognised keys raise ``TypeError``.

        Raises:
            TypeError: Uncrecogized or missing keys

        Returns:
            LatinHypercubeEnsemble: A fully constructed ensemble LatinHypercubeEnsemble
            instance.
        """
        try:
            cfg = LatinHypercubeConfig(**config)
        except TypeError as e:
            raise TypeError(
                f"Invalid config key for LatinHypercubeEnsemble: {e}. "
                f"Valid keys: {[f.name for f in fields(LatinHypercubeConfig)]}"
            ) from e
        return cls(cfg)

    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member, 
            each containing one ParameterSample per parameter.
        """
        
        # build latin hypercube
        latin_hypercube = self.build_lh(len(self.params), self.prebuilt)

        # draw samples
        samples = []
        for i in range(self.n_samples):
            ensemble_member = EnsembleMemberSample(
                parameter_samples = [
                    ParameterSample(
                        parameter=param,
                        normalized_value=float(latin_hypercube[i, j]),
                    )
                    for j, param in enumerate(self.params)
                ]
            )
            samples.append(ensemble_member)

        return samples

    def create_ensemble_member(self, sample: EnsembleMemberSample) -> xr.Dataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member. 
                Contains one ParameterSample per parameter.
        Returns:
            xr.Dataset: one member of the ensemble with updated values from default
        """
        
        ds = self.default_ds.copy(deep=False)
        
        for param_sample in sample:
            param = param_sample.parameter
            normalized_value = param_sample.normalized_value
            
            value = param.sample(normalized_value, self.default_ds)
            
            param.set_value(
                ds, self.default_ds, value, fixed_indices=self.fixed_indices
            )

        return ds

    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble 
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """
        
        param_dfs = []
        for i, sample in enumerate(samples):
            parameter_names = []
            sample_values = []
            for param_sample in sample:
                parameter_names.append(param_sample.parameter.spec.name)
                sample_values.append(param_sample.normalized_value)
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
    
        
class OneAtATimeParameterEnsemble(ParamEnsemble, ensemble_type="OAT"):
    """Concrete class for the One-at-a-time (OAT) ensemble class
    """

    def __init__(
        self,
        config: OneAtATimeConfig,
    ):
        super().__init__(config)
        
    @classmethod
    def from_config(cls, config: dict) -> OneAtATimeParameterEnsemble:
        """Construct from a plain dict (ensemble_type already removed)

        Args:
            config (dict): Must contain all required ``OneAtATimeConfig`` fields.
            Unrecognised keys raise ``TypeError``.

        Raises:
            TypeError: Uncrecogized or missing keys

        Returns:
            OneAtATimeParameterEnsemble: A fully constructed ensemble OneAtATimeParameterEnsemble
            instance.
        """
        try:
            cfg = OneAtATimeConfig(**config)
        except TypeError as e:
            raise TypeError(
                f"Invalid config key for OneAtATimeConfig: {e}. "
                f"Valid keys: {[f.name for f in fields(OneAtATimeConfig)]}"
            ) from e
        return cls(cfg)
    
    
    def create_samples(self) -> list[EnsembleMemberSample]:
        """Create samples from the list of parameters

        Returns:
            list[EnsembleMemberSample]: One EnsembleMemberSample per ensemble member, 
            each containing one ParameterSample per parameter.
        """
        
        samples = []
        for param in self.params:
            samples.append(
                EnsembleMemberSample([ParameterSample(param, 0.0)]) # minimum
            )
            samples.append(
                EnsembleMemberSample([ParameterSample(param, 1.0)]) # maximum
            )
        return samples
    
    
    def create_ensemble_member(self, sample: EnsembleMemberSample) -> xr.Dataset:
        """Create one member of the ensemble

        Args:
            sample (EnsembleMemberSample): Parameter samples for this member. 
                Contains one ParameterSample per parameter.
        Returns:
            xr.Dataset: one member of the ensemble with updated values from default
        """
        if len(sample) != 1:
            raise ValueError (
                f"OneAtATimeEnsemble expects exactly one ParameterSample per member "
                f"but got {len(sample)}."
            )
        
        param_sample = sample.parameter_samples[0]
        param = param_sample.parameter
        normalized_value = param_sample.normalized_value
        
        ds = self.default_ds.copy(deep=False)
        value = param.sample(normalized_value, self.default_ds)
        param.set_value(ds, self.default_ds, value, fixed_indices=self.fixed_indices)
            
        return ds
    
    
    def create_ensemble_key(self, samples: list[EnsembleMemberSample]) -> pd.DataFrame:
        """Create the ensemble key that goes with this ensemble

        Args:
            samples (list[EnsembleMemberSample]): One EnsembleMemberSample per ensemble 
                member, each containing one ParameterSample per parameter.

        Returns:
            pd.DataFrame: output data frame that serves as ensemble key
        """
        parameter_names = []
        ensembles = []
        values = []
        for i, sample in enumerate(samples):
            if len(sample) != 1:
                raise ValueError (
                    f"OneAtATimeEnsemble expects exactly one ParameterSample per member "
                    f"but got {len(sample)}."
                )
            
            param_sample = sample.parameter_samples[0]
            param = param_sample.parameter
            normalized_value = param_sample.normalized_value
            parameter_names.append(param.spec.name)
            values.append(normalized_value)
            ensembles.append(f"{self.file_prefix}_{_generate_suffix(i)}")

        param_df = pd.DataFrame({
            "ensemble": ensembles,
            "parameter": parameter_names, 
            "normalized_value": values
            })
        
        param_df['direction'] = param_df.normalized_value.map({0.0: 'minimum', 1.0: 'maximum'})
        nan_mask = param_df['direction'].isna()
        if nan_mask.any():
            raise ValueError (
                f"OneAtATimeEnsemble expects only 0.0 or 1.0 normalized values but found "
                f"{param_df[nan_mask]}"
                f"In the ensemble key"
            )
        return param_df.drop(columns=['normalized_value'])
    

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


def _validate_fixed_indices(
    fixed_indices: dict[str, list[int]],
    default_ds: xr.Dataset,
) -> None:
    """Check to make sure supplied fixed_indices dict is compatible with input
    default parameter dataset

    Args:
        fixed_indices (dict[str, list[int]]):  Mapping of dimension 
            name to 0-based indices to hold at default.
        default_ds (xr.Dataset): Default parameter dataset.

    Raises:
        ValueError: fixed_indices dimension not found in dataset
        ValueError: fixed_indices has out-of-range indices
    """
    for dim, indices in fixed_indices.items():
        if dim not in default_ds.sizes:
            raise ValueError(
                f"fixed_indices dimension '{dim}' not found in default dataset. "
                f"Available dimensions: {sorted(default_ds.sizes)}"
            )
        dim_size = default_ds.sizes[dim]
        bad = [i for i in indices if i < 0 or i >= dim_size]
        if bad:
            raise ValueError(
                f"fixed_indices['{dim}'] contains out-of-range indices {bad}. "
                f"Dimension '{dim}' has size {dim_size} (valid: 0–{dim_size - 1})."
            )

