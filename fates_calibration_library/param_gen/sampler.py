"""
Sampler class - generates a collection of samples
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from scipy.stats import qmc

class Sampler(ABC):
    """Abstract Sampler class
    """
    
    def __init__(self, prebuilt: Optional[np.ndarray] = None):
        self.prebuilt = prebuilt
    
    @abstractmethod
    def draw(self, n_samples: int, n_lh_dims: int) -> np.ndarray:
        """Generate n_samples draws over the given n_parameters
        """

class LatinHypercubeSampler(Sampler):
    """Generates samples using a Latin Hypercube design.
 
    Parameters
    ----------
    prebuilt : np.ndarray, optional
        Pre-built LH array of shape (n_samples, n_lh_dims). Generated
        automatically if not supplied. Useful for reproducibility or
        if you want to supply your own sample.
    """
    
    def draw(
        self,
        n_samples: int,
        n_lh_dims: int,
    ) -> np.ndarray:
        """Generate n_samples draws over the given n_parameters

        Args:
            n_samples (int): Number of ensemble members.
            n_lh_dims (int): Number of parameters

        Returns:
            np.ndarray: output array
        """
        
        lh = self._build_lh(n_lh_dims, n_samples)
    
    def _build_lh(self, n_dims: int, n_samples: int) -> np.ndarray:
        
        if n_dims == 0:
            return np.empty((n_samples, 0))
        
        # validate prebuilt LH 
        if self.prebuilt is not None:
            if self.prebuilt.shape != (n_samples, n_dims):
                raise ValueError(
                    f"Pre-built LH sample has shape {self.prebuilt.shape}, "
                    f"expected ({n_samples}, {n_dims})."
                )
            return self.prebuilt
        
        # else return the LH
        return qmc.LatinHypercube(d=n_dims).random(n=n_samples)
        
        
        
        
        