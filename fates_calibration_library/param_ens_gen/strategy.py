"""
Parameter strategy enum for parameter sampling strategies

Parameter strategy control's two things:
    1. Whether bounds (min/max) are meaningful for this parameter.
    2. Whether a posterior distribution is consulted at sample time.
"""

from __future__ import annotations
from enum import Enum

class Strategy(str, Enum):
    """Sampling strategy for a calibratable parameter.
 
    Variants
    --------
    UNIFORM
        Value is drawn from a [0-1] value and scaled between a
        resolved min and max bound. Bounds must be set; no posterior
        source is consulted.
    POSTERIOR
        Value is drawn from an external posterior distribution via
        PosteriorConfig. Bounds are meaningless and not set.
    """
    
    UNIFORM = 'uniform'
    POSTERIOR = 'posterior'

    def requires_bounds(self) -> bool:
        """Return True if this strategy draws from a posterior distribution.
        
        Used to decide whether a PosteriorConfig must be attached and
        whether PosteriorSource.draw() is called at sample time.

        Returns:
            bool
        """
        return self is Strategy.UNIFORM
    
    def requires_posterior(self) -> bool:
        """Return True if this strategy draws from a posterior distribution.
 
        Used to decide whether a PosteriorConfig must be attached and
        whether PosteriorSource.draw() is called at sample time.
 
        Returns:
            bool
        """
        return self is Strategy.POSTERIOR
    
    @classmethod
    def parse(cls, value: str) -> Strategy:
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError:
            valid = [s.value for s in cls]
            raise ValueError(
                f"Invalid strategy '{value}'. Must be one of: {valid}"
            )