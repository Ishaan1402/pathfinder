"""Study-specific parameter validators.

Generic studies use search-space-only bounds validation (in hpo_coordinator.py).
Studies with domain-specific constraints register validators here.
"""
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field, field_validator


# --- Registry ---

_MANUAL_VALIDATORS: Dict[str, Callable] = {}


def register_manual_validator(study_name: str, validator: Callable) -> None:
    """Register a study-specific validator for manual trial parameters."""
    _MANUAL_VALIDATORS[study_name] = validator


def get_manual_validator(study_name: str) -> Optional[Callable]:
    """Return a registered validator for ``study_name``, or None."""
    return _MANUAL_VALIDATORS.get(study_name)


# --- U-Net bridge-crack validator ---

class UNetHyperparameters(BaseModel):
    learning_rate: float = Field(..., ge=1e-6, le=1e-1)
    batch_size: int = Field(..., ge=2, le=128)
    resolution: int = Field(..., ge=128, le=2048)
    model_capacity: str = Field(..., pattern="^(narrow|wide)$")
    loss_weight_ratio: float = Field(..., ge=0.0, le=1.0)

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v: int) -> int:
        if v % 32 != 0:
            raise ValueError("Resolution must be a multiple of 32 for U-Net downsampling compatibility.")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in [2, 4, 8, 16, 32, 64, 128]:
            raise ValueError("Batch size must be a power of 2 (e.g. 2, 4, 8, 16, 32, 64, 128).")
        return v


LEGACY_UNET_PARAMS = {
    "learning_rate",
    "batch_size",
    "resolution",
    "model_capacity",
    "loss_weight_ratio",
}


def validate_unet_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a parameter dict against UNetHyperparameters constraints."""
    valid = UNetHyperparameters(**params)
    return {"ok": True, "params": valid.model_dump(), "error": None, "warnings": []}
