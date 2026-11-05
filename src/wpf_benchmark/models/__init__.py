"""Forecast models. Importing this package registers every built-in model.

Each model module registers itself with ``@register_model``. Modules must not import
torch or lightgbm at module level so that the registry loads without them.
"""

from .base import MODEL_REGISTRY, Forecaster, get_model, register_model
from . import naive, ridge, lightgbm_model, toto  # noqa: F401
from .neural import lstm, tcn, dlinear, patchtst, itransformer, agcrn  # noqa: F401

__all__ = ["MODEL_REGISTRY", "Forecaster", "get_model", "register_model"]
