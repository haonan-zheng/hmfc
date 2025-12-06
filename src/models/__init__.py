# public/models/__init__.py
from __future__ import annotations

from .base import HMFModel
from .zheng25 import Zheng25
from .reed07 import Reed07
from .sheth_tormen import ShethTormen
from .press_schechter import PressSchechter

# HMF models
_MODEL_REGISTRY: dict[str, type[HMFModel]] = {
    "zheng25": Zheng25,
    "Zheng25": Zheng25,
    "reed07": Reed07,
    "Reed07": Reed07,
    "st": ShethTormen,
    "ST": ShethTormen,
    "sheth_tormen": ShethTormen,
    "shethtormen": ShethTormen,
    "ShethTormen": ShethTormen,
    "ps": PressSchechter,
    "PS": PressSchechter,
    "press_schechter": PressSchechter,
    "pressschechter": PressSchechter,
    "PressSchechter": PressSchechter
}


def get_model(name: str) -> type[HMFModel]:
    """
    Return an HMF model class given its string name.

    Parameters
    ----------
    name : str
        Model name (case-insensitive).

    Returns
    -------
    model_cls : type[HMFModel]
        The corresponding model class.

    Raises
    ------
    ValueError
        If the model name is not recognized.
    """
    key = name.strip().lower()
    try:
        return _MODEL_REGISTRY[key]
    except KeyError:
        available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown HMF model '{name}'. Available models: {available}"
        )


__all__ = [
    "HMFModel",
    "Zheng25",
    "Reed07",
    "ShethTormen",
    "PressSchechter",
    "get_model"
]