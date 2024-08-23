# llm_hyperbolic/__init__.py

from .llm_hyperbolic import (
    get_model_ids_with_aliases,
    HyperbolicChat,
    HyperbolicCompletion,
    register_models,
    register_commands,
)

__all__ = [
    "get_model_ids_with_aliases",
    "HyperbolicChat",
    "HyperbolicCompletion",
    "register_models",
    "register_commands",
]
