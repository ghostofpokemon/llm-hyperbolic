# llm_hyperbolic/__init__.py

from .llm_hyperbolic import (
    get_model_ids_with_aliases,
    HyperbolicChat,
    HyperbolicCompletion,
    register_model,
    register_models,
    register_commands,
)

__all__ = [
    "get_model_ids_with_aliases",
    "HyperbolicChat",
    "HyperbolicCompletion",
    "register_model",
    "register_models",
    "register_commands",
]

# Version information
__version__ = "0.4.1"

# Docstrings
def get_model_ids_with_aliases():
    """
    Returns a list of model IDs with their corresponding aliases and types.
    """
    return [
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect"], "chat"),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec"], "chat"),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec-tc"], "chat"),
        ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyper-base-fp8"], "completion"),
        ("meta-llama/Meta-Llama-3.1-405B", ["hyper-base"], "completion"),
        ("meta-llama/Meta-Llama-3.1-405B-Instruct", ["hyper-chat"], "chat"),
        ("NousResearch/Hermes-3-Llama-3.1-70B", ["hyper-hermes-70"], "chat"),
        ("NousResearch/Hermes-3-Llama-3.1-70B-FP8", ["hyper-hermes-70-fp8"], "chat"),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", ["hyper-llama-70"], "chat"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", ["hyper-llama-8"], "chat"),
        ("meta-llama/Meta-Llama-3-70B-Instruct", ["hyper-llama-3-70"], "chat"),
        # ("01-ai/Yi-34B-Chat", ["hyper-yi-1"], "chat"),
        ("01-ai/Yi-1.5-34B-Chat", ["hyper-yi"], "chat"),
    ]

class HyperbolicChat:
    """
    A class representing a Hyperbolic Chat model.
    """
    pass

class HyperbolicCompletion:
    """
    A class representing a Hyperbolic Completion model.
    """
    pass

def register_model(cls):
    """
    Registers a model class.
    """
    pass

def register_models(register):
    """
    Registers all models with the given register function.
    """
    pass

def register_commands(cli):
    """
    Registers CLI commands with the given CLI object.
    """
    pass
