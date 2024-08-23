from click.testing import CliRunner
from llm.cli import cli
import json
import pytest
import llm_hyperbolic

@pytest.mark.parametrize("set_key", (False, True))
def test_llm_models(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(json.dumps({"hyperbolic": "x"}), "utf-8")
    result = runner.invoke(cli, ["models", "list"])
    assert result.exit_code == 0, result.output
    fragments = (
        "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-405B-Instruct",
        "Hyperbolic Completion: hyperboliccompletion/meta-llama/Meta-Llama-3.1-405B-FP8",
        "Hyperbolic Chat: hyperbolicchat/NousResearch/Hermes-3-Llama-3.1-70",
        "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-70B-Instruct",
        "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3-70B-Instruct",
    )
    for fragment in fragments:
        if set_key:
            assert fragment in result.output
        else:
            assert fragment not in result.output

@pytest.mark.parametrize("set_key", (False, True))
def test_hyperbolic_models_command(set_key, user_path):
    runner = CliRunner()
    if set_key:
        (user_path / "keys.json").write_text(json.dumps({"hyperbolic": "x"}), "utf-8")
    result = runner.invoke(cli, ["hyperbolic_models"])
    assert result.exit_code == 0, result.output
    if set_key:
        assert "Hyperbolic Completion: hyperboliccompletion/meta-llama/Meta-Llama-3.1-405B-FP8" in result.output
        assert "Aliases: hyper-base" in result.output
        assert "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-405B-Instruct" in result.output
        assert "Aliases: hyper-chat" in result.output
        assert "Hyperbolic Chat: hyperbolicchat/NousResearch/Hermes-3-Llama-3.1-70" in result.output
        assert "Aliases: hyper-hermes-70" in result.output
        assert "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-70B-Instruct" in result.output
        assert "Aliases: hyper-llama-70" in result.output
        assert "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3.1-8B-Instruct" in result.output
        assert "Aliases: hyper-llama-8" in result.output
        assert "Hyperbolic Chat: hyperbolicchat/meta-llama/Meta-Llama-3-70B-Instruct" in result.output
        assert "Aliases: hyper-llama-3-70" in result.output
    else:
        assert "Hyperbolic API key not set" in result.output
