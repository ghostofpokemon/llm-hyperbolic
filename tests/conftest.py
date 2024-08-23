import json
import llm
import pytest

DUMMY_MODELS = {
    "data": [
        {
            "id": "meta-llama/Meta-Llama-3.1-405B-FP8",
            "name": "Meta Llama 3.1 405B FP8",
            "context_length": 18000,
        },
        {
            "id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "name": "Meta Llama 3.1 405B Instruct",
            "context_length": 18000,
        },
        {
            "id": "NousResearch/Hermes-3-Llama-3.1-70",
            "name": "NousResearch Hermes 3 Llama 3.1 70",
            "context_length": 18000,
        },
        {
            "id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "name": "Meta Llama 3.1 70B Instruct",
            "context_length": 18000,
        },
        {
            "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "name": "Meta Llama 3.1 8B Instruct",
            "context_length": 18000,
        },
        {
            "id": "meta-llama/Meta-Llama-3-70B-Instruct",
            "name": "Meta Llama 3 70B Instruct",
            "context_length": 18000,
        },
    ]
}

@pytest.fixture
def user_path(tmpdir):
    dir = tmpdir / "llm.datasette.io"
    dir.mkdir()
    return dir

@pytest.fixture(autouse=True)
def env_setup(monkeypatch, user_path):
    monkeypatch.setenv("LLM_USER_PATH", str(user_path))
    # Write out the models.json file
    (llm.user_dir() / "hyperbolic_models.json").write_text(
        json.dumps(DUMMY_MODELS), "utf-8"
    )
