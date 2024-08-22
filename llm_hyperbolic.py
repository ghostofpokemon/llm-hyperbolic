import llm
from llm.default_plugins.openai_models import Chat, Completion
import httpx
import ijson

class HyperbolicChat(Chat):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"

    def __str__(self):
        return f"Hyperbolic Chat: {self.model_id}"

class HyperbolicCompletion(Completion):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"

    def __str__(self):
        return f"Hyperbolic Completion: {self.model_id}"

    def execute(self, prompt, stream=False, **kwargs):
        key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
        if not key:
            raise ValueError("Hyperbolic API key not set. Use 'llm keys set hyperbolic' to set it.")

        url = f"{self.api_base}/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "prompt": str(prompt),
            "stream": stream,
            **kwargs
        }

        response = httpx.post(url, headers=headers, json=data)
        response.raise_for_status()

        chunks = []
        if stream:
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                        if chunk.get('choices'):
                            chunks.extend(chunk['choices'])
                    except (json.JSONDecodeError, IndexError):
                        continue
        else:
            response_data = response.json()
            if not response_data.get("choices"):
                raise ValueError("No response from Hyperbolic API")
            chunks = response_data["choices"]

        if not chunks:
            raise ValueError("No valid chunks received from Hyperbolic API")

        return chunks

@llm.hookimpl
def register_models(register):
    register(
        HyperbolicChat(
            model_id="hyperbolicchat/meta-llama/Meta-Llama-3.1-405B-Instruct",
            model_name="meta-llama/Meta-Llama-3.1-405B-Instruct",
            api_base="https://api.hyperbolic.xyz/v1",
        ),
        aliases=["hyperbolic-instruct-chat"]
    )
    register(
        HyperbolicCompletion(
            model_id="hyperboliccompletion/meta-llama/Meta-Llama-3.1-405B-FP8",
            model_name="meta-llama/Meta-Llama-3.1-405B-FP8",
            api_base="https://api.hyperbolic.xyz/v1",
        ),
        aliases=["hyperbolic-fp8-completion"]
    )

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    def hyperbolic_models():
        "List available Hyperbolic models"
        key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
        if not key:
            print("Hyperbolic API key not set. Use 'llm keys set hyperbolic' to set it.")
            return

        models = [
            ("meta-llama/Meta-Llama-3.1-405B-Instruct", ["hyperbolic-instruct-chat"]),
            ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyperbolic-fp8-completion"])
        ]

        for model_id, aliases in models:
            if "Instruct" in model_id:
                print(f"Hyperbolic Chat: hyperbolicchat/{model_id}")
            else:
                print(f"Hyperbolic Completion: hyperboliccompletion/{model_id}")
            print(f"  Aliases: {', '.join(aliases)}")
            print()
