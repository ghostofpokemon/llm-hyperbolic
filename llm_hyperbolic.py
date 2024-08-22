import llm
from llm.default_plugins.openai_models import Chat, Completion
import httpx
import ijson
import json

class HyperbolicChat(Chat):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"

    def __str__(self):
        return f"Hyperbolic Chat: {self.model_id}"

    def execute(self, messages, stream=False, **kwargs):
        key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
        if not key:
            raise ValueError("Hyperbolic API key not set. Use 'llm keys set hyperbolic' to set it.")

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        data.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()

                if stream:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                                if chunk.get('choices'):
                                    for choice in chunk['choices']:
                                        yield choice['message']['content']
                            except (json.JSONDecodeError, IndexError) as e:
                                raise ValueError(f"Error parsing streamed response: {e}") from e
                else:
                    response_data = response.json()
                    if not response_data.get("choices"):
                        raise ValueError("No response from Hyperbolic API")
                    yield response_data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_detail = error_json['error'].get('message', error_detail)
            except json.JSONDecodeError:
                pass
            raise ValueError(f"Hyperbolic API returned an error: {error_detail}") from e
        except httpx.RequestError as e:
            raise ValueError(f"Error communicating with Hyperbolic API: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON response from Hyperbolic API: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error occurred: {e}") from e

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
        }
        # Remove any None values from kwargs before updating data
        data.update({k: v for k, v in kwargs.items() if v is not None})

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=data)
                response.raise_for_status()

                if stream:
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                                if chunk.get('choices'):
                                    for choice in chunk['choices']:
                                        yield choice['text']
                            except (json.JSONDecodeError, IndexError) as e:
                                raise ValueError(f"Error parsing streamed response: {e}") from e
                else:
                    response_data = response.json()
                    if not response_data.get("choices"):
                        raise ValueError("No response from Hyperbolic API")
                    for choice in response_data["choices"]:
                        yield choice['text']
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_json = e.response.json()
                if 'error' in error_json:
                    error_detail = error_json['error'].get('message', error_detail)
            except json.JSONDecodeError:
                pass
            raise ValueError(f"Hyperbolic API returned an error: {error_detail}") from e
        except httpx.RequestError as e:
            raise ValueError(f"Error communicating with Hyperbolic API: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON response from Hyperbolic API: {e}") from e
        except Exception as e:
            raise ValueError(f"Unexpected error occurred: {e}") from e

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
