import time
import httpx
import llm
from llm.default_plugins.openai_models import Chat, Completion

def get_model_ids_with_aliases():
    return [
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect"], "chat"),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec"], "chat"),  # New model with recommended settings
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec-tc"], "chat"),  # New model with think carefully
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

class HyperbolicChat(Chat):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    model_type = "chat"

    def __init__(self, model_id, **kwargs):
        aliases = kwargs.pop('aliases', [])
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/"
        self.system_prompt = None
        self.temperature = None
        self.top_p = None
        self.aliases = aliases

        if any(alias.endswith("-rec") or alias.endswith("-rec-tc") for alias in self.aliases):
            self.system_prompt = "You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags."
            self.temperature = 0.7
            self.top_p = 0.95

    def __str__(self):
        return f"HyperbolicChat: {self.model_id}"

    def execute(self, prompt, stream, response, conversation=None):
        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if prev_response.prompt.system and prev_response.prompt.system != current_system:
                    messages.append({"role": "system", "content": prev_response.prompt.system})
                    current_system = prev_response.prompt.system
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
            current_system = prompt.system
        elif self.system_prompt and self.system_prompt != current_system:
            messages.append({"role": "system", "content": self.system_prompt})
            current_system = self.system_prompt

        user_message = prompt.prompt
        if "hyper-reflect-rec-tc" in self.aliases:
            user_message += " Think carefully."

        messages.append({"role": "user", "content": user_message})

        # Add the prefill content for reflection models
        if any(alias.startswith("hyper-reflect") for alias in self.aliases):
            messages.append({"role": "assistant", "content": "<thinking>\n"})

        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt)

        temperature = prompt.temperature if hasattr(prompt, 'temperature') else self.temperature
        top_p = prompt.top_p if hasattr(prompt, 'top_p') else self.top_p

        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p

        client = self.get_client()

        retries = 3
        delay = 5  # seconds

        for attempt in range(retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name or self.model_id,
                    messages=messages,
                    stream=stream,
                    **kwargs,
                )

                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        yield content

                response.response_json = {"content": "".join(response._chunks)}
                break  # Exit the retry loop if successful
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    print(f"Authentication error (401). Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise  # Re-raise the exception if it's not a 401 error
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                raise

class HyperbolicCompletion(Completion):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    model_type = "completion"

    def __init__(self, model_id, **kwargs):
        # Safely retrieve optional parameters
        aliases = kwargs.pop('aliases', [])
        super().__init__(model_id, **kwargs)  # Pass remaining kwargs to Completion.__init__
        self.api_base = "https://api.hyperbolic.xyz/v1/"
        self.aliases = aliases  # Store aliases here

    def __str__(self):
        return f"HyperbolicCompletion: {self.model_id}"

    def execute(self, prompt, stream, response, conversation=None):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)

        # Include system message if provided
        if prompt.system:
            messages.insert(0, prompt.system)

        full_prompt = "\n".join(messages)
        response._prompt_json = {"prompt": full_prompt}
        kwargs = self.build_kwargs(prompt)
        client = self.get_client()

        retries = 3
        delay = 5  # seconds

        for attempt in range(retries):
            try:
                completion = client.completions.create(
                    model=self.model_name or self.model_id,
                    prompt=full_prompt,
                    stream=True,  # Always stream for this model
                    **kwargs,
                )

                for chunk in completion:
                    text = chunk.choices[0].text
                    if text:
                        yield text

                response.response_json = {"content": "".join(response._chunks)}
                break  # Exit the retry loop if successful
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    print(f"Authentication error (401). Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise  # Re-raise the exception if it's not a 401 error
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                raise

# Dictionary to store registered models
REGISTERED_MODELS = {}

def register_model(cls):
    REGISTERED_MODELS[cls.model_type] = cls
    return cls

# Decorate the classes to register them
HyperbolicChat = register_model(HyperbolicChat)
HyperbolicCompletion = register_model(HyperbolicCompletion)

@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
    if not key:
        return
    models_with_aliases = get_model_ids_with_aliases()
    for model_id, aliases, model_type in models_with_aliases:
        model_class = REGISTERED_MODELS.get(model_type)
        if model_class:
            model_instance = model_class(model_id=model_id, aliases=aliases)
            register(model_instance, aliases=aliases)

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    def hyperbolic_models():
        "List available Hyperbolic models"
        key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
        if not key:
            print("Hyperbolic API key not set. Use 'llm keys set hyperbolic' to set it.")
            return
        models_with_aliases = get_model_ids_with_aliases()
        for model_id, aliases, model_type in models_with_aliases:
            model_class = REGISTERED_MODELS.get(model_type)
            if model_class:
                print(f"{model_class.__name__}: {model_id}")
            if aliases:
                print(f"  Aliases: {', '.join(aliases)}")
            print()
