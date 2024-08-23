import llm
from llm.default_plugins.openai_models import Chat, Completion

def get_model_ids_with_aliases():
    return [
        ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyper-base"], "completion"),
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
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/"

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
        messages.append({"role": "user", "content": prompt.prompt})
        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt)
        client = self.get_client()

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

class HyperbolicCompletion(Completion):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    model_type = "completion"

    def __init__(self, model_id, **kwargs):
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/"

    def __str__(self):
        return f"HyperbolicCompletion: {self.model_id}"

    def execute(self, prompt, stream, response, conversation=None):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)
        response._prompt_json = {"prompt": "\n".join(messages)}
        kwargs = self.build_kwargs(prompt)
        client = self.get_client()

        completion = client.completions.create(
            model=self.model_name or self.model_id,
            prompt="\n".join(messages),
            stream=True,  # Always stream for this model
            **kwargs,
        )

        for chunk in completion:
            text = chunk.choices[0].text
            if text:
                yield text

        response.response_json = {"content": "".join(response._chunks)}

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
            register(
                model_class(
                    model_id=model_id,
                    model_name=model_id,
                ),
                aliases=aliases
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
        models_with_aliases = get_model_ids_with_aliases()
        for model_id, aliases, model_type in models_with_aliases:
            model_class = REGISTERED_MODELS.get(model_type)
            if model_class:
                print(f"{model_class.__name__}: {model_id}")
            if aliases:
                print(f"  Aliases: {', '.join(aliases)}")
            print()
