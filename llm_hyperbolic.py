import requests
from typing import Optional
import time
import httpx
import llm
from llm import Model
from llm.default_plugins.openai_models import Chat, Completion
import click
from pydantic import Field, Extra
import base64
import os
import subprocess
import json

def get_model_ids_with_aliases():
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
        ("StableDiffusion", ["hyper-sd"], "image"),
        ("Monad", ["hyper-monad"], "image"),
        ("Wifhat", ["hyper-wifhat"], "image"),
        ("FLUX.1-dev", ["hyper-flux"], "image"),
    ]

class HyperbolicImage(Model):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    can_stream = False
    model_type = "image"

    class Options(llm.Options):
        steps: int = Field(default=30, description="Number of inference steps")
        cfg_scale: float = Field(default=5, description="CFG scale")
        enable_refiner: bool = Field(default=False, description="Enable refiner")
        height: int = Field(default=1024, description="Image height")
        width: int = Field(default=1024, description="Image width")
        backend: str = Field(default="auto", description="Backend to use")

        class Config:
            extra = Extra.allow

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.api_base = "https://api.hyperbolic.xyz/v1/image/generation"

    def __str__(self):
        return f"HyperbolicImage: {self.model_id}"

    @property
    def stream(self):
        return False

    def execute(self, prompt, stream, response, conversation=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }

        data = {
            "model_name": self.model_id,
            "prompt": prompt.prompt,
            "steps": prompt.options.steps,
            "cfg_scale": prompt.options.cfg_scale,
            "enable_refiner": prompt.options.enable_refiner,
            "height": prompt.options.height,
            "width": prompt.options.width,
            "backend": prompt.options.backend
        }

        response._prompt_json = data
        api_response = requests.post(self.api_base, headers=headers, json=data)

        if api_response.status_code != 200:
            raise Exception(f"Error {api_response.status_code} from Hyperbolic API: {api_response.text}")

        response.response_json = api_response.json()

        if stream:
            yield json.dumps(response.response_json)
        else:
            return json.dumps(response.response_json)

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
        aliases = kwargs.pop('aliases', [])
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/"
        self.aliases = aliases

    def __str__(self):
        return f"HyperbolicCompletion: {self.model_id}"

    def execute(self, prompt, stream, response, conversation=None):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)

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
                    stream=True,
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

REGISTERED_MODELS = {}

def register_model(cls):
    REGISTERED_MODELS[cls.model_type] = cls
    return cls

HyperbolicChat = register_model(HyperbolicChat)
HyperbolicCompletion = register_model(HyperbolicCompletion)
HyperbolicImage = register_model(HyperbolicImage)

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
    @click.option("--model", default="hyper-flux", help="Image model to use")
    @click.option("--prompt", required=True, help="Prompt for image generation")
    @click.option("--steps", default=30, help="Number of inference steps")
    @click.option("--cfg-scale", default=5.0, help="CFG scale")
    @click.option("--enable-refiner", is_flag=True, help="Enable refiner")
    @click.option("--height", default=1024, help="Image height")
    @click.option("--width", default=1024, help="Image width")
    @click.option("--backend", default="auto", help="Backend to use")
    def generate_image(model, prompt, steps, cfg_scale, enable_refiner, height, width, backend):
        "Generate an image using Hyperbolic API"
        model_instance = llm.get_model(model)
        if not isinstance(model_instance, HyperbolicImage):
            raise click.ClickException(f"Model {model} is not an image generation model")

        options = {
            "steps": steps,
            "cfg_scale": cfg_scale,
            "enable_refiner": enable_refiner,
            "height": height,
            "width": width,
            "backend": backend
        }
        response = model_instance.prompt(prompt, options=options)

        # Parse the JSON response
        try:
            response_data = json.loads(response.text())
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON response from the API")

        if 'images' not in response_data or not response_data['images']:
            raise click.ClickException("No image data received from the API")

        base64_image = response_data['images'][0]['image']
        image_data = base64.b64decode(base64_image)

        # Generate a unique filename
        base_filename = "".join(c for c in prompt if c.isalnum() or c in (' ', '_'))[:50].rstrip()
        counter = 1
        while True:
            if counter == 1:
                filename = f"{base_filename}.png"
            else:
                filename = f"{base_filename}_{counter}.png"

            if not os.path.exists(filename):
                break
            counter += 1

        # Save the image
        with open(filename, "wb") as f:
            f.write(image_data)

        click.echo(f"Image saved as: {filename}")

        # Display the image using imgcat
        try:
            subprocess.run(["imgcat", filename], check=True)
        except subprocess.CalledProcessError:
            click.echo("Unable to display image with imgcat. Please check if it's installed.")
        except FileNotFoundError:
            click.echo("imgcat not found. Please install it to display images in the terminal.")

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
