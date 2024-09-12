import requests
from typing import Optional, Dict
import time
import httpx
import llm
from llm import Model
from llm.default_plugins.openai_models import Chat, Completion
import click
from pydantic import Field, Extra
import json
import base64
from io import BytesIO
from PIL import Image
import os
import subprocess

def get_model_ids_with_aliases():
    return [
        ("FLUX.1-dev", ["hyper-flux"], "image"),
        ("SDXL1.0-base", ["hyper-sdxl"], "image"),
        ("SD1.5", ["hyper-sd15"], "image"),
        ("SD2", ["hyper-sd2"], "image"),
        ("SSD", ["hyper-ssd"], "image"),
        ("SDXL-turbo", ["hyper-sdxl-turbo"], "image"),
        ("playground-v2.5", ["hyper-playground"], "image"),
        ("SD1.5-ControlNet", ["hyper-sd15-controlnet"], "image"),
        ("SDXL-ControlNet", ["hyper-sdxl-controlnet"], "image"),
        # ("Fluently-XL-v4", ["hyper-fluently-xl-v4"], "image"), # Error: Error 400 from Hyperbolic API: {"object":"error","message":"Runtime Error: We would fix it asap.","code":404}
        # ("Fluently-XL-Final", ["hyper-fluently-xl-final"], "image"), # Error: Error 400 from Hyperbolic API: {"object":"error","message":"Runtime Error: We would fix it asap.","code":404}
        # ("PixArt-Sigma-XL-2-1024-MS", ["hyper-pixart"], "image"), # Error: Error 400 from Hyperbolic API: {"object":"error","message":"Runtime Error: We would fix it asap.","code":404}
        # ("dreamshaper-xl-lightning", ["hyper-dreamshaper"], "image"), # Error: Error 400 from Hyperbolic API: {"object":"error","message":"Runtime Error: We would fix it asap.","code":404}
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect"], "chat"),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec"], "chat"),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec-tc"], "chat"),
        ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyper-base-fp8"], "completion"),
        ("meta-llama/Meta-Llama-3.1-405B", ["hyper-base"], "completion"),
        ("meta-llama/Meta-Llama-3.1-405B-Instruct", ["hyper-chat"], "chat"),
        ("NousResearch/Hermes-3-Llama-3.1-70B", ["hyper-hermes-70"], "chat"),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", ["hyper-llama-70"], "chat"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", ["hyper-llama-8"], "chat"),
        ("meta-llama/Meta-Llama-3-70B-Instruct", ["hyper-llama-3-70"], "chat"),
        ("Qwen/Qwen2-VL-7B-Instruct", ["hyper-qwen"], "chat"),
        ("deepseek-ai/DeepSeek-V2.5", ["hyper-seek"], "chat"),
    ]

class HyperbolicImage(Model):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    can_stream = False
    model_type = "image"

    class Options(llm.Options):
        height: int = Field(default=1024, description="Height of the image to generate")
        width: int = Field(default=1024, description="Width of the image to generate")
        backend: str = Field(default="auto", description="Computational backend (auto, tvm, torch)")
        prompt_2: Optional[str] = Field(default=None, description="Secondary prompt for Stable Diffusion XL")
        negative_prompt: Optional[str] = Field(default=None, description="Text specifying what the model should not generate")
        negative_prompt_2: Optional[str] = Field(default=None, description="Secondary negative prompt for Stable Diffusion XL")
        image: Optional[str] = Field(default=None, description="Path to reference image for img-to-img pipeline")
        strength: Optional[float] = Field(default=None, description="Strength of transformation for img-to-img (0-1)")
        seed: Optional[int] = Field(default=None, description="Seed for random number generation")
        cfg_scale: float = Field(default=7.5, description="Guidance scale for image relevance to prompt")
        sampler: Optional[str] = Field(default=None, description="Name of the sampling algorithm")
        steps: int = Field(default=30, description="Number of inference steps")
        style_preset: Optional[str] = Field(default=None, description="Style preset to guide the image model")
        enable_refiner: bool = Field(default=False, description="Enable Stable Diffusion XL-refiner")
        controlnet_name: Optional[str] = Field(default=None, description="Name of ControlNet to use")
        controlnet_image: Optional[str] = Field(default=None, description="Path to reference image for ControlNet")
        loras: Optional[Dict[str, float]] = Field(default=None, description="LoRA name and weight pairs")

        class Config:
            extra = Extra.allow
            protected_namespaces = ()

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id.replace("hyperbolic/", "")
        self.api_base = "https://api.hyperbolic.xyz/v1/image/generation"
        self.aliases = kwargs.pop('aliases', [])

    def __str__(self):
        return f"Hyperbolic: hyperbolic/{self.model_id}"

    def encode_image(self, image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def execute(self, prompt, stream, response, conversation=None):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }

        data = {
            "model_name": self.model_id,
            "prompt": prompt.prompt,
            "height": prompt.options.height,
            "width": prompt.options.width,
            "backend": prompt.options.backend,
        }

        optional_params = [
            "prompt_2", "negative_prompt", "negative_prompt_2", "image", "strength",
            "seed", "cfg_scale", "sampler", "steps", "style_preset", "enable_refiner",
            "controlnet_name", "controlnet_image", "loras"
        ]

        for param in optional_params:
            value = getattr(prompt.options, param)
            if value is not None:
                if param == "image" or param == "controlnet_image":
                    data[param] = self.encode_image(value)
                else:
                    data[param] = value

        response._prompt_json = data
        api_response = requests.post(self.api_base, headers=headers, json=data)

        if api_response.status_code != 200:
            raise Exception(f"Error {api_response.status_code} from Hyperbolic API: {api_response.text}")

        response.response_json = api_response.json()

        if 'images' in response.response_json and response.response_json['images']:
            base64_image = response.response_json['images'][0]['image']
            image_data = base64.b64decode(base64_image)

            # Create a more elegant filename
            prompt_part = "".join(c for c in prompt.prompt[:30] if c.isalnum() or c in (' ', '_')).rstrip()
            prompt_part = prompt_part.replace(' ', '_')

            options_part = []
            important_options = ['strength', 'cfg_scale', 'steps', 'seed']
            for key in important_options:
                value = getattr(prompt.options, key)
                if value is not None:
                    options_part.append(f"{key}-{value}")

            if prompt.options.image:
                options_part.append("img2img")
            if prompt.options.controlnet_name:
                options_part.append(f"controlnet-{prompt.options.controlnet_name}")
            if prompt.options.loras:
                options_part.append("lora")

            options_string = "_".join(options_part)

            base_filename = f"{prompt_part}_{self.model_id}"
            if options_string:
                base_filename += f"_{options_string}"

            counter = 1
            while True:
                if counter == 1:
                    filename = f"{base_filename}.png"
                else:
                    filename = f"{base_filename}_{counter}.png"

                if not os.path.exists(filename):
                    break
                counter += 1

            with open(filename, "wb") as f:
                f.write(image_data)

            response._text = f"Image saved as: {filename}"

            try:
                subprocess.run(["imgcat", filename], check=True)
            except subprocess.CalledProcessError:
                response._text += "\nUnable to display image with imgcat. Please check if it's installed."
            except FileNotFoundError:
                response._text += "\nimgcat not found. Please install it to display images in the terminal."
        else:
            response._text = "No image data received from the API"

        return response._text

    def prompt(self, prompt, *args, **kwargs):
        stream = kwargs.pop('stream', False)
        options = self.Options(**kwargs)
        llm_prompt = llm.Prompt(prompt, model=self, options=options)
        response = llm.Response(model=self, prompt=llm_prompt, stream=stream)
        result = self.execute(llm_prompt, stream=stream, response=response)
        return response

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
        return f"Hyperbolic: hyperbolic/{self.model_id}"

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
        return f"Hyperbolic: hyperbolic/{self.model_id}"

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

@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
    if not key:
        return
    models_with_aliases = get_model_ids_with_aliases()
    for model_id, aliases, model_type in models_with_aliases:
        if model_type == "chat":
            model_instance = HyperbolicChat(model_id=model_id, aliases=aliases)
        elif model_type == "completion":
            model_instance = HyperbolicCompletion(model_id=model_id, aliases=aliases)
        elif model_type == "image":
            model_instance = HyperbolicImage(model_id=model_id)
            model_instance.aliases = aliases
        else:
            continue  # Skip unknown model types

        register(model_instance, aliases=aliases)
