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
import re

def get_model_ids_with_aliases():
    return [
        ("FLUX.1-dev", ["hyper-flux"], "image", False),
        ("SDXL1.0-base", ["hyper-sdxl"], "image", False),
        ("SD1.5", ["hyper-sd15"], "image", False),
        ("SD2", ["hyper-sd2"], "image", False),
        ("SSD", ["hyper-ssd"], "image", False),
        ("SDXL-turbo", ["hyper-sdxl-turbo"], "image", False),
        ("playground-v2.5", ["hyper-playground"], "image", False),
        ("SD1.5-ControlNet", ["hyper-sd15-controlnet"], "image", False),
        ("SDXL-ControlNet", ["hyper-sdxl-controlnet"], "image", False),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect"], "chat", False),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec"], "chat", False),
        ("mattshumer/Reflection-Llama-3.1-70B", ["hyper-reflect-rec-tc"], "chat", False),
        ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyper-base-fp8"], "completion", False),
        ("meta-llama/Meta-Llama-3.1-405B", ["hyper-base"], "completion", False),
        ("meta-llama/Meta-Llama-3.1-405B-Instruct", ["hyper-chat"], "chat", False),
        ("NousResearch/Hermes-3-Llama-3.1-70B", ["hyper-hermes-70"], "chat", False),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", ["hyper-llama-70"], "chat", False),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", ["hyper-llama-8"], "chat", False),
        ("meta-llama/Meta-Llama-3-70B-Instruct", ["hyper-llama-3-70"], "chat", False),
        ("Qwen/Qwen2-VL-7B-Instruct", ["hyper-qwen"], "chat", True),
        ("mistralai/Pixtral-12B-2409", ["hyper-pixtral"], "chat", True),
        ("deepseek-ai/DeepSeek-V2.5", ["hyper-seek"], "chat", False),
    ]



class HyperbolicImage(llm.Model):
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
                if param in ["image", "controlnet_image"]:
                    data[param] = self.encode_image(value)
                else:
                    data[param] = value

        if 'loras' in data:
            print("Warning: The API may silently accept invalid LoRAs. Please ensure you're using a valid LoRA name.")
            print("If the resulting image doesn't reflect the expected LoRA effect, the specified LoRA might not exist.")

        response._prompt_json = data
        retries = 3
        delay = 15  # seconds

        for attempt in range(retries):
            api_response = requests.post(self.api_base, headers=headers, json=data)
            if api_response.status_code == 200:
                response.response_json = api_response.json()
                break  # Exit the retry loop if successful
            elif api_response.status_code == 429:
                print(f"Rate limit exceeded (429). Retrying in {delay} seconds...")
                for remaining in range(delay, 0, -1):
                    print(f"Retrying in {remaining} seconds...", end="\r")
                    time.sleep(1)
                delay *= 2  # Exponential backoff
            else:
                error_data = json.loads(api_response.text)
                error_message = error_data.get("message", "")

                # Handle ControlNet error
                if "Unexpected controlnet_name" in error_message:
                    available_controlnets = re.findall(r"\['(.+?)'\]", error_message)
                    if available_controlnets:
                        available_controlnets = available_controlnets[0].split("', '")
                        print(f"Error: The controlnet_name you provided is not supported.")
                        print(f"Available ControlNet options: {tuple(available_controlnets)}")
                        new_controlnet = input("Please enter a valid controlnet_name: ").strip()
                        while new_controlnet not in available_controlnets:
                            print(f"Invalid option. Please choose from: {tuple(available_controlnets)}")
                            new_controlnet = input("Enter a valid controlnet_name: ").strip()
                        setattr(prompt.options, 'controlnet_name', new_controlnet)
                        return self.execute(prompt, stream, response, conversation)

                # Handle other parameter errors
                param_error_keys = {
                    "style_preset": "style_preset",
                    "sampler": "sampler",
                }

                for param, error_key in param_error_keys.items():
                    if error_key in error_message.lower():
                        print(f"Error: The {param} you provided is not supported.")
                        available_options = re.findall(r"\[(.+?)\]", error_message)
                        if available_options:
                            available_options = available_options[0].split(", ")
                            available_options = [opt.strip("'") for opt in available_options]
                            print(f"Please choose one from the available options: {tuple(available_options)}")
                            new_value = input(f"Enter a valid {param}: ").strip()
                            while new_value not in available_options:
                                print(f"Invalid option. Please choose from: {tuple(available_options)}")
                                new_value = input(f"Enter a valid {param}: ").strip()
                        else:
                            new_value = input(f"Please enter a valid {param}: ").strip()
                        setattr(prompt.options, param, new_value)
                        return self.execute(prompt, stream, response, conversation)

                # If we get here, it's an error we haven't specifically handled
                raise Exception(f"Error {api_response.status_code} from Hyperbolic API: {api_response.text}")

        if 'images' in response.response_json and response.response_json['images']:
            base64_image = response.response_json['images'][0]['image']
            image_data = base64.b64decode(base64_image)
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
    conversation_contexts = {}  # Class variable to store contexts

    class Options(llm.Options):
        image: Optional[str] = Field(default=None, description="Path to an image file for vision models")

    def __init__(self, model_id, **kwargs):
        aliases = kwargs.pop('aliases', [])
        self.is_vision_model = kwargs.pop('is_vision_model', False)
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/chat/completions"
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
        encoded_image = None
        image_sent = False

        if conversation is not None:
            context = self.get_conversation_context(conversation)
            image_sent = context.get('image_sent', False)
            for prev_response in conversation.responses:
                if prev_response.prompt.options.image and encoded_image is None:
                    encoded_image = self.encode_image(prev_response.prompt.options.image)
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append({"role": "assistant", "content": prev_response.text()})

        if prompt.options.image:
            if not self.is_vision_model:
                print(f"Warning: The model '{self.model_id}' does not support image input. Continuing without the image.")
                prompt.options.image = None
            elif encoded_image is None:
                encoded_image = self.encode_image(prompt.options.image)

        if self.is_vision_model and encoded_image and not image_sent:
            user_message = [
                {"type": "text", "text": prompt.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
            image_sent = True
        else:
            user_message = prompt.prompt

        messages.append({"role": "user", "content": user_message})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }

        data = {
            "model": self.model_name or self.model_id,
            "messages": messages,
            "stream": stream,
        }

        if hasattr(prompt, 'temperature') and prompt.temperature is not None:
            data["temperature"] = prompt.temperature
        elif self.temperature is not None:
            data["temperature"] = self.temperature

        if hasattr(prompt, 'top_p') and prompt.top_p is not None:
            data["top_p"] = prompt.top_p
        elif self.top_p is not None:
            data["top_p"] = self.top_p

        response._prompt_json = data

        try:
            api_response = requests.post(self.api_base, headers=headers, json=data, stream=stream)
            api_response.raise_for_status()

            if stream:
                for line in api_response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8').split('data: ')[1])
                        content = chunk['choices'][0]['delta'].get('content')
                        if content:
                            yield content
            else:
                response_json = api_response.json()
                content = response_json['choices'][0]['message']['content']
                yield content

            response.response_json = {"content": "".join(response._chunks)}

        except requests.RequestException as e:
            print(f"An error occurred: {str(e)}")
            raise

        if conversation is not None:
            self.set_conversation_context(conversation, {'image_sent': image_sent})

    @staticmethod
    def encode_image(image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @classmethod
    def get_conversation_context(cls, conversation):
        return cls.conversation_contexts.get(id(conversation), {'image_sent': False})

    @classmethod
    def set_conversation_context(cls, conversation, context):
        cls.conversation_contexts[id(conversation)] = context

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
    for model_id, aliases, model_type, is_vision_model in models_with_aliases:
        if model_type == "chat":
            model_instance = HyperbolicChat(model_id=model_id, aliases=aliases, is_vision_model=is_vision_model)
        elif model_type == "completion":
            model_instance = HyperbolicCompletion(model_id=model_id, aliases=aliases)
        elif model_type == "image":
            model_instance = HyperbolicImage(model_id=model_id)
            model_instance.aliases = aliases
        else:
            continue  # Skip unknown model types

        register(model_instance, aliases=aliases)
