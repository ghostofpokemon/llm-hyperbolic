import requests
from typing import Optional, Dict, List, Any, Tuple
import time
import llm
from llm import Model
from llm.default_plugins.openai_models import Chat, Completion, SharedOptions
import click
from pydantic import Field, Extra
import json
import base64
from io import BytesIO
from PIL import Image
import os
import subprocess
import re
import threading
from pathlib import Path
import httpx
from enum import Enum

audio_lock = threading.Lock()

class ModelType(Enum):
    TTS = "tts"
    IMAGE = "image"
    VISION = "vision"
    TEXT = "text"

# List of models to exclude from registration
EXCLUDED_MODELS = ["StableDiffusion"]

def fetch_cached_json(url: str, path: Path, cache_timeout: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        mod_time = path.stat().st_mtime
        if time.time() - mod_time < cache_timeout:
            with open(path, "r") as file:
                return json.load(file)

    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        with open(path, "w") as file:
            json.dump(data, file)
        return data
    except httpx.HTTPError:
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            raise Exception(f"Failed to download data and no cache is available at {path}")

def get_hyperbolic_models() -> List[Dict[str, Any]]:
    key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
    if not key:
        print("Hyperbolic API key not found.")
        return []

    url = "https://api.hyperbolic.xyz/v1/models"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        return models
    except requests.RequestException as e:
        print(f"Failed to fetch models: {e}")
        return []

def get_model_ids_with_aliases() -> List[Tuple[str, List[str], ModelType]]:
    return [
        ("FLUX.1-dev", ["hyper-flux"], ModelType.IMAGE),
        ("SDXL1.0-base", ["hyper-sdxl"], ModelType.IMAGE),
        ("SD1.5", ["hyper-sd15"], ModelType.IMAGE),
        ("SD2", ["hyper-sd2"], ModelType.IMAGE),
        ("SSD", ["hyper-ssd"], ModelType.IMAGE),
        ("SDXL-turbo", ["hyper-sdxl-turbo"], ModelType.IMAGE),
        ("playground-v2.5", ["hyper-playground"], ModelType.IMAGE),
        ("SD1.5-ControlNet", ["hyper-sd15-controlnet"], ModelType.IMAGE),
        ("SDXL-ControlNet", ["hyper-sdxl-controlnet"], ModelType.IMAGE),
        ("TTS", ["hyper-tts"], ModelType.TTS),
        ("meta-llama/Meta-Llama-3.1-405B-FP8", ["hyper-l405-fp8"], ModelType.TEXT),
        ("meta-llama/Meta-Llama-3.1-405B", ["hyper-l405"], ModelType.TEXT),
        ("meta-llama/Meta-Llama-3.1-405B-Instruct", ["hyper-l405i"], ModelType.TEXT),
        ("NousResearch/Hermes-3-Llama-3.1-70B", ["hyper-h70"], ModelType.TEXT),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", ["hyper-l70i"], ModelType.TEXT),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", ["hyper-l8i"], ModelType.TEXT),
        ("meta-llama/Meta-Llama-3-70B-Instruct", ["hyper-l70i"], ModelType.TEXT),
        ("mistralai/Pixtral-12B-2409", ["hyper-pixtral"], ModelType.VISION),
        ("deepseek-ai/DeepSeek-V2.5", ["hyper-ds"], ModelType.TEXT),
        ("Qwen/Qwen2-VL-7B-Instruct", ["hyper-q7vi"], ModelType.VISION),
        ("Qwen/Qwen2.5-72B-Instruct", ["hyper-q72i"], ModelType.TEXT),
        ("meta-llama/Llama-3.2-90B-Vision", ["hyper-l90v"], ModelType.VISION),
        ("Qwen/Qwen2-VL-72B-Instruct", ["hyper-q72vi"], ModelType.VISION),
        ("meta-llama/Llama-3.2-3B-Instruct", ["hyper-l3i"], ModelType.TEXT),
        ("Qwen/Qwen2.5-Coder-32B-Instruct", ["hyper-qcode-32i"], ModelType.TEXT),
        ("meta-llama/Llama-3.1-70B-Instruct-FP8", ["hyper-l70i"], ModelType.TEXT),
        ("meta-llama/Llama-3.2-90B-Vision-Instruct", ["hyper-l90vi"], ModelType.VISION),
    ]

class HyperbolicBase(Model):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    can_stream = False  # Default, can be overridden in subclasses

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id  # Keep the full model_id (e.g., hyperbolic/FLUX.1-dev)
        self.api_base = ""
        self.aliases = kwargs.pop('aliases', [])

    def __str__(self):
        # Use a set to ensure aliases are unique and prevent duplication
        aliases_set = set(self.aliases)
        aliases_str = ', '.join(sorted(aliases_set)) if aliases_set else ''
        return f"Hyperbolic: {self.model_id}"

    def full_model_id(self) -> str:
        return self.model_id


class HyperbolicTTS(Model):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    can_stream = False
    model_type = ModelType.TTS.value

    class Options(llm.Options):
        speed: float = Field(default=1.0, description="Speed of speech (0.5 to 2.0)")

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id  # Full model ID, e.g., "hyperbolic/TTS"
        self.model_name = model_id.replace("hyperbolic/", "")
        self.api_base = "https://api.hyperbolic.xyz/v1/audio/generation"
        self.aliases = kwargs.pop('aliases', [])
        self.audio_playing = False  # Flag to ensure audio is only played once

    def __str__(self):
        return f"Hyperbolic: {self.model_id}"

    def execute(self, prompt, stream, response, conversation=None, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }
        data = {
            "text": prompt.prompt,
            "speed": prompt.options.speed
        }

        response._prompt_json = data
        try:
            api_response = requests.post(self.api_base, headers=headers, json=data)
            api_response.raise_for_status()
            response_json = api_response.json()
            audio_data = base64.b64decode(response_json["audio"])

            # Save the audio file
            filename = f"tts_output_{int(time.time())}.wav"
            with open(filename, "wb") as f:
                f.write(audio_data)

            response._text = f"Audio saved as: {filename}"
            response.response_json = response_json

            # Play audio asynchronously, but only if it's not already playing
            if not self.audio_playing:
                self.audio_playing = True
                threading.Thread(target=self.play_audio, args=(filename,), daemon=True).start()
        except requests.HTTPError as e:
            raise Exception(f"Error {api_response.status_code} from Hyperbolic API: {api_response.text}") from e

        return response._text

    def play_audio(self, filename: str):
        with audio_lock:  # Use the lock to ensure only one audio plays at a time
            try:
                subprocess.run(["afplay", filename], check=True)
            except subprocess.CalledProcessError:
                print("Unable to play audio with afplay. Please check if it's installed.")
            except FileNotFoundError:
                print("afplay not found. Please install it to play audio in the terminal.")
            finally:
                self.audio_playing = False  # Reset the flag after playing

    def prompt(self, prompt, *args, **kwargs):
        stream = kwargs.pop('stream', False)
        options = self.Options(**kwargs)
        llm_prompt = llm.Prompt(prompt, model=self, options=options)
        response = llm.Response(model=self, prompt=llm_prompt, stream=stream)
        self.execute(llm_prompt, stream=stream, response=response)
        return response

class HyperbolicImage(Model):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    can_stream = False
    model_type = ModelType.IMAGE.value

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

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id  # Full model ID, e.g., "hyperbolic/SDXL1.0-base"
        self.model_name = model_id.replace("hyperbolic/", "")
        self.api_base = "https://api.hyperbolic.xyz/v1/image/generation"
        self.aliases = kwargs.pop('aliases', [])

    def __str__(self):
        return f"Hyperbolic: {self.model_id}"

    def encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def execute(self, prompt, stream, response, conversation=None, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }
        data = {
            "model_name": self.model_name,  # Use self.model_name here
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
            try:
                api_response = requests.post(self.api_base, headers=headers, json=data)
                api_response.raise_for_status()
                response.response_json = api_response.json()
                break  # Exit the retry loop if successful
            except requests.HTTPError as e:
                if api_response.status_code == 429:
                    print(f"Rate limit exceeded (429). Retrying in {delay} seconds...")
                    for remaining in range(delay, 0, -1):
                        print(f"Retrying in {remaining} seconds...", end="\r")
                        time.sleep(1)
                    delay *= 2  # Exponential backoff
                else:
                    error_data = api_response.json()
                    error_message = error_data.get("message", "")

                    # Handle ControlNet error
                    controlnet_error = re.findall(r"'(.+?)'", error_message)
                    if "Unexpected controlnet_name" in error_message and controlnet_error:
                        available_controlnets = controlnet_error[0].split("', '")
                        print("Error: The controlnet_name you provided is not supported.")
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
                            available_options = re.findall(r"'(.+?)'", error_message)
                            if available_options:
                                available_options = available_options[0].split("', '")
                                print(f"Please choose one from the available options: {tuple(available_options)}")
                                new_value = input(f"Enter a valid {param}: ").strip()
                                while new_value not in available_options:
                                    print(f"Invalid option. Please choose from: {tuple(available_options)}")
                                    new_value = input(f"Enter a valid {param}: ").strip()
                            else:
                                new_value = input(f"Please enter a valid {param}: ").strip()
                            setattr(prompt.options, param, new_value)
                            return self.execute(prompt, stream, response, conversation)

                    # Unhandled error
                    raise Exception(f"Error {api_response.status_code} from Hyperbolic API: {api_response.text}") from e

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
            base_filename = f"{prompt_part}_{self.model_name}"  # Use self.model_name here
            if options_string:
                base_filename += f"_{options_string}"
            counter = 1
            while True:
                filename = f"{base_filename}_{counter}.png" if counter > 1 else f"{base_filename}.png"
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
        self.execute(llm_prompt, stream=stream, response=response)
        return response

class HyperbolicChat(Chat):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    model_type = ModelType.TEXT.value
    conversation_contexts: Dict[int, Dict[str, Any]] = {}  # Class variable to store contexts

    class Options(SharedOptions):
        image: Optional[str] = Field(default=None, description="Path to an image file for vision models")
        top_k: Optional[int] = Field(default=None, description="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.")

    def __init__(self, model_id: str, **kwargs):
        aliases = kwargs.pop('aliases', [])
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/chat/completions"
        self.aliases = aliases
        self.last_response = None

    def __str__(self):
        # Use a set to ensure aliases are unique and prevent duplication
        aliases_set = set(self.aliases)
        aliases_str = ', '.join(sorted(aliases_set)) if aliases_set else ''
        return f"Hyperbolic: {self.model_id}"

    def handle_tts_command(self, response):
        if self.last_response:
            tts_model = HyperbolicTTS("hyperbolic/TTS")
            tts_response = tts_model.prompt(self.last_response)
            return tts_response.text()
        else:
            return "No previous response to convert to speech."

    def execute(self, prompt, stream, response, conversation=None, **kwargs):
        if prompt.prompt.strip() == "!tts":
            tts_response = self.handle_tts_command(response)
            response._text = tts_response
            yield response._text
            return

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
                encoded_image = self.encode_image(prompt.options.image)

        if encoded_image and not image_sent:
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

        response._prompt_json = data

        try:
            api_response = requests.post(self.api_base, headers=headers, json=data, stream=stream)
            api_response.raise_for_status()

            full_response = ""
            if stream:
                for line in api_response.iter_lines():
                    if line:
                        try:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                line_text = line_text[6:]  # Remove 'data: ' prefix
                                if line_text.strip() == '[DONE]':
                                    break
                                chunk = json.loads(line_text)
                                content = chunk['choices'][0]['delta'].get('content')
                                if content:
                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse line: {line_text}")
                            continue

            self.last_response = full_response  # Store the last response
            response.response_json = {"content": full_response}

        except requests.HTTPError as e:
            print(f"An error occurred: {str(e)}")
            if e.response.status_code == 422:
                error_data = e.response.json()
                error_message = error_data.get("message", "")
                print(f"422 Unprocessable Entity: {error_message}")
                if "required" in error_message:
                    missing_fields = re.findall(r"'(.+?)'", error_message)
                    print(f"Missing required fields: {missing_fields}")
                elif "format" in error_message:
                    invalid_fields = re.findall(r"'(.+?)'", error_message)
                    print(f"Invalid format for fields: {invalid_fields}")
                else:
                    print(f"Error message: {error_message}")
            else:
                print(f"Error {e.response.status_code} from Hyperbolic API: {e.response.text}")

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

    def build_kwargs(self, prompt):
        # Return any additional kwargs needed for the API call
        kwargs = {}
        # Add any other parameters you need here
        return kwargs




class HyperbolicCompletion(Completion):
    needs_key = "hyperbolic"
    key_env_var = "LLM_HYPERBOLIC_KEY"
    model_type = ModelType.TEXT.value

    class Options(SharedOptions):
        image: Optional[str] = Field(default=None, description="Path to an image file for vision models")
        top_k: Optional[int] = Field(default=None, description="Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.")

    def __init__(self, model_id: str, **kwargs):
        aliases = kwargs.pop('aliases', [])
        super().__init__(model_id, **kwargs)
        self.api_base = "https://api.hyperbolic.xyz/v1/completions"
        self.aliases = aliases

    def __str__(self):
        aliases_set = set(self.aliases)
        aliases_str = ', '.join(sorted(aliases_set)) if aliases_set else ''
        return f"Hyperbolic: {self.model_id}"

    @staticmethod
    def encode_image(image_path):
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def execute(self, prompt, stream, response, conversation=None, **kwargs):
        messages = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append(prev_response.prompt.prompt)
                messages.append(prev_response.text())
        messages.append(prompt.prompt)

        if prompt.system:
            messages.insert(0, prompt.system)

        full_prompt = "\n".join(messages)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.get_key()}"
        }

        data = {
            "prompt": full_prompt,
            "model": self.model_name or self.model_id,
            "stream": stream,
            **self.build_kwargs(prompt)
        }

        if prompt.options.image:
            encoded_image = self.encode_image(prompt.options.image)
            data["images"] = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            }]

        response._prompt_json = data

        retries = 3
        delay = 5  # seconds

        for attempt in range(retries):
            try:
                api_response = requests.post(self.api_base, headers=headers, json=data, stream=stream)
                api_response.raise_for_status()

                if stream:
                    for line in api_response.iter_lines():
                        if line:
                            try:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    line_text = line_text[6:]  # Remove 'data: ' prefix
                                    if line_text.strip() == '[DONE]':
                                        break
                                    chunk = json.loads(line_text)
                                    text = chunk['choices'][0].get('text')
                                    if text:
                                        yield text
                            except json.JSONDecodeError:
                                print(f"Warning: Could not parse line: {line_text}")
                                continue

                response.response_json = {"content": "".join(response._chunks)}
                break  # Exit the retry loop if successful
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    print(f"Authentication error (401). Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                raise

    def build_kwargs(self, prompt):
        # Return any additional kwargs needed for the API call
        kwargs = {}
        # Add any other parameters you need here
        return kwargs


@llm.hookimpl
def register_models(register):
    key = llm.get_key("", "hyperbolic", "LLM_HYPERBOLIC_KEY")
    if not key:
        print("Hyperbolic API key not found. Skipping model registration.")
        return

    # Excluded models
    excluded_models = EXCLUDED_MODELS

    # Get existing models with aliases
    models_with_aliases = get_model_ids_with_aliases()
    existing_model_ids = {model_id for model_id, _, _ in models_with_aliases}

    # Initialize lists to hold models by type
    chat_models = []
    vision_models = []
    image_models = []
    completion_models = []
    tts_models = []

    # Collect models to register
    for model_id, aliases, model_type in models_with_aliases:
        if model_id in excluded_models:
            continue  # Skip excluded models

        if model_type == ModelType.TEXT:
            # Prepare chat model
            chat_aliases = [f"{alias}" for alias in aliases]
            chat_model = HyperbolicChat(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=chat_aliases,
            )
            chat_models.append((chat_model, chat_aliases))

            # Prepare completion model
            completion_aliases = [f"{alias}-base" for alias in aliases]
            completion_model = HyperbolicCompletion(
                model_id=f"hyperboliccompletion/{model_id}",
                model_name=model_id,
                aliases=completion_aliases,
            )
            completion_models.append((completion_model, completion_aliases))

        elif model_type == ModelType.VISION:
            # Prepare vision model
            vision_model = HyperbolicChat(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=aliases,
            )
            vision_models.append((vision_model, aliases))

        elif model_type == ModelType.IMAGE:
            # Prepare image model
            image_model = HyperbolicImage(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=aliases,
            )
            image_models.append((image_model, aliases))

        elif model_type == ModelType.TTS:
            # Prepare TTS model
            tts_model = HyperbolicTTS(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=aliases,
            )
            tts_models.append((tts_model, aliases))

    # Fetch dynamic models from the API
    fetched_models = get_hyperbolic_models()

    # Collect dynamic models
    for model_definition in fetched_models:
        model_id = model_definition["id"]
        if model_id in existing_model_ids or model_id in excluded_models:
            continue  # Skip already registered or excluded models

        model_type = determine_model_type(model_definition)

        if model_type == ModelType.TEXT:
            # Prepare chat model
            chat_model = HyperbolicChat(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=[],
            )
            chat_models.append((chat_model, []))

            # Prepare completion model
            completion_model = HyperbolicCompletion(
                model_id=f"hyperboliccompletion/{model_id}",
                model_name=model_id,
                aliases=[],
            )
            completion_models.append((completion_model, []))

        elif model_type == ModelType.VISION:
            # Prepare vision model
            vision_model = HyperbolicChat(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=[],
            )
            vision_models.append((vision_model, []))

        elif model_type == ModelType.IMAGE:
            # Prepare image model
            image_model = HyperbolicImage(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=[],
            )
            image_models.append((image_model, []))

        elif model_type == ModelType.TTS:
            # Prepare TTS model
            tts_model = HyperbolicTTS(
                model_id=f"hyperbolic/{model_id}",
                model_name=model_id,
                aliases=[],
            )
            tts_models.append((tts_model, []))

    # Register models in the desired order
    for model, aliases in chat_models:
        register(model, aliases=aliases)

    for model, aliases in vision_models:
        register(model, aliases=aliases)

    for model, aliases in image_models:
        register(model, aliases=aliases)

    for model, aliases in completion_models:
        register(model, aliases=aliases)

    for model, aliases in tts_models:
        register(model, aliases=aliases)


def determine_model_type(model_definition: Dict[str, Any]) -> ModelType:
    model_id = model_definition.get("id", "")
    if '/' not in model_id:
        if "tts" in model_id.lower():
            return ModelType.TTS
        else:
            return ModelType.IMAGE
    else:
        if model_definition.get("supports_chat", False):
            return ModelType.TEXT
        elif model_definition.get("supports_image_input", True):
            return ModelType.VISION
        else:
            return ModelType.TEXT  # Default to TEXT if uncertain
