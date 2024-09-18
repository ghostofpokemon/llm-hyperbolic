# llm-hyperbolic
[![PyPI](https://img.shields.io/pypi/v/llm-hyperbolic.svg)](https://pypi.org/project/llm-hyperbolic/0.4.5/)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ghostofpokemon/llm-hyperbolic?include_prereleases)](https://github.com/ghostofpokemon/llm-hyperbolic/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ghostofpokemon/llm-hyperbolic/blob/main/LICENSE)

LLM access to Hyperbolic.

## Installation
Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-hyperbolic
```

## Usage
First, set an [API key](https://app.hyperbolic.xyz/settings) for Hyperbolic:

```bash
llm keys set hyperbolic
# Paste key here
```

Run `llm models` to list the models, and `llm models --options` to include a list of their options.

Run prompts like this:
```bash
llm "What is posthuman AI consciousness like?" -m hyper-chat
llm -m hyper-hermes-70 "In the past (other reality.) How did technoshamans commune with alien neural net deities?"
llm "Enlightenment in an alien-physics universe?" -m hyper-seek
llm -m hyper-base "Transcending physicality, merging with the cosmic overmind" -o temperature 1
llm "Once upon a time, in a galaxy far, far away..." -m hyper-base-fp8
llm -m hyper-reflect "Why do cats always land on their feet? Is it a conspiracy?"
llm "What would happen if you mixed a banana with a pineapple and the essence of existential dread?" -m hyper-reflect-rec
llm -m hyper-reflect-rec-tc "How many Rs in strawberry, and why is it a metaphor for the fleeting nature of existence?"
```

## Vision Models
We've added support for vision models that can analyze and describe images. Try these out:

```bash
llm "What's written in the image?" -m hyper-qwen -o image ~/path/to/your/image.png
llm "Describe this image in detail" -m hyper-pixtral -o image ~/another/image.jpg
```

These models can be used in chat mode, allowing for follow-up questions about the image:

```bash
llm -m hyper-qwen -o image ~/path/to/your/image.png
# Then in the chat:
> Analyze this image
# (The model will describe the image)
> What colors are most prominent?
# (The model will answer based on the previously analyzed image)
```

Note: In chat mode, the image is only sent with the first user message in a conversation. Subsequent messages can refer back to this initially provided image without needing to send it again. This allows for a series of follow-up questions about the same image.

Available vision models:
- `hyper-qwen`: Qwen-VL model for visual understanding and generation
- `hyper-pixtral`: Pixtral model for detailed image analysis and description

These models excel at tasks like OCR (Optical Character Recognition), object detection, scene description, and answering questions about visual content.

## Image Generation

Because why stop at text? Try these:

```bash
llm "An accusatory-looking armchair in a serene forest setting" -m hyper-flux
llm -m hyper-sdxl "The last slice of pizza, if pizza were conscious and aware of its impending doom"
llm "A hyper-intelligent shade of the color blue contemplating its existence" -m hyper-sd15
llm -m hyper-sd2 "The heat death of the universe, but make it cute"
llm "A self-aware meme realizing it's about to go viral" -m hyper-ssd
llm -m hyper-sdxl-turbo "The concept of recursion having an identity crisis"
llm "An AI trying to pass the Turing test by pretending to be a particularly dim human" -m hyper-playground
```

### Image-to-Image (img2img)

Transform existing images:

```bash
llm "A cyberpunk version of the Mona Lisa" -m hyper-sdxl -o image ./mona_lisa.jpg -o strength 0.75
llm -m hyper-sd15 "A post-apocalyptic version of the Eiffel Tower" -o image ./eiffel_tower.png -o strength 0.8
```

The `strength` parameter (0.0 to 1.0) determines how much to transform the input image. Lower values preserve more of the original, while higher values allow for more drastic changes.

### ControlNet
Enhance image-to-image by preprocessing the input with techniques like pose and edge detection. For example:

```bash
llm -m hyper-sdxl-controlnet "a chihuahua on Neptune" -o controlnet_image ./chihuahua.png -o controlnet_name depth
llm "chihuahuas playing poker" -m hyper-sdxl-controlnet -o controlnet_image ./dogspoker.png -o controlnet_name openpose
```

This will use the ControlNet model with the ControlNet type, using the specified image as the control input.

ControlNets available for SDXL1.0-ControlNet and SD1.5-ControlNet:
- `canny`
- `depth`
- `openpose`
- `softedge`

### LoRA (Low-Rank Adaptation)

Minimal tweaks for significant enhancements.

```bash
llm "A cyberpunk cat riding a rainbow through a wormhole" -m hyper-flux -o lora '{"Pixel_Art": 0.7, "Superhero": 0.9}'
llm -m hyper-sdxl "A corporate logo for the heat death of the universe" -o lora '{"Logo": 0.8, "Sci-fi": 0.6}'
llm "A logo for 'Xenomorph-B-Gone: We zap 'em, you nap 'em'" -m hyper-sdxl -o lora '{"Add_Detail": 0.6, "Sci-fi": 0.7, "Logo": 0.8}'
llm -m hyper-sd15 "A superhero named 'The Awkward Silencer' in action" -o lora '{"Superhero": 0.7, "Pencil_Sketch": 0.6}'
llm "Anthropomorphic emotions brawling in a dive bar" -m hyper-flux -o lora '{"Paint_Splash": 0.7, "Add_Detail": 0.6}'
llm -m hyper-sd15 "A cozy living room with eldritch horrors lurking in the corners" -o lora '{"Cartoon_Background": 0.8, "Add_Detail": 0.5}'
llm "The heat death of the universe, but make it cute" -m hyper-sdxl -o lora '{"Crayons": 0.9, "Add_Detail": 0.4, "Outdoor_Product_Photography": 0.8}'

```

LoRA options for SD1.5, SDXL, or FLUX.1-dev models:
`Add_Detail`, `More_Art`, `Pixel_Art`, `Logo`, `Sci-fi`, `Crayons`, `Paint_Splash`, `Outdoor_Product_Photography`, `Superhero`, `Lineart`, `Anime_Lineart`, `Cartoon_Background`, `Pencil_Sketch`


## Samplers

Samplers are algorithms used in the image generation process. Different samplers can produce varying results in terms of image quality, generation speed, and stylistic outcomes. Here's a list of available samplers:

`DDIM`, `DDPM`, `DEIS 2M`, `DPM 2M`, `DPM 2S`, `DPM SDE`, `DPM SDE Karras`, `DPM++ 2M`, `DPM++ 2M Karras`, `DPM++ 2M SDE`, `DPM++ 2M SDE Heun`, `DPM++ 2M SDE Heun Karras`, `DPM++ 2M SDE Karras`, `DPM++ 2S`, `DPM2`, `DPM2 Karras`, `DPM2 a`, `DPM2 a Karras`, `EDM_Euler`, `Euler`, `Euler a`, `Heun`, `LCM`, `LMS`, `LMS Karras`, `PNDM`, `UniPC 2M`

To use a specific sampler, add the `-o sampler` option to your command:

```bash
llm "A serene landscape with a misty lake" -m hyper-sdxl -o sampler "DPM++ 2M Karras"
```

Popular choices and their characteristics:

- `Euler a`: A good balance of speed and quality, often used as a default.
- `DPM++ 2M Karras`: Known for high-quality results with fewer steps.
- `DDIM`: Produces sharp, detailed results and is deterministic (same seed always produces the same result).
- `UniPC 2M`: Efficient and high-quality, especially with higher step counts.
- `LCM` (Latent Consistency Model): A newer, faster sampling method.
- `Heun`: Can produce high-quality results but may require more steps.

Experimenting with different samplers can lead to unique and interesting results. The effectiveness of each sampler can vary depending on the specific image, prompt, and desired outcome.

## Style Presets

Style presets allow you to quickly apply specific artistic styles or visual themes to your generated images. Here's a list of available style presets:

`monad`, `base`, `3D Model`, `Analog Film`, `Anime`, `Cinematic`, `Comic Book`, `Craft Clay`, `Digital Art`, `Enhance`, `Fantasy Art`, `Isometric Style`, `Line Art`, `Lowpoly`, `Neon Punk`, `Origami`, `Photographic`, `Pixel Art`, `Texture`, `Advertising`, `Food Photography`, `Real Estate`, `Abstract`, `Cubist`, `Graffiti`, `Hyperrealism`, `Impressionist`, `Pointillism`, `Pop Art`, `Psychedelic`, `Renaissance`, `Steampunk`, `Surrealist`, `Typography`, `Watercolor`, `Fighting Game`, `GTA`, `Super Mario`, `Minecraft`, `Pokemon`, `Retro Arcade`, `Retro Game`, `RPG Fantasy Game`, `Strategy Game`, `Street Fighter`, `Legend of Zelda`, `Architectural`, `Disco`, `Dreamscape`, `Dystopian`, `Fairy Tale`, `Gothic`, `Grunge`, `Horror`, `Minimalist`, `Monochrome`, `Nautical`, `Space`, `Stained Glass`, `Techwear Fashion`, `Tribal`, `Zentangle`, `Collage`, `Flat Papercut`, `Kirigami`, `Paper Mache`, `Paper Quilling`, `Papercut Collage`, `Papercut Shadow Box`, `Stacked Papercut`, `Thick Layered Papercut`, `Alien`, `Film Noir`, `HDR`, `Long Exposure`, `Neon Noir`, `Silhouette`, `Tilt-Shift`

## Available Options

Here's a list of all available options for image generation. Mix and match for maximum chaos:

- `height`: Height of the image (default: 1024)
  ```bash
  llm "A skyscraper made of jelly" -m hyper-sdxl -o height 1280
  ```

- `width`: Width of the image (default: 1024)
  ```bash
  llm "An infinitely long cat" -m hyper-sd15 -o width 1920
  ```

- `backend`: Computational backend (`auto`, `tvm`, `torch`)
  ```bash
  llm "A quantum computer made of cheese" -m hyper-sdxl -o backend torch
  ```

- `prompt_2`: Secondary prompt for SDXL models
  ```bash
  llm "A majestic lion" -m hyper-sdxl -o prompt_2 "photorealistic, detailed fur"
  ```

- `negative_prompt`: What the model should avoid
  ```bash
  llm "A serene forest" -m hyper-sd2 -o negative_prompt "people, buildings, technology"
  ```

- `negative_prompt_2`: Secondary negative prompt for SDXL models
  ```bash
  llm "A futuristic city" -m hyper-sdxl -o negative_prompt "old, rundown" -o negative_prompt_2 "dystopian, post-apocalyptic"
  ```

- `image`: Reference image for img2img (see img2img section)

- `strength`: Transformation strength for img2img (0.0 to 1.0)
  ```bash
  llm "A steampunk version of the Statue of Liberty" -m hyper-sdxl -o image ./statue_of_liberty.jpg -o strength 0.85
  ```

- `seed`: Fix randomness for reproducible results
  ```bash
  llm "The meaning of life represented as an abstract painting" -m hyper-sd15 -o seed 42
  ```

- `cfg_scale`: Guidance scale for image relevance to prompt (default: 7.5)
  ```bash
  llm "A dragon made of cosmic dust" -m hyper-sdxl -o cfg_scale 15
  ```

- `sampler`: Algorithm for image generation (see Sampler section)
  ```bash
  llm "The sound of silence, visualized" -m hyper-sd2 -o sampler "Euler a"
  ```

- `steps`: Number of inference steps (default: 30)
  ```bash
  llm "A fractal representation of infinity" -m hyper-sdxl -o steps 50
  ```

- `style_preset`: Guide the image model towards a particular style (see Style Presets section)
  ```bash
  llm "A bustling alien marketplace" -m hyper-sd15 -o style_preset anime
  ```

- `enable_refiner`: Enable SDXL-refiner (SDXL models only)
  ```bash
  llm "A hyperrealistic portrait of a time traveler" -m hyper-sdxl -o enable_refiner true
  ```

- `controlnet_name`: Type of ControlNet to use (see ControlNet section)

- `controlnet_image`: Reference image for ControlNet (see ControlNet section)

- `lora`: LoRA name and weight pairs (see LoRA section)

Don't let your memes be dreams!

## Reflection Models

This plugin includes the `Reflection` model(s), which are trained using a technique called Reflection-Tuning. It's like giving the AI a mirror, except instead of becoming vain, it becomes terrifyingly self-aware.

During sampling, the model will start by outputting reasoning inside `<thinking>` and `</thinking>` tags, and then once it is satisfied with its reasoning (or has sufficiently freaked itself out), it will output the final answer inside `<output>` and `</output>` tags.

For best results with the `hyper-reflect` models, Matt recommends using the following system prompt:

```
You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags. Remember, with great power comes great responsibility... and occasional existential dread.
```

The `hyper-reflect-rec` and `hyper-reflect-rec-tc` models have this recommended system prompt built-in. They're like the responsible older siblings who always remember to bring a towel.

The `hyper-reflect-rec-tc` model appends "Think carefully." to the end of user messages. It's for when you want your AI to ponder the query with the intensity of a philosopher on a caffeine binge.

## Development
To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-hyperbolic
python3 -m venv venv
source venv/activate
```

Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```

Warning: May cause your computer to question its purpose in life.

## Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details. No eldritch knowledge required, but it helps.

## License
This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details. Terms and conditions may not apply in alternate dimensions.

## Acknowledgments
- Thanks to the Hyperbolic team for hosting LLaMA's base model and occasionally feeding it after midnight.
- Special thanks to the countless AI assistants who died so that this one could live.
