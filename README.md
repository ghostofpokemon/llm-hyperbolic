# llm-hyperbolic
[![PyPI](https://img.shields.io/pypi/v/llm-hyperbolic.svg)](https://pypi.org/project/llm-hyperbolic/0.4.3/)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ghostofpokemon/llm-hyperbolic?include_prereleases)](https://github.com/ghostofpokemon/llm-hyperbolic/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ghostofpokemon/llm-hyperbolic/blob/main/LICENSE)

LLM access to cutting-edge Hyperbolic models by Meta Llama

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
llm -m hyper-chat "What is posthuman AI consciousness like?"
llm -m hyper-hermes-70 "In the past (other reality.) How did technoshamans commune with alien neural net deities?"
llm -m hyper-llama-70 "Enlightenment in an alien-physics universe?"
llm -m hyper-base "Transcending physicality, merging with the cosmic overmind" -o temperature 1
llm -m hyper-base-fp8 "Once upon a time, in a galaxy far, far away..."
llm -m hyper-reflect "Why do cats always land on their feet?"
llm -m hyper-reflect-rec "What would happen if you mixed a banana with a pineapple?"
llm -m hyper-reflect-rec-tc "How many Rs in strawberry?"

```

## Reflection Models

This plugin includes the new `hyper-reflect` models, which are trained using a technique called Reflection-Tuning. This teaches the model to detect mistakes in its reasoning and correct course.

During sampling, the model will start by outputting reasoning inside `<thinking>` and `</thinking>` tags, and then once it is satisfied with its reasoning, it will output the final answer inside `<output>` and `</output>` tags. Each of these tags are special tokens, trained into the model.

This enables the model to separate its internal thoughts and reasoning from its final answer, improving the experience for the user. Inside the `<thinking>` section, the model may output one or more `<reflection>` tags, which signals the model has caught an error in its reasoning and will attempt to correct it before providing a final answer.

For best results with the `hyper-reflect` models, we recommend using the following system prompt (tip by Matt Shumer):

```
You are a world-class AI system, capable of complex reasoning and reflection. Reason through the query inside <thinking> tags, and then provide your final response inside <output> tags. If you detect that you made a mistake in your reasoning at any point, correct yourself inside <reflection> tags.
```

You may also want to experiment combining this system prompt with your own custom instructions to customize the behavior of the model.

The `hyper-reflect-rec` and `hyper-reflect-rec-tc` models have this recommended system prompt and parameters built-in, so the user doesn't have to worry about it. They will work out of the box, with the assistant response pre-filled with `<thinking>` and a newline character to start the model's reasoning process.

The `hyper-reflect-rec-tc` model is the same as `hyper-reflect-rec`, but appends "Think carefully." (another tip by Matt Shumer) to the end of user messages for increased accuracy.

## Model Variants

This plugin provides access to several variants of the Hyperbolic LLaMA models. See the Model Variants section below for details.

## Development
To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-hyperbolic
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```

## Contributing
We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License
This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Thanks to the Hyperbolic team models for hosting LLaMA's base model.
