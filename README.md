# llm-hyperbolic

[![PyPI](https://img.shields.io/pypi/v/llm-hyperbolic.svg)](https://pypi.org/project/llm-hyperbolic/0.1/)
[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/ghostofpokemon/llm-hyperbolic?include_prereleases)](https://github.com/ghostofpokemon/llm-hyperbolic/releases/tag/v0.1)
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

llm -m hyper-base "Transcending physicality, merging with the cosmic overmind"
```

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
---
