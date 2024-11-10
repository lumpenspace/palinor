# Pngr

Pngr is a powerful tool designed to facilitate the generation of datasets, training of control vectors, and inference with dynamically injected control vectors. This document outlines the core functionalities and how to use them effectively.

## Installation

```bash
poetry install git+https://github.com/lumpenspace/pngr.git
```

## Features

`pngr` is a command line interface for the Pngr application.

### Dataset Generation for Training Control Vectors

Pngr allows users to generate datasets tailored for training control vectors. This process involves creating personality prompts based on specified adjectives, which are then used to train models to understand and generate text that aligns with these personality traits.

**How to Use:**

1. Run the `pngr dataset` command with the required adjectives.
2. The system will automatically generate a dataset and save it in a specified format.

### Control Vector Training

Once a dataset is prepared, Pngr can train control vectors that are capable of guiding the behavior of language models. These control vectors can be tuned to influence the generated text in specific ways, such as altering the tone, style, or thematic elements.

**How to Use:**

1. Use the `pngr train` command with the path to your dataset and the desired model configuration.
2. The training process will optimize control vectors to achieve the desired text manipulations.

### Inference with Dynamically Injected Control Vectors

Pngr supports dynamic injection of control vectors during inference, allowing real-time manipulation of text generation. This feature is particularly useful for applications requiring on-the-fly adjustments to the output, such as interactive chatbots or adaptive content generation systems.

**How to Use:**

1. Start the interactive shell using `pngr shell`.
2. Load the desired model and control vectors.
3. Perform inference by injecting different control vectors as needed to steer the output dynamically.

## License

(c) 2024 Lumpenspace and Vie McCoy

## Acknowledgements

This project was inspired by the work of [vgel](https://x.com/vooooooogel), [Repeng](https://github.com/vgel/repeng),
one of the most influential yet least cited projects in the field of controllable generation.
