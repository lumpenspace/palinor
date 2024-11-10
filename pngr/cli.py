from os import path
import click
import code
import sys

from . import create_dataset
from .ControllableModel import ControllableModel
from .ControlVector import ControlVector


@click.group()
def pngr():
    """Pngr CLI"""
    pass


@click.command()
@click.argument("a", type=str)
@click.argument("b", type=str)
@click.option("--templates", "-t", type=click.Path(exists=True))
def dataset(templates: str, a: str, b: str):
    """Create a dataset of personality prompts."""
    template_path = templates or path.join(
        path.dirname(__file__), "..", "dataset_templates/alphapenger.yaml"
    )
    a_adjective = a
    b_adjective = b

    prompts = create_dataset.create_personality_prompts(
        template_path, a_adjective, b_adjective
    )

    # Save to output file
    output_file = "vector_dataset.jsonl"
    create_dataset.save_prompts(prompts, output_file)


@click.command()
@click.option("--dataset", "-d", type=click.Path(exists=True))
@click.option("--model-name", "-m", type=str)
def train(model_name: str, dataset: str):
    """Train a control vector."""
    pass


@click.command()
def shell():
    """Launch an interactive pngr shell."""
    # Prepare the namespace with commonly used classes/functions
    namespace = {
        "ControllableModel": ControllableModel,
        "ControlVector": ControlVector,
        "create_dataset": create_dataset,
    }

    banner = """
    Welcome to the pngr interactive shell!
    
    Available objects:
    - ControllableModel: Work with controllable language models
    - ControlVector: Create and manipulate control vectors
    - create_dataset: Dataset creation utilities
    
    Type help(<object>) for more information.
    """

    # Start the interactive shell
    try:
        code.InteractiveConsole(namespace).interact(banner=banner)
    except KeyboardInterrupt:
        sys.exit(0)


pngr.add_command(dataset)
pngr.add_command(train)
pngr.add_command(shell)
if __name__ == "__main__":
    pngr()
