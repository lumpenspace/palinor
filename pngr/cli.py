from os import path
import click
import code
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.traceback import install as rich_traceback_install
from transformers import PreTrainedModel
from . import create_dataset
from .ControllableModel import ControllableModel
from .ControlVector import ControlVector

# Install Rich traceback globally in the script for better error display
rich_traceback_install()


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
def inference(model_name: str, control_vector: str):
    """Perform inference with a control vector."""
    # load model
    vector_info = ControlVector.from_file(control_vector)
    model = ControllableModel(
        PreTrainedModel.from_pretrained(vector_info.model_type), layer_ids=[11]
    )
    model.load_control_vector(vector_info)


@click.command()
def shell():
    """Launch an interactive pngr shell with enhanced checks and prompts."""
    console = Console()
    console.print("[bold magenta]Welcome to the pngr interactive shell![/bold magenta]")

    # Check for model selection
    model_name = Prompt.ask("Enter your model name", default="default_model")
    try:
        model = ControllableModel(
            PreTrainedModel.from_pretrained(model_name), layer_ids=[11]
        )
        console.print(f"Loaded model [bold green]{model_name}[/bold green]")
    except FileNotFoundError:
        console.print(
            f"[bold red]Model {model_name} not found, proceeding without loading a model.[/bold red]"
        )
        model = None

    # Check for existing control vectors
    if not path.exists("vector_dataset.jsonl"):
        a_adjective = Prompt.ask("Enter the first adjective for dataset creation")
        b_adjective = Prompt.ask("Enter the second adjective for dataset creation")
        template_path = path.join(
            path.dirname(__file__), "..", "dataset_templates/alphapenger.yaml"
        )
        prompts = create_dataset.create_personality_prompts(
            template_path, a_adjective, b_adjective
        )
        create_dataset.save_prompts(prompts, "vector_dataset.jsonl")
        console.print(
            "[bold green]Dataset created and saved to vector_dataset.jsonl[/bold green]"
        )
    else:
        console.print(
            "[bold yellow]Found existing dataset vector_dataset.jsonl[/bold yellow]"
        )

    # Prepare the namespace with commonly used classes/functions
    namespace = {
        "ControllableModel": ControllableModel,
        "ControlVector": ControlVector,
        "create_dataset": create_dataset,
        "console": console,
        "model": model,
    }

    banner = "Interactive shell is ready."
    banner += "\nType help(<object>) for more information on specific objects."

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
