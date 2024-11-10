"""
Command-line interface for pngr.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.traceback import install as rich_traceback_install

from . import create_dataset
from .manager import PngrManager

# Set up rich error handling
rich_traceback_install()
console = Console()


@click.group()
def pngr():
    """Pngr CLI for controlling language models."""
    pass


@click.command()
@click.argument("a", type=str)
@click.argument("b", type=str)
@click.option("--templates", "-t", type=click.Path(exists=True))
def dataset(templates: str, a: str, b: str) -> None:
    """Create a dataset of personality prompts."""
    template_path = (
        templates or Path(__file__).parent.parent / "dataset_templates/alphapenger.yaml"
    )
    a_adjective = a
    b_adjective = b

    prompts = create_dataset.create_personality_prompts(
        str(template_path), a_adjective, b_adjective
    )
    output_file = "vector_dataset.jsonl"
    create_dataset.save_prompts(prompts, output_file)


@click.command()
@click.argument("name")
@click.argument("a_trait")
@click.argument("b_trait")
@click.option("--model", "-m", help="Model to use", default="facebook/opt-125m")
@click.option("--token", help="HuggingFace token for gated models")
def train(name: str, a_trait: str, b_trait: str, model: str, token: str):
    """Train a new control vector."""
    manager = PngrManager(model, hf_token=token)

    with console.status("Training vector..."):
        vector = manager.train_vector(name, a_trait, b_trait)

    console.print(f"[green]Vector {name} trained and saved![/green]")
    return vector


@click.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Model to use", default="facebook/opt-125m")
@click.option("--vector", "-v", help="Vector to use")
@click.option("--coeff", "-c", help="Control strength", default=1.0, type=float)
@click.option("--token", help="HuggingFace token for gated models")
def generate(prompt: str, model: str, vector: str, coeff: float, token: str):
    """Generate text with optional control."""
    manager = PngrManager(model, hf_token=token)

    output = manager.generate(prompt, vector_name=vector, coeff=coeff)
    console.print(output)


@click.command()
@click.option("--model", "-m", help="Model to use", default="facebook/opt-125m")
def list_vectors(model: str):
    """List available vectors for a model."""
    manager = PngrManager(model)
    vectors = manager.list_vectors()

    if vectors:
        console.print("[bold]Available vectors:[/bold]")
        for vector in vectors:
            console.print(f"  • {vector}")
    else:
        console.print("[yellow]No vectors found for this model[/yellow]")


# Add commands to CLI group
pngr.add_command(train)
pngr.add_command(generate)
pngr.add_command(list_vectors)
pngr.add_command(dataset)

if __name__ == "__main__":
    pngr()
