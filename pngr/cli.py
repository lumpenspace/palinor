"""
Command-line interface for pngr.
"""

from pathlib import Path
import click
from IPython import start_ipython
from rich.console import Console
from rich.traceback import install as rich_traceback_install
from . import create_dataset
from .manager import PngrManager

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
@click.option("--output", "-o", type=str, default="vector_dataset.jsonl")
def dataset(templates: str, a: str, b: str, output: str) -> None:
    """Create a dataset of personality prompts."""
    template_path = (
        templates or Path(__file__).parent.parent / "dataset_templates/alphapenger.yaml"
    )
    prompts = create_dataset.create_personality_prompts(str(template_path), a, b)
    create_dataset.save_prompts(prompts, output)
    console.print(f"[green]Dataset saved to {output}[/green]")


@click.command()
@click.argument("name")
@click.argument("a_trait")
@click.argument("b_trait")
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
@click.option("--token", help="HuggingFace token for gated models")
@click.option("--layer-ids", "-l", help="Layer IDs to control", multiple=True, type=int)
def train(
    name: str,
    a_trait: str,
    b_trait: str,
    model: str,
    token: str,
    layer_ids: tuple[int],
):
    """Train a new control vector."""
    layer_ids_list = list(layer_ids) if layer_ids else [-1, -2, -3]
    manager = PngrManager(model, hf_token=token, layer_ids=layer_ids_list)

    with console.status(f"Training vector '{name}' ({a_trait} vs {b_trait})..."):
        vector = manager.train_vector(name, a_trait, b_trait)

    console.print(f"[green]Vector {name} trained and saved![/green]")
    return vector


@click.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
@click.option("--vector", "-v", help="Vector to use")
@click.option("--coeff", "-c", help="Control strength", default=1.0, type=float)
@click.option("--max-tokens", "-t", help="Max tokens to generate", default=50, type=int)
@click.option("--token", help="HuggingFace token for gated models")
def generate(
    prompt: str, model: str, vector: str, coeff: float, max_tokens: int, token: str
):
    """Generate text with optional control."""
    manager = PngrManager(model, hf_token=token)

    with console.status("Generating..."):
        output = manager.generate(
            prompt, vector_name=vector, coeff=coeff, max_new_tokens=max_tokens
        )

    console.print("\n[bold]Generated text:[/bold]")
    console.print(output)


@click.command()
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
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


@click.command()
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
def shell(model: str):
    """Start a shell after initialising a manager."""
    manager = PngrManager(model_name=model)

    # start interactive python shell with manager available
    banner = f"Pngr shell for model {model} (type 'help' for commands)"
    start_ipython(argv=[], user_ns={"manager": manager}, banner1=banner)


# Add commands to CLI group
pngr.add_command(train)
pngr.add_command(generate)
pngr.add_command(list_vectors)
pngr.add_command(dataset)
pngr.add_command(shell)

if __name__ == "__main__":
    pngr()
