"""
Command-line interface for pngr.
"""

from pathlib import Path
from typing import Optional
import click
from IPython import start_ipython
from rich.console import Console
from rich.traceback import install as rich_traceback_install
from . import create_dataset
from .manager import PngrManager

rich_traceback_install()
console = Console()


class PngrShell:
    """Interactive shell helper for Pngr operations."""
    
    def __init__(self, manager: PngrManager):
        self.manager = manager
        self.datasets = {}  # Store created datasets
    
    def create_dataset(self, name: str, a_trait: str, b_trait: str, 
                      template_path: Optional[str] = None):
        """Create a new dataset of personality prompts."""
        if template_path is None:
            template_path = str(
                Path(__file__).parent.parent / "dataset_templates/alphapenger.yaml"
            )
        
        prompts = create_dataset.create_personality_prompts(
            template_path, a_trait, b_trait
        )
        self.datasets[name] = prompts
        console.print(f"[green]Created dataset '{name}' with {len(prompts)} prompts[/green]")
    
    def list_datasets(self):
        """List all available datasets."""
        if not self.datasets:
            console.print("[yellow]No datasets created yet[/yellow]")
            return
        
        console.print("[bold]Available datasets:[/bold]")
        for name, dataset in self.datasets.items():
            console.print(f"  • {name} ({len(dataset)} prompts)")
    
    def train_vector(self, name: str, dataset_name: str):
        """Train a new control vector using a dataset."""
        if dataset_name not in self.datasets:
            console.print(f"[red]Dataset '{dataset_name}' not found[/red]")
            return
        
        dataset = self.datasets[dataset_name]
        with console.status(f"Training vector '{name}'..."):
            self.manager.train_vector(name, dataset[0].a_trait, dataset[0].b_trait)
        console.print(f"[green]Vector {name} trained and saved![/green]")
    
    def list_vectors(self):
        """List all available vectors."""
        vectors = self.manager.list_vectors()
        if not vectors:
            console.print("[yellow]No vectors available[/yellow]")
            return
        
        console.print("[bold]Available vectors:[/bold]")
        for vector in vectors:
            console.print(f"  • {vector}")
    
    def complete(self, prompt: str, vector: Optional[str] = None, 
                strength: float = 1.0):
        """Complete text with optional vector control."""
        output = self.manager.generate(prompt, vector_name=vector, coeff=strength)
        console.print("\n[bold]Generated text:[/bold]")
        console.print(output)
        return output
    
    def help(self):
        """Show available commands."""
        console.print("[bold]Available commands:[/bold]")
        console.print("  • shell.create_dataset(name, a_trait, b_trait)")
        console.print("  • shell.list_datasets()")
        console.print("  • shell.train_vector(name, dataset_name)")
        console.print("  • shell.list_vectors()")
        console.print("  • shell.complete(prompt, vector=None, strength=1.0)")
        console.print("  • shell.help()")


@click.group()
def pngr():
    """Pngr CLI for controlling language models."""
    pass


@click.command()
@click.argument("name")
@click.argument("a_trait")
@click.argument("b_trait")
@click.option("--templates", "-t", type=click.Path(exists=True))
@click.option("--output", "-o", type=str, default="vector_dataset.jsonl")
def dataset(name: str, a_trait: str, b_trait: str, 
           templates: Optional[str], output: str) -> None:
    """Create a dataset of personality prompts."""
    template_path = (
        templates or Path(__file__).parent.parent / "dataset_templates/alphapenger.yaml"
    )
    prompts = create_dataset.create_personality_prompts(str(template_path), a_trait, b_trait)
    create_dataset.save_prompts(prompts, output)
    console.print(f"[green]Dataset saved to {output}[/green]")


@click.command()
@click.argument("name")
@click.argument("a_trait")
@click.argument("b_trait")
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
@click.option("--token", help="HuggingFace token for gated models")
def train(name: str, a_trait: str, b_trait: str, model: str, token: Optional[str]):
    """Train a new control vector."""
    manager = PngrManager(model, hf_token=token)
    with console.status(f"Training vector '{name}' ({a_trait} vs {b_trait})..."):
        manager.train_vector(name, a_trait, b_trait)
    console.print(f"[green]Vector {name} trained and saved![/green]")


@click.command()
@click.argument("prompt")
@click.option("--model", "-m", help="Model to use", default="meta-llama/Llama-3.2-1B")
@click.option("--vector", "-v", help="Vector to use")
@click.option("--strength", "-s", help="Control strength", default=1.0, type=float)
@click.option("--token", help="HuggingFace token for gated models")
def complete(prompt: str, model: str, vector: Optional[str], 
            strength: float, token: Optional[str]):
    """Generate text with optional vector control."""
    manager = PngrManager(model, hf_token=token)
    output = manager.generate(prompt, vector_name=vector, coeff=strength)
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
    shell_helper = PngrShell(manager)

    banner = (
        f"Pngr shell for model {model}\n"
        "Type shell.help() for available commands"
    )
    start_ipython(
        argv=[],
        user_ns={"manager": manager, "shell": shell_helper},
        banner1=banner
    )


# Add commands to CLI group
pngr.add_command(dataset)
pngr.add_command(train)
pngr.add_command(complete)
pngr.add_command(list_vectors)
pngr.add_command(shell)

if __name__ == "__main__":
    pngr()
