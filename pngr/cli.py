import click
from rich.console import Console

console = Console()


@click.group()
def cli():
    """Palinor CLI."""
    pass


# Example command
@cli.command()
def greet():
    """Greet the user."""
    console.print("Hello from Palinor!")


if __name__ == "__main__":
    cli()
