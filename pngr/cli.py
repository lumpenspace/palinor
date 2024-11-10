from os import path
import click

from . import create_dataset


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
def train(dataset: str):
    """Train a control vector."""
    pass


pngr.add_command(dataset)
if __name__ == "__main__":
    pngr()
