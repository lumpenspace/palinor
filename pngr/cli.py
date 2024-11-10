from os import path
import click
import code
import sys
<<<<<<< HEAD
from rich.console import Console
from rich.prompt import Prompt
from rich.traceback import install as rich_traceback_install
=======
import torch
import transformers
from huggingface_hub import HfApi
>>>>>>> 80cff54 (working on local queries)

from pngr import create_dataset
from pngr.ControllableModel import ControllableModel
from pngr.ControlVector import ControlVector

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
<<<<<<< HEAD
@click.option("--dataset", "-d", type=click.Path(exists=True))
@click.option("--model-name", "-m", type=str)
def train(model_name: str, dataset: str):
    """Train a control vector."""
    pass


@click.command()
def shell():
    """Launch an interactive pngr shell with enhanced checks and prompts."""
    console = Console()
    console.print("[bold magenta]Welcome to the pngr interactive shell![/bold magenta]")

    # Check for model selection
    model_name = Prompt.ask("Enter your model name", default="default_model")
    try:
        model = ControllableModel(model=model_name, layer_ids=[11])
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

    banner = "Interactive shell is ready. Type help(<object>) for more information on specific objects."
=======
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True)
@click.option("--model-name", "-m", type=str, required=True)
@click.option("--output", "-o", type=str, default="control_vector.pt")
def train(model_name: str, dataset: str, output: str):
    """Train a control vector and start an interactive session with it."""
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    
    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Searching for model {model_name}...")
    api = HfApi()
    
    # Load dataset
    try:
        print(f"Attempting to load {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Direct loading failed, searching for similar models...")
        try:
            models = list(api.list_models(search=model_name, filter="text-generation"))
            if not models:
                print(f"No models found matching '{model_name}'. Using 'distilgpt2' as fallback...")
                model_name = "distilgpt2"
            else:
                model_name = models[0].id
                print(f"Found model: {model_name}")
            
            print(f"Loading model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading models, falling back to 'distilgpt2'...")
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    print(f"Loading dataset from {dataset}...")
    dataset_entries = []
    with open(dataset, 'r') as f:
        for line in f:
            entry = json.loads(line)
            dataset_entries.append({"a": entry['a'], "b": entry['b']})
    
    # Train control vector
    print("Training control vector...")
    control_vector = ControlVector.train(model, tokenizer, dataset_entries)
    # Save control vector
    print(f"Saving control vector to {output}...")
    torch.save(control_vector, output)
    
    # Create controllable model
    print("Creating controllable model...")
    controllable_model = ControllableModel(model, layer_ids=range(-1, -model.config.num_hidden_layers, -1))
    
    # Start interactive shell with the model loaded
    namespace = {
        "model": controllable_model,
        "tokenizer": tokenizer,
        "control_vector": control_vector,
        "generate_text": lambda prompt, coeff=1.0: generate_controlled_text(
            controllable_model, tokenizer, control_vector, prompt, coeff
        )
    }
    
    banner = """
    Control vector trained and loaded! Available objects:
    - model: The controllable model
    - tokenizer: The model's tokenizer
    - control_vector: The trained control vector
    - generate_text(prompt, coeff=1.0): Generate text with control
    
    Example:
    >>> generate_text("Once upon a time", 1.0)
    """
    print("Starting interactive session...")
    while True:
        query = input("\nWould you like to query the model? (Y/N): ").strip().lower()
        if query != 'y':
            break
            
        try:
            coeff = float(input("Enter control coefficient (-1 to 1): "))
            if not -1 <= coeff <= 1:
                print("Coefficient must be between -1 and 1")
                continue
                
            prompt = input("Enter your prompt: ")
            print(f"\nRunning model with control coefficient: {coeff}")
            print("Applying control vectors...")
            generated_text = generate_controlled_text(controllable_model, tokenizer, control_vector, prompt, coeff)
            print("\nGenerated text:")
            print(generated_text)
            
        except ValueError as ve:
            print(f"Invalid input: {ve}")
            print("Please enter a number between -1 and 1")
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Please try again with a different prompt or coefficient")
    
    try:
        code.InteractiveConsole(namespace).interact(banner=banner)
    except KeyboardInterrupt:
        sys.exit(0)
    print("Interactive session ended.")
>>>>>>> 80cff54 (working on local queries)


def generate_controlled_text(model: ControllableModel, 
                           tokenizer: "transformers.PreTrainedTokenizerBase",
                           control_vector: ControlVector,
                           prompt: str,
                           coefficient: float = 1.0) -> str:
    """Generate text using the controlled model."""
    print(f"Applying control vector with coefficient: {coefficient}")
    model.set_control(control_vector, coeff=coefficient)
    inputs = tokenizer(prompt, return_tensors="pt")
    print("Generating text...")
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model.reset()
@click.command()
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True)
@click.option("--model-name", "-m", type=str, required=True)
@click.option("--output", "-o", type=str, default="control_vector.pt")
def train(model_name: str, dataset: str, output: str):
    """Train a control vector and start an interactive session with it."""
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import TRANSFORMERS_CACHE
    import os
    
    # Check if model exists locally
    local_model_path = os.path.join(TRANSFORMERS_CACHE, model_name)
    if os.path.exists(local_model_path):
        print(f"Loading local model from {local_model_path}...")
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        print(f"Local model not found. Downloading {model_name} from Hugging Face Hub...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("Attempting to find similar model...")
            # Try without specific version tag if present
            base_model = model_name.split(':')[0] if ':' in model_name else model_name
            model = AutoModelForCausalLM.from_pretrained(base_model)
            tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load dataset
    print(f"Loading dataset from {dataset}...")
    dataset_entries = []
    with open(dataset, 'r') as f:
        for line in f:
            entry = json.loads(line)
            dataset_entries.append({"a": entry['a'], "b": entry['b']})
    
    # Train control vector
    print("Training control vector...")
    control_vector = ControlVector.train(model, tokenizer, dataset_entries)
    # Save control vector
    print(f"Saving control vector to {output}...")
    torch.save(control_vector, output)
    
    # Create controllable model
    controllable_model = ControllableModel(model, layer_ids=range(-1, -model.config.num_hidden_layers, -1))
    
    # Start interactive shell with the model loaded
    namespace = {
        "model": controllable_model,
        "tokenizer": tokenizer,
        "control_vector": control_vector,
        "generate_text": lambda prompt, coeff=1.0: generate_controlled_text(
            controllable_model, tokenizer, control_vector, prompt, coeff
        )
    }
    
    banner = """
    Control vector trained and loaded! Available objects:
    - model: The controllable model
    - tokenizer: The model's tokenizer
    - control_vector: The trained control vector
    - generate_text(prompt, coeff=1.0): Generate text with control
    
    Example:
    >>> generate_text("Once upon a time", 1.0)
    """
    
    try:
        code.InteractiveConsole(namespace).interact(banner=banner)
    except KeyboardInterrupt:
        sys.exit(0)


pngr.add_command(dataset)
pngr.add_command(train)

if __name__ == "__main__":
    pngr()