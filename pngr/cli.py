from os import path
import click
import code
# Write the fixed content to the file
with open('/content/pngr/pngr/ControlLayer.py', 'w') as f:
    f.write('''[paste the above code here]''')

# Then import and run your code
import sys
sys.path.append('/content/pngr')

from pngr.cli import train

# Run the training
train(model_name="meta-llama/Llama-2-7b-hf", 
      dataset="vector_dataset.jsonl", 
      output="control_vector.pt")
from rich.console import Console
from rich.prompt import Prompt
from rich.traceback import install as rich_traceback_install
import torch
import transformers
from difflib import get_close_matches
import os
from huggingface_hub import login
from transformers import utils

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
@click.option("--dataset", "-d", type=click.Path(exists=True), required=True)
@click.option("--model-name", "-m", type=str, required=True)
@click.option("--output", "-o", type=str, default="control_vector.pt")
def train(model_name: str, dataset: str, output: str):
    """Train a control vector and start an interactive session with it."""
    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # List of available LLaMA models
    LLAMA_MODELS = [
        # Llama 3.2 Models
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B-Instruct",
        
        # Llama 3.1 Models
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-3.1-70B-Instruct",
        
        # Meta Llama 3 Models
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        
        # Llama 2 Models
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        
        # Code Llama Models
        "meta-llama/CodeLlama-7b-hf",
        "meta-llama/CodeLlama-13b-hf",
        "meta-llama/CodeLlama-34b-hf",
        "meta-llama/CodeLlama-70b-hf"
    ]
    
    print("\nAvailable LLaMA models:")
    for model in LLAMA_MODELS:
        print(f"- {model}")
    
    # Find closest matching LLaMA model
    closest_match = get_close_matches(model_name.lower(), [m.lower() for m in LLAMA_MODELS], n=1)
    if not closest_match:
        print(f"\nNo matching LLaMA model found for '{model_name}'")
        print("Please select from the available models listed above")
        return
    
    selected_model = LLAMA_MODELS[[m.lower() for m in LLAMA_MODELS].index(closest_match[0])]
    print(f"\nSelected model: {selected_model}")
    
    # Confirm with user
    confirm = input("Would you like to proceed with this model? (Y/N): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled")
        return

    # Check for HF token
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("No HuggingFace token found in environment.")
        token = input("Please enter your HuggingFace token (or press enter to try without): ").strip()
        if token:
            login(token)
    
    try:
        print(f"\nLoading model {selected_model}...")
        # Add device and load in 8-bit to reduce memory usage
        model = AutoModelForCausalLM.from_pretrained(
            selected_model,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(selected_model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nThis could be due to:")
        print("1. Insufficient memory (LLaMA models are very large)")
        print("2. Missing HuggingFace access token")
        print("3. Not having accepted the model terms")
        print("\nTrying to load smaller model...")
        try:
            print("Attempting to load Llama-2-7b-hf with reduced precision...")
            model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16,
            )
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        except Exception as e2:
            print(f"\nStill failed to load model: {str(e2)}")
            print("\nPlease ensure you:")
            print("1. Have enough GPU/CPU memory")
            print("2. Have run 'huggingface-cli login' with your token")
            print("3. Have accepted the terms at https://huggingface.co/meta-llama")
            return

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
    print("Creating controllable model...")
    controllable_model = ControllableModel(model, layer_ids=range(-1, -model.config.num_hidden_layers, -1))
    
    # Interactive query loop
    print("\nModel training complete! Starting interactive query session...")
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
            print(f"\nGenerating with control coefficient: {coeff}")
            generated_text = generate_controlled_text(
                controllable_model, 
                tokenizer, 
                control_vector, 
                prompt, 
                coeff
            )
            print("\nGenerated text:")
            print(generated_text)
        
        except ValueError:
            print("Invalid input. Please enter a number between -1 and 1")
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Please try again with a different prompt or coefficient")

    print("Query session ended.")


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
    return generated_text


pngr.add_command(dataset)
pngr.add_command(train)

if __name__ == "__main__":
    pngr()