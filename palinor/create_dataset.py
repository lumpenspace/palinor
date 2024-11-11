"""Create a dataset of polar prompts."""

import json
from typing import Any, Sequence
import yaml
from pathlib import Path
from .Message import DatasetEntry, Message


def load_yaml_template(file_path: str) -> list[dict[str, Any]]:
    """Load YAML template file."""
    # Get absolute paths
    package_dir = Path(__file__).parent.resolve()
    project_root = package_dir.parent.resolve()

    # Remove any absolute path from file_path if present
    file_path = Path(file_path).name

    # Try project root first since that's where the file actually is
    template_path = project_root / "dataset_templates" / file_path

    if not template_path.exists():
        # Fallback to package templates
        template_path = package_dir / "templates" / file_path

        if not template_path.exists():
            msg = (
                f"Template file not found. Tried:\n"
                f"- Project templates: {template_path}\n"
                f"- Package templates: {package_dir}/templates/{file_path}"
            )
            raise FileNotFoundError(msg)

    with open(template_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_prompts(prompts: Sequence[DatasetEntry], output_path: str) -> None:
    """Save prompts to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt.to_dict()) + "\n")


def create_personality_prompts(
    template_path: str,
    a_adjective: str, 
    b_adjective: str,
) -> list[DatasetEntry]:
    """Create prompts with proper case formatting."""
    templates = load_yaml_template(template_path)
    prompts = []

    for template in templates:
        user_prompts = template["prompts"]
        
        for user_prompt in user_prompts:
            # Format A in lowercase, B in uppercase
            a_prompt = user_prompt.lower()  # Lowercase for sad/quiet
            b_prompt = user_prompt.upper()  # Uppercase for excited/loud
            
            entry = DatasetEntry(
                a=[Message(role="user", content=a_prompt)],
                b=[Message(role="user", content=b_prompt)],
                a_trait=a_adjective,
                b_trait=b_adjective
            )
            prompts.append(entry)
    return prompts