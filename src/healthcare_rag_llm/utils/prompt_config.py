# src/healthcare_rag_llm/utils/prompt_config.py

from pathlib import Path
from typing import Optional


def load_system_prompt(config_path: Optional[str] = None) -> str:
    """
    Load system prompt from configuration file.

    Args:
        config_path: Path to system prompt text file.
                    If None, uses default path: configs/system_prompt.txt

    Returns:
        System prompt string

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        # Default path relative to project root
        config_path = Path("configs/system_prompt.txt")
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"System prompt configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
