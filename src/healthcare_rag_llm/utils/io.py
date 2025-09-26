# src/healthcare_rag_llm/utils/io.py
from pathlib import Path
import json
import yaml


def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def dump_json(obj, path: Path) -> None:
    """Write dict/list to JSON file (utf-8, pretty)."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    """Load JSON file (utf-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    """Read UTF-8 text from file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    """Write text to file (UTF-8)."""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def read_yaml(path: str | Path):
    """Read YAML config into dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data) -> None:
    """Write dict to YAML file."""
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
