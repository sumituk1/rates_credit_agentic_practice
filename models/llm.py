import os
import yaml
from pathlib import Path
from langchain_ollama import ChatOllama


def _load_settings() -> dict:
    settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(settings_path) as f:
        return yaml.safe_load(f)


def get_llm() -> ChatOllama:
    """Return a ChatOllama instance configured from settings.yaml.

    Requires Ollama running at OLLAMA_BASE_URL (default http://localhost:11434).
    Override the model via config/settings.yaml → llm.model.
    """
    settings = _load_settings()
    cfg = settings["llm"]
    base_url = os.environ.get("OLLAMA_BASE_URL", cfg.get("base_url", "http://localhost:11434"))
    return ChatOllama(
        model=cfg["model"],
        temperature=cfg["temperature"],
        base_url=base_url,
    )