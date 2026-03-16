"""
Persistent conversation memory — saves/loads gemma3:12b chat history as JSON.
"""
import json
import os
from typing import List, Dict


def load(path: str) -> List[Dict]:
    """Load conversation history from JSON. Returns [] if file missing or corrupt."""
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save(path: str, history: List[Dict]) -> None:
    """Persist conversation history to JSON."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def clear(path: str) -> None:
    """Delete the history file (resets memory)."""
    if os.path.exists(path):
        os.remove(path)
