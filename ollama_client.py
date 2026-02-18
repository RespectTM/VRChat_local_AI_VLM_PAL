import subprocess
from typing import Optional

def run_ollama(model: str, prompt: str, timeout: Optional[int] = 60) -> str:
    """Run an Ollama model via the `ollama` CLI and return its output.

    Falls back gracefully if `ollama` is not available or the command fails.
    """
    try:
        proc = subprocess.run([
            "ollama", "run", model, prompt
        ], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return "ollama CLI not found. Install Ollama or ensure it's on PATH."
    except subprocess.TimeoutExpired:
        return "ollama call timed out"

    # Some models/CLI versions may write content to stderr; prefer stdout
    out = proc.stdout.strip()
    if out:
        return out
    err = proc.stderr.strip()
    if err:
        return err
    return f"ollama exited with rc={proc.returncode} but produced no output"

if __name__ == '__main__':
    # quick manual test
    print(run_ollama("vrchat-moondream2", "Hello, test from ollama!"))
