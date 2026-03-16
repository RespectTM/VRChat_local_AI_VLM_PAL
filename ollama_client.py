import base64
import json
import urllib.request
from typing import Optional, List, Dict, Tuple
from PIL import Image
from io import BytesIO

OLLAMA_URL = 'http://localhost:11434'


def _img_to_b64(path: str) -> str:
    """Load an image file and return a base64-encoded JPEG string."""
    img = Image.open(path).convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def query(model: str, prompt: str,
          image_path: Optional[str] = None,
          timeout: int = 120) -> str:
    """
    Send a prompt (and optional image) to an Ollama model via the HTTP API.
    Returns the full response text.
    """
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
    }
    if image_path:
        payload['images'] = [_img_to_b64(image_path)]

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/generate',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
            return result.get('response', '').strip()
    except urllib.error.URLError as e:
        return f'Ollama connection error: {e}'
    except Exception as e:
        return f'Ollama error: {e}'


# ---------------------------------------------------------------------------
# Stateful chat — uses /api/chat so the model sees conversation history.
# messages format: [{'role': 'system'|'user'|'assistant', 'content': str}, ...]
# Returns (response_text, updated_messages_list).
# ---------------------------------------------------------------------------

def chat(model: str, messages: List[Dict[str, str]],
         timeout: int = 120) -> Tuple[str, List[Dict[str, str]]]:
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/chat',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
            reply = result.get('message', {}).get('content', '').strip()
            updated = messages + [{'role': 'assistant', 'content': reply}]
            return reply, updated
    except urllib.error.URLError as e:
        return f'Ollama connection error: {e}', messages
    except Exception as e:
        return f'Ollama error: {e}', messages


# Legacy helper kept for backwards compat
def run_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    return query(model, prompt, timeout=timeout)


if __name__ == '__main__':
    print(query('moondream:latest', 'Describe this image.', image_path='captures/latest.png'))
