import base64
import json
import threading
import urllib.request
from typing import Optional, List, Dict, Tuple
from PIL import Image
from io import BytesIO

OLLAMA_URL = 'http://localhost:11434'


def _img_to_b64(path: str, max_size: int = 0) -> str:
    """Load an image file and return a base64-encoded JPEG string.
    If max_size > 0, downscale so the longest edge is at most max_size pixels
    (maintains aspect ratio). Has no effect if the image is already smaller.
    """
    img = Image.open(path).convert('RGB')
    if max_size > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_size:
            scale = max_size / longest
            # Snap to multiples of 28: qwen2.5vl needs dims divisible by 28
            # (14-px patch size × spatial_merge_size=2). Plain int() truncation
            # causes off-by-one on non-round resolutions (e.g. 1366×768 →
            # 447.999... → 447) which triggers GGML_ASSERT in the vision encoder.
            new_w = max(28, round(w * scale / 28) * 28)
            new_h = max(28, round(h * scale / 28) * 28)
            img = img.resize((new_w, new_h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _post(url: str, payload: dict, timeout: int) -> dict:
    """POST JSON to Ollama with a hard wall-clock timeout enforced by a daemon thread.

    urllib's socket timeout is unreliable on Windows for long-running localhost
    connections — Ollama sends HTTP 200 headers immediately, keeping the socket
    'active', so the per-read timeout never fires.  A daemon thread with
    threading.Thread.join(timeout) is the only reliable Windows fix.
    """
    data = json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    result: dict = {}
    error_box: list = []

    def _worker():
        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                result.update(json.load(resp))
        except Exception as exc:
            error_box.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f'timed out after {timeout}s')
    if error_box:
        raise error_box[0]
    return result


def query(model: str, prompt: str,
          image_path: Optional[str] = None,
          image_paths: Optional[List[str]] = None,
          timeout: int = 120,
          max_image_size: int = 0,
          num_ctx: int = 0) -> str:
    """
    Send a prompt (and optional image/images) to an Ollama model via the HTTP API.
    image_paths: list of frame paths (oldest first) — encodes all as a video strip.
    image_path:  single frame path (legacy; image_paths takes priority if both given).
    max_image_size: if > 0, resize image so longest edge <= this before encoding.
    num_ctx: if > 0, override the context window size (important for VLMs —
             qwen2.5vl default is 32k which creates a huge KV cache and is slow).
    Returns the full response text.
    """
    opts: dict = {'num_gpu': 99}
    if num_ctx > 0:
        opts['num_ctx'] = num_ctx
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'keep_alive': -1,          # keep model pinned in VRAM indefinitely
        'options': opts,
    }
    if image_paths:
        payload['images'] = [_img_to_b64(p, max_size=max_image_size) for p in image_paths]
    elif image_path:
        payload['images'] = [_img_to_b64(image_path, max_size=max_image_size)]

    try:
        result = _post(f'{OLLAMA_URL}/api/generate', payload, timeout)
        return result.get('response', '').strip()
    except TimeoutError:
        return 'Ollama error: timed out'
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
         timeout: int = 120, base_url: str = '',
         num_ctx: int = 0, repeat_penalty: float = 0.0) -> Tuple[str, List[Dict[str, str]]]:
    _url = (base_url.rstrip('/') if base_url else OLLAMA_URL)
    opts: dict = {'num_gpu': 99, 'num_ctx': num_ctx if num_ctx > 0 else 3072}
    if repeat_penalty > 0.0:
        opts['repeat_penalty'] = repeat_penalty
        opts['repeat_last_n'] = 256   # look back 256 tokens when penalising repetition
    payload = {
        'model': model,
        'messages': messages,
        'stream': False,
        'keep_alive': -1,          # keep model pinned in VRAM indefinitely
        'options': opts,
    }
    try:
        result = _post(f'{_url}/api/chat', payload, timeout)
        reply = result.get('message', {}).get('content', '').strip()
        updated = messages + [{'role': 'assistant', 'content': reply}]
        return reply, updated
    except TimeoutError:
        return 'Ollama error: timed out', messages
    except urllib.error.URLError as e:
        return f'Ollama connection error: {e}', messages
    except Exception as e:
        return f'Ollama error: {e}', messages


# Legacy helper kept for backwards compat
def run_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    return query(model, prompt, timeout=timeout)


if __name__ == '__main__':
    print(query('moondream:latest', 'Describe this image.', image_path='captures/latest.png'))
