"""
Moondream Station client — calls the local REST API served by the
Moondream Station desktop app (https://moondream.ai/station).

Default endpoint: http://localhost:2020
API:  POST /v1/query
      Body: { "image_url": "data:image/jpeg;base64,...", "question": "..." }
      Response: { "answer": "..." }
"""
import base64
import json
import urllib.request
import urllib.error
from io import BytesIO

from PIL import Image


STATION_URL = 'http://localhost:2020'


def query(prompt: str, image_path: str,
          url: str = STATION_URL,
          timeout: int = 30) -> str:
    """Send a vision query to Moondream Station and return the answer text."""
    img = Image.open(image_path).convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        'image_url': f'data:image/jpeg;base64,{b64}',
        'question': prompt,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f'{url}/v1/query',
        data=data,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
            return result.get('answer', '').strip()
    except urllib.error.HTTPError as e:
        return f'Moondream Station HTTP {e.code}: {e.reason}'
    except urllib.error.URLError as e:
        return f'Moondream Station connection error: {e.reason}'
    except Exception as e:
        return f'Moondream Station error: {e}'


def health_check(url: str = STATION_URL, timeout: int = 5) -> bool:
    """Return True if Moondream Station is reachable."""
    try:
        urllib.request.urlopen(f'{url}/', timeout=timeout)
        return True
    except Exception:
        # A 404 or 405 still means the server is up
        try:
            urllib.request.urlopen(
                urllib.request.Request(f'{url}/v1/query', method='GET'),
                timeout=timeout,
            )
        except urllib.error.HTTPError:
            return True  # any HTTP error = server is responding
        except Exception:
            return False
    return True
