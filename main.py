"""
VRChat Local AI PAL — main loop
Pipeline: capture → moondream2 (vision) → gemma3:12b (thinking)
       → TTS voice + VRChat chatbox (OSC) + web dashboard

Usage:
    python main.py                    # full pipeline (reads config.yaml)
    python main.py --no-osc           # skip chatbox
    python main.py --no-tts           # skip voice
    python main.py --no-dashboard     # skip web dashboard
    python main.py --interval 15      # override capture interval
    python main.py --reset-memory     # wipe saved conversation and start fresh
"""

import time
import os
import argparse
from datetime import datetime

import yaml
from PIL import ImageStat, Image

from capture_vrchat import find_vrchat_window, capture_window_hwnd
from ollama_client import query, chat
import memory as mem
import tts
import dashboard


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = 'config.yaml') -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Config file not found: {path}\n'
            'Make sure config.yaml exists in the project root.'
        )
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Capture (with WGC fallback on black frame)
# ---------------------------------------------------------------------------

def _is_black(img_path: str, threshold: float = 5.0) -> bool:
    """Return True if the image is essentially all-black (minimized / fully occluded)."""
    try:
        img = Image.open(img_path).convert('L')
        return ImageStat.Stat(img).mean[0] < threshold
    except Exception:
        return False


def capture_frame(hwnd: int, img_path: str) -> None:
    """
    Capture via GDI PrintWindow first.
    If the result is a black frame (VRChat minimised / occluded),
    fall back to Windows Graphics Capture (handles minimised windows).
    """
    capture_window_hwnd(hwnd, img_path)
    if _is_black(img_path):
        try:
            from capture_wgc import capture_window_wgc
            capture_window_wgc('VRChat', img_path, timeout=5.0)
            print('  [capture] black frame — used WGC fallback')
        except Exception as e:
            print(f'  [capture] WGC fallback failed: {e}')


# ---------------------------------------------------------------------------
# AI pipeline stages
# ---------------------------------------------------------------------------

def describe_scene(image_path: str, model: str) -> str:
    """Stage 1 — fast image-to-text via moondream2."""
    return query(
        model,
        'Describe what you see in this VRChat scene. '
        'Be specific about people, environment, and any notable details.',
        image_path=image_path,
        timeout=60,
    )


def think(scene: str, history: list, model: str,
          system_prompt: str, max_history: int) -> tuple[str, list]:
    """Stage 2 — gemma3:12b reasons about the scene with full conversation memory."""
    if not history:
        history = [{'role': 'system', 'content': system_prompt}]

    history = history + [{'role': 'user', 'content': f'Current scene: {scene}'}]

    # Keep system prompt + last max_history messages to stay in context window
    if len(history) > max_history + 1:
        history = history[:1] + history[-max_history:]

    reply, history = chat(model, history, timeout=90)
    return reply, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='VRChat Local AI PAL')
    parser.add_argument('--config',         default='config.yaml')
    parser.add_argument('--interval',       type=float, default=None,
                        help='Override capture interval (seconds)')
    parser.add_argument('--no-osc',         action='store_true')
    parser.add_argument('--no-tts',         action='store_true')
    parser.add_argument('--no-dashboard',   action='store_true')
    parser.add_argument('--reset-memory',   action='store_true',
                        help='Clear saved conversation history before starting')
    args = parser.parse_args()

    cfg           = load_config(args.config)
    vision_model  = cfg['vision_model']
    think_model   = cfg['think_model']
    capture_dir   = cfg['capture_dir']
    interval      = args.interval if args.interval is not None else cfg['interval']
    system_prompt = cfg['think_system_prompt'].strip()
    max_history   = cfg['memory']['max_history']
    mem_file      = cfg['memory']['file']

    os.makedirs(capture_dir, exist_ok=True)

    # --- Memory ---
    if args.reset_memory:
        mem.clear(mem_file)
        print('Memory cleared.')
    conversation_history = mem.load(mem_file)
    print(f'Memory: loaded {len(conversation_history)} messages from {mem_file}')

    # --- Dashboard ---
    dash_cfg = cfg.get('dashboard', {})
    if not args.no_dashboard and dash_cfg.get('enabled', False):
        dashboard.start(
            host=dash_cfg.get('host', '127.0.0.1'),
            port=dash_cfg.get('port', 5000),
            capture_dir=capture_dir,
        )
        print(f'Dashboard: http://{dash_cfg.get("host","127.0.0.1")}:{dash_cfg.get("port",5000)}')

    # --- TTS ---
    tts_cfg     = cfg.get('tts', {})
    tts_enabled = (not args.no_tts and tts_cfg.get('enabled', False)
                   and tts.init(tts_cfg.get('rate', 175), tts_cfg.get('volume', 1.0)))
    if tts_enabled:
        print('TTS: enabled (Windows SAPI)')

    # --- OSC ---
    osc     = None
    osc_cfg = cfg.get('osc', {})
    if not args.no_osc and osc_cfg.get('enabled', True):
        try:
            from pythonosc import udp_client
            osc = udp_client.SimpleUDPClient(
                osc_cfg.get('ip', '127.0.0.1'),
                osc_cfg.get('port', 9000),
            )
            print(f'OSC: ready → {osc_cfg.get("ip")}:{osc_cfg.get("port")}')
        except ImportError:
            print('OSC: python-osc not found — chatbox disabled.')

    print(f'Vision  : {vision_model}')
    print(f'Think   : {think_model}')
    print(f'Interval: {interval}s')
    print('-' * 52)

    while True:
        # --- Find VRChat ---
        try:
            hwnd, _ = find_vrchat_window()
        except Exception as e:
            print(f'Window search error: {e}')
            time.sleep(5)
            continue

        if not hwnd:
            print('VRChat not found — waiting…')
            time.sleep(5)
            continue

        ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
        img_path = os.path.join(capture_dir, f'{ts}.png')

        # --- Capture ---
        try:
            capture_frame(hwnd, img_path)
        except Exception as e:
            print(f'[{ts}] Capture error: {e}')
            time.sleep(interval)
            continue
        print(f'[{ts}] Captured')

        # --- Stage 1: Vision ---
        try:
            scene = describe_scene(img_path, vision_model)
            print(f'  [{vision_model}] {scene}')
        except Exception as e:
            print(f'  [{vision_model}] error: {e}')
            time.sleep(interval)
            continue

        # --- Stage 2: Think ---
        try:
            thought, conversation_history = think(
                scene, conversation_history, think_model, system_prompt, max_history
            )
            print(f'  [{think_model}] {thought}')
        except Exception as e:
            print(f'  [{think_model}] error: {e}')
            time.sleep(interval)
            continue

        # --- Save memory ---
        try:
            mem.save(mem_file, conversation_history)
        except Exception as e:
            print(f'  [memory] save error: {e}')

        # --- Dashboard ---
        if not args.no_dashboard and dash_cfg.get('enabled', False):
            dashboard.update(scene, thought, ts)

        # --- TTS ---
        if tts_enabled:
            tts.speak(thought)

        # --- OSC chatbox ---
        if osc:
            try:
                text = thought.strip()[:osc_cfg.get('chatbox_limit', 144)]
                osc.send_message('/chatbox/input', [text, True, False])
                print('  → chatbox ✓')
            except Exception as e:
                print(f'  → chatbox error: {e}')

        time.sleep(interval)


if __name__ == '__main__':
    main()

