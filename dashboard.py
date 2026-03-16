"""
Web dashboard — http://localhost:5000
Shows the latest VRChat capture and AI commentary, auto-refreshing every 5 s.
Runs in a background thread; call start() once at startup.
"""
import glob
import os
import threading
from typing import Optional

_state: dict = {
    'scene': '—',
    'thought': '—',
    'timestamp': '—',
    'capture_dir': 'captures',
}

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VRChat AI PAL</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0f0; padding: 24px; }
    h1 { color: #9c88ff; margin-bottom: 20px; font-size: 1.4em; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
    .card { background: #1a1a2e; border-radius: 10px; padding: 16px; }
    .card img { width: 100%; border-radius: 6px; display: block; }
    .right { display: flex; flex-direction: column; gap: 16px; }
    .label { font-size: 0.72em; color: #7c7caa; text-transform: uppercase;
             letter-spacing: 0.1em; margin-bottom: 6px; }
    .text { font-size: 0.97em; line-height: 1.55; }
    .footer { color: #333355; font-size: 0.75em; margin-top: 16px; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: #00d97e; margin-right: 7px;
           animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }
  </style>
</head>
<body>
  <h1><span class="dot"></span>VRChat AI PAL</h1>
  <div class="grid">
    <div class="card">
      <div class="label">Latest Capture</div>
      <img id="img" src="/capture" alt="VRChat capture">
    </div>
    <div class="card right">
      <div>
        <div class="label">moondream2 — scene</div>
        <div class="text" id="scene">Waiting for first capture…</div>
      </div>
      <div>
        <div class="label">gemma3:12b — thought</div>
        <div class="text" id="thought">—</div>
      </div>
    </div>
  </div>
  <div class="footer" id="ts">—</div>
  <script>
    function refresh() {
      fetch('/api/state')
        .then(r => r.json())
        .then(d => {
          document.getElementById('scene').textContent   = d.scene;
          document.getElementById('thought').textContent = d.thought;
          document.getElementById('ts').textContent      = 'Last update: ' + d.timestamp;
          document.getElementById('img').src = '/capture?t=' + Date.now();
        })
        .catch(() => {});
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


def update(scene: str, thought: str, timestamp: str) -> None:
    """Called from main loop after each AI cycle to push new data to the dashboard."""
    _state['scene'] = scene
    _state['thought'] = thought
    _state['timestamp'] = timestamp


def _run_server(host: str, port: int) -> None:
    try:
        from flask import Flask, send_file, jsonify, Response

        app = Flask(__name__)

        @app.route('/')
        def index():
            return Response(_HTML, mimetype='text/html')

        @app.route('/api/state')
        def state():
            return jsonify(_state)

        @app.route('/capture')
        def capture():
            d = _state['capture_dir']
            files = (glob.glob(os.path.join(d, '*.png'))
                     + glob.glob(os.path.join(d, '*.jpg')))
            if not files:
                return Response('No captures yet', status=404)
            files.sort(key=os.path.getmtime, reverse=True)
            return send_file(os.path.abspath(files[0]), mimetype='image/png')

        app.run(host=host, port=port, debug=False, use_reloader=False)
    except ImportError:
        print('[Dashboard] Flask not installed — dashboard disabled.')
    except Exception as e:
        print(f'[Dashboard] failed to start: {e}')


def start(host: str = '127.0.0.1', port: int = 5000,
          capture_dir: str = 'captures') -> None:
    """Start the dashboard HTTP server in a background daemon thread."""
    _state['capture_dir'] = capture_dir
    t = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    t.start()
