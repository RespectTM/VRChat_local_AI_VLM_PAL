"""
Web dashboard — http://localhost:5000
Shows the latest VRChat capture and AI commentary, auto-refreshing every 5 s.
Runs in a background thread; call start() once at startup.
"""
import glob
import json
import os
import threading
from typing import Optional

_state: dict = {
    'scene': '—',
    'thinking': '—',
    'thought': '—',
    'consideration': '—',
    'last_sent': '—',
    'timestamp': '—',
    'capture_dir': 'captures',
    'vision_model': 'vision',
    'think_model': 'think',
    'operator_hint': '',
    'hint_log': [],     # list of {text, ts} dicts — newest first
    'world_map': '',
    'explorer_map': '',   # ASCII art occupancy grid, updated each explore cycle
    'mood_status': '',    # current mood name + strength bar (e.g. CURIOUS ████░░░░░░ 62%)
    'move_queue': [],   # list of (direction, duration) pending execution in main loop
    'move_status': '',  # last move executed (for display)
    'nav_status': '',   # current navigator state (idle / approaching / wandering)
    'chat_log': [],     # list of {name, text, ts} dicts — newest first
    'recognition': {     # name recognition stats (updated by main loop)
        'known_count':   0,
        'unknown_count': 0,
        'resolved_count': 0,
        'success_rate':  0.0,
        'recent': [],    # list of {name, confidence, ts}
        'pending': 0,
    },
}

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VRChat AI PAL</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0f0; padding: 20px; }
    h1 { color: #9c88ff; margin-bottom: 16px; font-size: 1.4em; }
    .row { display: grid; gap: 14px; margin-bottom: 14px; }
    .row-2 { grid-template-columns: 1fr 1fr; }
    .row-3 { grid-template-columns: 1fr 2fr; }
    .row-1 { grid-template-columns: 1fr; }
    @media (max-width: 900px) { .row-2, .row-3 { grid-template-columns: 1fr; } }
    .card { background: #1a1a2e; border-radius: 10px; padding: 14px; }
    .card img { width: 100%; border-radius: 6px; display: block; }
    .label { font-size: 0.72em; color: #7c7caa; text-transform: uppercase;
             letter-spacing: 0.1em; margin-bottom: 8px; }
    .label.combined  { color: #56b6c2; }
    .label.considers { color: #c678dd; }
    .label.thinks    { color: #e5c07b; }
    .label.says      { color: #98c379; }
    .label.guide     { color: #e06c75; }
    .label.worldmap  { color: #61afef; }
    .label.explorermap { color: #56b6c2; }
    .label.movement  { color: #d19a66; }
    .label.chatlog   { color: #e5c07b; }
    .label.recog     { color: #56b6c2; }
    .recog-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0; }
    .recog-stat { background: #0f0f1a; border-radius: 6px; padding: 8px; text-align: center; }
    .recog-stat .rs-val { font-size: 1.4em; font-weight: 700; color: #56b6c2; }
    .recog-stat .rs-lbl { font-size: 0.72em; color: #7c7caa; margin-top: 2px; }
    .recog-recent { margin-top: 8px; max-height: 150px; overflow-y: auto; }
    .recog-item { display: flex; gap: 8px; padding: 4px 0; border-bottom: 1px solid #252540; font-size: 0.84em; }
    .recog-item:last-child { border-bottom: none; }
    .recog-item .ri-name { color: #c678dd; font-weight: 600; flex: 1; }
    .recog-item .ri-conf { color: #56b6c2; white-space: nowrap; }
    .recog-item .ri-ts   { color: #555577; white-space: nowrap; font-size: 0.82em; }
    .chat-entry { display: flex; gap: 8px; padding: 5px 0;
                  border-bottom: 1px solid #252540; font-size: 0.88em; }
    .chat-entry:last-child { border-bottom: none; }
    .chat-entry .chat-name { color: #c678dd; font-weight: 600; min-width: 80px;
                             max-width: 140px; overflow: hidden; text-overflow: ellipsis;
                             white-space: nowrap; flex-shrink: 0; }
    .chat-entry .chat-ts   { color: #555577; font-size: 0.82em; white-space: nowrap;
                             flex-shrink: 0; align-self: center; }
    .chat-entry .chat-text { color: #e0e0f0; line-height: 1.45; }
    #chat_log_panel { max-height: 220px; overflow-y: auto; }
    .text { font-size: 0.93em; line-height: 1.6; white-space: pre-wrap; }
    /* Operator hint */
    .hint-row { display: flex; gap: 8px; margin-top: 8px; }
    .hint-row input { flex: 1; background: #0f0f1a; border: 1px solid #3a3a5c; border-radius: 6px;
                      color: #e0e0f0; font-size: 0.93em; padding: 7px 11px; outline: none; }
    .hint-row input:focus { border-color: #e06c75; }
    .hint-row button { background: #e06c75; color: #fff; border: none; border-radius: 6px;
                       padding: 7px 14px; font-size: 0.88em; cursor: pointer; white-space: nowrap; }
    .hint-row button:hover { background: #c85a63; }
    .hint-row button.clear { background: #3a3a5c; }
    .hint-row button.clear:hover { background: #4a4a7c; }
    .hint-active { color: #e06c75; font-style: italic; font-size: 0.88em; margin-bottom: 6px; }
    /* Hint history */
    .hint-history { margin-top: 10px; max-height: 140px; overflow-y: auto; }
    .hint-history-item { display: flex; gap: 8px; padding: 4px 0;
                         border-bottom: 1px solid #252540; font-size: 0.83em; }
    .hint-history-item:last-child { border-bottom: none; }
    .hint-history-item .h-ts   { color: #555577; white-space: nowrap; flex-shrink: 0; }
    .hint-history-item .h-text { color: #e06c75; opacity: 0.85; }
    /* Movement controls */
    .move-wrap { display: flex; gap: 16px; align-items: flex-start; margin-top: 8px; flex-wrap: wrap; }
    .dpad { display: grid; grid-template-columns: repeat(3, 54px); grid-template-rows: repeat(3, 54px); gap: 5px; }
    .dpad button { background: #252540; border: 1px solid #3a3a5c; border-radius: 8px;
                   color: #e0e0f0; font-size: 1.3em; cursor: pointer;
                   transition: background 0.12s, transform 0.08s; width: 54px; height: 54px; }
    .dpad button:hover  { background: #d19a66; color: #1a1a2e; }
    .dpad button:active { background: #b07840; transform: scale(0.93); }
    .dpad .ph { visibility: hidden; }
    .move-sidebar { display: flex; flex-direction: column; gap: 8px; justify-content: center; }
    .move-sidebar label { font-size: 0.8em; color: #7c7caa; }
    .move-sidebar .dur-label { color: #d19a66; font-size: 0.92em; font-weight: 600; text-align: center; }
    .move-sidebar input[type=range] { width: 100%; accent-color: #d19a66; cursor: pointer; }
    .move-sidebar button { background: #252540; border: 1px solid #3a3a5c; border-radius: 8px;
                           color: #e0e0f0; font-size: 1.15em; padding: 8px 0; cursor: pointer;
                           transition: background 0.12s; }
    .move-sidebar button:hover  { background: #d19a66; color: #1a1a2e; }
    .move-sidebar button:active { background: #b07840; }
    .move-status { font-size: 0.83em; color: #d19a66; margin-top: 8px; }
    .nav-status  { font-size: 0.83em; color: #56b6c2; margin-top: 4px; }
    .mood-status { font-size: 0.88em; color: #c678dd; font-family: 'Cascadia Code','Consolas','Courier New',monospace;
                  margin-top: 6px; letter-spacing: 0.03em; }
    /* Explorer ASCII map */
    .map-pre { font-family: 'Cascadia Code','Consolas','Courier New',monospace;
               font-size: 0.72em; line-height: 1.22; color: #56b6c2;
               white-space: pre; overflow-x: auto; background: #0d0d1a;
               padding: 8px; border-radius: 6px; margin-top: 6px; }
    /* Footer */
    .footer { color: #333355; font-size: 0.75em; margin-top: 14px; }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           background: #00d97e; margin-right: 7px;
           animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }
  </style>
</head>
<body>
  <h1><span class="dot"></span>VRChat AI PAL</h1>

  <!-- Row 1: Operator hint (left) + Movement controls (right) -->
  <div class="row row-2">
    <div class="card">
      <div class="label guide">🎮 Operator Hint — whisper a private instruction (used once)</div>
      <div class="hint-active" id="hint_display" style="display:none"></div>
      <div class="hint-row">
        <input type="text" id="hint_input" placeholder="e.g. compliment their outfit, ask about the world…" maxlength="200">
        <button onclick="sendHint()">Send</button>
        <button class="clear" onclick="clearHint()">Clear</button>
      </div>
      <div class="hint-history" id="hint_log_panel"></div>
    </div>
    <div class="card">
      <div class="label movement">🕹️ Movement Controls</div>
      <div class="move-wrap">
        <div class="dpad">
          <div class="ph"></div>
          <button onclick="sendMove('forward')"   title="Forward">↑</button>
          <div class="ph"></div>
          <button onclick="sendMove('left')"      title="Strafe left">←</button>
          <button onclick="sendMove('backward')"  title="Back">↓</button>
          <button onclick="sendMove('right')"     title="Strafe right">→</button>
          <button onclick="sendMove('turn_left')"  title="Turn left">↶</button>
          <button onclick="sendMove('jump', 0.3)"       title="Jump">⬆</button>
          <button onclick="sendMove('turn_right')" title="Turn right">↷</button>
        </div>
        <div class="move-sidebar">
          <label>Duration</label>
          <span class="dur-label" id="dur_display">1.0 s</span>
          <input type="range" id="move_dur" min="0.05" max="1" step="0.05" value="1"
                 oninput="document.getElementById('dur_display').textContent = parseFloat(this.value).toFixed(2) + ' s'">
        </div>
      </div>
      <div class="move-status" id="move_status"></div>
      <div class="nav-status"  id="nav_status"></div>
      <div class="mood-status" id="mood_status">🧠 Inner state loading…</div>
    </div>
  </div>

  <!-- Row 2: Capture image (left) + Scene description (right) -->
  <div class="row row-2">
    <div class="card">
      <div class="label">Latest Capture</div>
      <img id="img" src="/capture" alt="VRChat capture">
    </div>
    <div class="card">
      <div class="label combined">🔭 Scene — <span id="vision_model">…</span></div>
      <div class="text" id="scene">Waiting for first capture…</div>
    </div>
  </div>

  <!-- Row 3: Considers + Thinks -->
  <div class="row row-2">
    <div class="card">
      <div class="label considers">⚖️ Considers</div>
      <div class="text" id="consideration">—</div>
    </div>
    <div class="card">
      <div class="label thinks">🧠 Thinks — <span id="think_model">…</span></div>
      <div class="text" id="thinking">—</div>
    </div>
  </div>

  <!-- Row 4: Says (full width) -->
  <div class="row row-1">
    <div class="card">
      <div class="label says">💬 Says — last sent to chatbox</div>
      <div class="text" id="last_sent">—</div>
    </div>
  </div>

  <!-- Row 5: World Memory (full width) -->
  <div class="row row-1">
    <div class="card">
      <div class="label worldmap">🗺️ World Memory — accumulates across sessions</div>
      <div class="text" id="world_map" style="font-size:0.82em;opacity:0.85">Gathering observations…</div>
    </div>
  </div>

  <!-- Row 6: Explorer Map (left) + Chat Log (right) -->
  <div class="row row-2">
    <div class="card">
      <div class="label explorermap">🗺 Explorer Map — walkable surface (dead-reckoning)</div>
      <pre id="explorer_map" class="map-pre">Mapping in progress…</pre>
    </div>
    <div class="card">
      <div class="label chatlog">💬 Chat Log — messages heard in VRChat (newest first)</div>
      <div id="chat_log_panel"><span style="color:#555577;font-size:0.88em">No messages yet…</span></div>
    </div>
  </div>

  <!-- Row 7: Recognition stats (full width) -->
  <div class="row row-1">
    <div class="card">
      <div class="label recog">👁 Nametag Recognition — visual name identification engine</div>
      <div class="recog-grid">
        <div class="recog-stat"><div class="rs-val" id="recog_known">0</div><div class="rs-lbl">Known Players</div></div>
        <div class="recog-stat"><div class="rs-val" id="recog_unknown">0</div><div class="rs-lbl">Unresolved</div></div>
        <div class="recog-stat"><div class="rs-val" id="recog_resolved">0</div><div class="rs-lbl">Resolved</div></div>
        <div class="recog-stat"><div class="rs-val" id="recog_rate">0%</div><div class="rs-lbl">Success Rate</div></div>
      </div>
      <div class="recog-recent" id="recog_recent_panel"><span style="color:#555577;font-size:0.88em">No resolutions yet…</span></div>
    </div>
  </div>

  <div class="footer" id="ts">—</div>
  <script>
    function refresh() {
      fetch('/api/state')
        .then(r => r.json())
        .then(d => {
          document.getElementById('scene').textContent         = d.scene;
          document.getElementById('consideration').textContent = d.consideration || '—';
          document.getElementById('thinking').textContent      = d.thinking || '(no <think> block — model replied directly)';
          document.getElementById('last_sent').textContent     = d.last_sent || '—';
          document.getElementById('ts').textContent            = 'Last update: ' + d.timestamp;
          if (d.world_map) document.getElementById('world_map').textContent = d.world_map;
          document.getElementById('img').src = '/capture?t=' + Date.now();
          if (d.vision_model) document.getElementById('vision_model').textContent = d.vision_model;
          if (d.think_model)  document.getElementById('think_model').textContent  = d.think_model;
          const hd = document.getElementById('hint_display');
          if (d.operator_hint) {
            hd.textContent = '⏳ Pending: ' + d.operator_hint;
            hd.style.display = 'block';
          } else {
            hd.style.display = 'none';
          }
          const ms = document.getElementById('move_status');
          if (d.move_status) ms.textContent = d.move_status;
          const ns = document.getElementById('nav_status');
          if (ns && d.nav_status) ns.textContent = d.nav_status;
          const moods = document.getElementById('mood_status');
          if (moods && d.mood_status) moods.textContent = '\ud83e\udde0 ' + d.mood_status;
          // Explorer ASCII map
          const em = document.getElementById('explorer_map');
          if (em && d.explorer_map) em.textContent = d.explorer_map;
          // Hint log — render below input (newest first)
          const hlp = document.getElementById('hint_log_panel');
          if (d.hint_log && d.hint_log.length > 0) {
            hlp.innerHTML = d.hint_log.map(e =>
              '<div class="hint-history-item">'+
              '<span class="h-ts">'  + escHtml(e.ts)   + '</span>'+
              '<span class="h-text">'+ escHtml(e.text) + '</span>'+
              '</div>'
            ).join('');
          } else {
            hlp.innerHTML = '';
          }
          // Recognition stats
          if (d.recognition) {
            const r = d.recognition;
            const ce = (id, v) => { const el = document.getElementById(id); if(el) el.textContent = v; };
            ce('recog_known',    r.known_count   || 0);
            ce('recog_unknown',  r.unknown_count || 0);
            ce('recog_resolved', r.resolved_count || 0);
            ce('recog_rate',     ((r.success_rate || 0) * 100).toFixed(0) + '%');
            const rp = document.getElementById('recog_recent_panel');
            if (rp && r.recent && r.recent.length > 0) {
              function escHtml(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
              rp.innerHTML = r.recent.map(e =>
                '<div class="recog-item">'+
                '<span class="ri-ts">'  + escHtml(e.ts)   + '</span>'+
                '<span class="ri-name">'+ escHtml(e.name) + '</span>'+
                '<span class="ri-conf">'+ (e.confidence*100).toFixed(0)+'%</span>'+
                '</div>'
              ).join('');
            } else if (rp) {
              rp.innerHTML = '<span style="color:#555577;font-size:0.88em">No resolutions yet…</span>';
            }
          }
          // Chat log — render newest-first list (always update to restore after page reload)
          function escHtml(s) {
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                            .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
          }
          const clp = document.getElementById('chat_log_panel');
          if (d.chat_log && d.chat_log.length > 0) {
            clp.innerHTML = d.chat_log.map(e =>
              '<div class="chat-entry">'+
              '<span class="chat-ts">'  + escHtml(e.ts)   + '</span>'+
              '<span class="chat-name">'+ escHtml(e.name) + '</span>'+
              '<span class="chat-text">'+ escHtml(e.text) + '</span>'+
              '</div>'
            ).join('');
          } else {
            clp.innerHTML = '<span style="color:#555577;font-size:0.88em">No messages yet…</span>';
          }
        })
        .catch(() => {});
    }
    function sendHint() {
      const v = document.getElementById('hint_input').value.trim();
      if (!v) return;
      fetch('/api/hint', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({hint: v})})
        .then(() => { document.getElementById('hint_input').value = ''; refresh(); })
        .catch(() => alert('Could not reach PAL dashboard'));
    }
    function clearHint() {
      fetch('/api/hint', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({hint: ''})})
        .then(() => { document.getElementById('hint_input').value = ''; refresh(); })
        .catch(() => {});
    }
    document.getElementById('hint_input').addEventListener('keydown', e => { if (e.key === 'Enter') sendHint(); });
    function sendMove(dir, fixedDur) {
      const dur = fixedDur !== undefined ? fixedDur
                  : parseFloat(document.getElementById('move_dur').value) || 1.0;
      fetch('/api/move', {method:'POST', headers:{'Content-Type':'application/json'},
                          body: JSON.stringify({direction: dir, duration: dur})})
        .then(() => { document.getElementById('move_status').textContent = '⏳ Queued: ' + dir + ' ' + dur + 's'; })
        .catch(() => alert('Could not reach PAL dashboard'));
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


def update(scene: str, thinking: str, thought: str, timestamp: str) -> None:
    """Called from main loop after each AI cycle to push new data to the dashboard."""
    _state['scene']    = scene
    _state['thinking'] = thinking
    _state['thought']  = thought
    _state['timestamp'] = timestamp


def set_consideration(text: str) -> None:
    """Update the Considers panel — call at each decision point in the main loop."""
    _state['consideration'] = text


def set_sent(text: str, timestamp: str) -> None:
    """Update the Says panel — call only when a message is actually dispatched."""
    _state['last_sent'] = text
    _state['timestamp'] = timestamp


def add_chat_entry(name: str, text: str, timestamp: str) -> None:
    """Prepend a chatbox message to the in-memory chat log (newest first, capped at 50)."""
    _state.setdefault('chat_log', []).insert(0, {'name': name, 'text': text, 'ts': timestamp})
    if len(_state['chat_log']) > 50:
        _state['chat_log'] = _state['chat_log'][:50]


def add_hint_entry(text: str, timestamp: str) -> None:
    """Prepend an operator hint to the in-memory hint log (newest first, capped at 100)."""
    _state.setdefault('hint_log', []).insert(0, {'text': text, 'ts': timestamp})
    if len(_state['hint_log']) > 100:
        _state['hint_log'] = _state['hint_log'][:100]


def update_recognition_stats(
    known_count:    int   = 0,
    unknown_count:  int   = 0,
    resolved_count: int   = 0,
    recent:         Optional[list] = None,
    pending:        int   = 0,
) -> None:
    """Update the recognition stats panel from the main loop.

    Call once per think cycle after draining resolved_q.

    Parameters
    ----------
    known_count    : total number of known players in gallery
    unknown_count  : total number of unresolved unknown persons
    resolved_count : total resolved across all sessions
    recent         : list of {'name': str, 'confidence': float, 'ts': str} (newest first)
    pending        : frames currently queued or in retry backoff
    """
    total = known_count + unknown_count
    rate = resolved_count / max(total, 1) if total else 0.0
    _state['recognition'] = {
        'known_count':    known_count,
        'unknown_count':  unknown_count,
        'resolved_count': resolved_count,
        'success_rate':   round(rate, 3),
        'recent':         (recent or [])[:20],
        'pending':        pending,
    }
    # Also expose gallery_dir for the /gallery/img endpoint
    _state.setdefault('gallery_dir', 'snapshots/gallery')


def preload_chat_log(entries: list) -> None:
    """Populate the in-memory chat log from a previously saved list (called at startup)."""
    _state['chat_log'] = list(entries[:50])


def _run_server(host: str, port: int, use_ngrok: bool = False, ngrok_token: str = '',
                vision_model: str = '', think_model: str = '') -> None:
    if vision_model:
        _state['vision_model'] = vision_model
    if think_model:
        _state['think_model'] = think_model
    try:
        from flask import Flask, send_file, jsonify, Response

        import logging as _logging
        _logging.getLogger('werkzeug').setLevel(_logging.ERROR)  # suppress HTTP request logs

        app = Flask(__name__)

        @app.route('/')
        def index():
            return Response(_HTML, mimetype='text/html')

        @app.route('/api/state')
        def state():
            return jsonify(_state)

        @app.route('/api/hint', methods=['POST'])
        def set_hint():
            from flask import request as _req
            import time as _time
            data = _req.get_json(force=True, silent=True) or {}
            hint_text = str(data.get('hint', '')).strip()[:200]
            _state['operator_hint'] = hint_text
            # Persist non-empty hints to hint_log.json and in-memory list
            if hint_text:
                _ts = _time.strftime('%H:%M:%S')
                add_hint_entry(hint_text, _ts)
                _hlog_path = os.path.join(
                    os.path.dirname(_state.get('capture_dir', 'captures')),
                    'memory', 'hint_log.json'
                )
                try:
                    _existing: list = []
                    if os.path.exists(_hlog_path):
                        with open(_hlog_path, 'r', encoding='utf-8') as _hf:
                            _existing = json.load(_hf)
                    _existing.insert(0, {'text': hint_text, 'ts': _ts})
                    if len(_existing) > 100:
                        _existing = _existing[:100]
                    os.makedirs(os.path.dirname(_hlog_path), exist_ok=True)
                    with open(_hlog_path, 'w', encoding='utf-8') as _hf:
                        json.dump(_existing, _hf, indent=2, ensure_ascii=False)
                except Exception:
                    pass
            return jsonify({'ok': True})

        @app.route('/api/move', methods=['POST'])
        def queue_move():
            from flask import request as _req
            data = _req.get_json(force=True, silent=True) or {}
            direction = str(data.get('direction', 'forward')).strip()[:20]
            try:
                duration = max(0.1, min(float(data.get('duration', 1.0)), 10.0))
            except (TypeError, ValueError):
                duration = 1.0
            _state.setdefault('move_queue', []).append((direction, duration))
            return jsonify({'ok': True, 'direction': direction, 'duration': duration})

        @app.route('/capture')
        def capture():
            d = _state['capture_dir']
            files = (glob.glob(os.path.join(d, '*.png'))
                     + glob.glob(os.path.join(d, '*.jpg')))
            if not files:
                return Response('No captures yet', status=404)
            files.sort(key=os.path.getmtime, reverse=True)
            return send_file(os.path.abspath(files[0]), mimetype='image/png')

        @app.route('/gallery')
        def gallery_overview():
            """Return JSON list of all known players from the people gallery."""
            gallery_dir = _state.get('gallery_dir', 'snapshots/gallery')
            index_path  = os.path.join(gallery_dir, 'gallery.json')
            if not os.path.exists(index_path):
                return jsonify({'known': [], 'unknown_count': 0})
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                known = data.get('known', [])
                unknowns = [u for u in data.get('unknown', []) if not u.get('resolved_as')]
                return jsonify({
                    'known': [
                        {
                            'name':          p.get('name', ''),
                            'first_seen':    p.get('first_seen', ''),
                            'last_seen':     p.get('last_seen', ''),
                            'sighting_count':p.get('sighting_count', 0),
                            'crops':         len(p.get('ref_crop_paths', [])),
                        }
                        for p in known
                    ],
                    'unknown_count': len(unknowns),
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @app.route('/gallery/img/<path:name>')
        def gallery_img(name: str):
            """Serve the best reference crop for a known player."""
            gallery_dir = _state.get('gallery_dir', 'snapshots/gallery')
            import re as _re
            safe = _re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)[:60]
            person_dir = os.path.join(gallery_dir, 'known', safe)
            if not os.path.isdir(person_dir):
                return Response('Not found', status=404)
            crops = sorted(
                [f for f in os.listdir(person_dir) if f.endswith('.png')],
                reverse=True,
            )
            if not crops:
                return Response('No crops', status=404)
            return send_file(os.path.join(person_dir, crops[0]), mimetype='image/png')

        if use_ngrok:
            try:
                from pyngrok import ngrok as _ngrok, conf as _conf
                if ngrok_token:
                    _conf.get_default().auth_token = ngrok_token
                tunnel = _ngrok.connect(port, bind_tls=True)
                print(f'[Dashboard] ngrok public URL: {tunnel.public_url}')
            except Exception as e:
                print(f'[Dashboard] ngrok tunnel failed: {e}')

        app.run(host=host, port=port, debug=False, use_reloader=False)
    except ImportError:
        print('[Dashboard] Flask not installed — dashboard disabled.')
    except Exception as e:
        print(f'[Dashboard] failed to start: {e}')


def start(host: str = '127.0.0.1', port: int = 5000,
          capture_dir: str = 'captures', ngrok: bool = False,
          ngrok_token: str = '', vision_model: str = '', think_model: str = '',
          chat_log_file: str = '') -> None:
    """Start the dashboard HTTP server in a background daemon thread.
    If ngrok=True, also open a public ngrok tunnel and print the URL.
    """
    _state['capture_dir'] = capture_dir
    # Synchronously preload chat log from disk so history survives restarts/reloads
    if chat_log_file and os.path.exists(chat_log_file):
        try:
            with open(chat_log_file, 'r', encoding='utf-8') as _f:
                _entries = json.load(_f)
            _state['chat_log'] = list(_entries[:50])
            print(f'[Dashboard] Chat log: {len(_state["chat_log"])} entries preloaded')
        except Exception:
            pass
    hint_log_file = os.path.join(os.path.dirname(chat_log_file), 'hint_log.json') if chat_log_file else ''
    if hint_log_file and os.path.exists(hint_log_file):
        try:
            with open(hint_log_file, 'r', encoding='utf-8') as _f:
                _hentries = json.load(_f)
            _state['hint_log'] = list(_hentries[:100])
            print(f'[Dashboard] Hint log: {len(_state["hint_log"])} entries preloaded')
        except Exception:
            pass
    t = threading.Thread(
        target=_run_server,
        args=(host, port, ngrok, ngrok_token, vision_model, think_model),
        daemon=True,
    )
    t.start()
