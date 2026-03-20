# VRChat Local AI VLM PAL

A VRChat AI companion that uses a local vision model (GPU) to describe scenes and a remote LLM to generate chatbox responses.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### One-time: set permanent GPU env vars

These must be in Ollama's environment when it starts. Setting them permanently in the Windows user registry means they survive reboots and apply even when Ollama auto-starts from the system tray:

```powershell
[System.Environment]::SetEnvironmentVariable('ROCR_VISIBLE_DEVICES',     '0',       'User')
[System.Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES',      '0',       'User')
[System.Environment]::SetEnvironmentVariable('HSA_OVERRIDE_GFX_VERSION', '12.0.1',  'User')
[System.Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '2',       'User')
[System.Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION',   '1',       'User')
```

Run this **once**, then reboot (or kill the Ollama tray and relaunch it). After that, just:

```powershell
.venv\Scripts\python.exe main.py --no-tts --no-osc
```

> **Why `OLLAMA_MAX_LOADED_MODELS=2`?** PAL runs two models simultaneously — a vision model
> (`minicpm-v:8b`, ~4.3 GB VRAM) and a think model (`qwen2.5:7b`, ~4.6 GB VRAM). Without this
> setting Ollama defaults to 1 and evicts whichever model isn't currently being queried.

## Web Dashboard

The dashboard runs at `http://localhost:5000` by default (configurable in `config.yaml`).

## ngrok (Remote Dashboard Access)

To expose the dashboard over the internet, run ngrok **in a separate terminal** before or after starting PAL.

### 1. Install ngrok

Download from https://ngrok.com/download and add to PATH, or install via `winget`:

```powershell
winget install ngrok.ngrok
```

ngrok installs to `$env:USERPROFILE\AppData\Local\ngrok\ngrok.exe` and is **not** added to PATH automatically.

### 2. Authenticate (one-time)

The auth token is already in `config.yaml`. Run this once:

```powershell
& "$env:USERPROFILE\AppData\Local\ngrok\ngrok.exe" config add-authtoken 2vlP9Rc2d5VFEeLMDyjvw4oTLPW_8fkTkdERxMCj1bhsKEh4
```

### 3. Start the tunnel

In a **separate** PowerShell terminal, point ngrok at the dashboard port:

```powershell
& "$env:USERPROFILE\AppData\Local\ngrok\ngrok.exe" http 5000
```

> **ngrok not on PATH?** The `winget` install puts `ngrok.exe` in `%LOCALAPPDATA%\ngrok\` but doesn't add it to PATH.
> Use the full path above, or add it manually:
> ```powershell
> [System.Environment]::SetEnvironmentVariable('PATH', $env:PATH + ";$env:USERPROFILE\AppData\Local\ngrok", 'User')
> ```
> Then restart your terminal and you can just type `ngrok http 5000`.

ngrok will print a public URL like `https://xxxx.ngrok-free.app` — open that in any browser.

> **Why a separate terminal?**  
> PAL can optionally start ngrok itself (set `ngrok: true` in `config.yaml`), but this only works
> if ngrok is already authenticated and on PATH. Running it independently is more reliable —
> it keeps running even if PAL restarts, and you can see the ngrok console output directly.

### 4. Keep ngrok alive across PAL restarts

Leave the ngrok terminal open. The tunnel URL stays the same until you close it.
If you close ngrok and restart it, you get a new URL (on the free plan).

## Configuration

All settings are in `config.yaml`:

| Key | Description |
|-----|-------------|
| `vision_model` | Ollama vision model — must be a multimodal/vision model (e.g. `minicpm-v:8b`) |
| `think_model` | LLM for personality/responses (e.g. `qwen2.5:7b` for local, `qwen3:8b` for remote) |
| `think_url` | Base URL of the Ollama instance for think model — leave empty `''` to use local |
| `vision_max_size` | Max image dimension sent to vision model (default `448`) |
| `vision_num_ctx` | Vision model context window (default `2048`) |
| `dashboard.port` | Local port for the web dashboard (default `5000`) |
| `dashboard.ngrok` | Set `true` to have PAL launch ngrok automatically |
| `dashboard.ngrok_token` | Your ngrok auth token |
