# aihost

Your own home-made, OpenAI-compatible API for open-source models, running on your GPU.

```
your app ──(Bearer sk_local_...)──> aihost (FastAPI :8000) ──> llama-server (llama.cpp :8080) ──> GPU
                                        │
                                        └── SQLite: API keys (hashed) + usage logs
```

- **llama.cpp** (`llama-server`) loads a `.gguf` model and runs it on the GPU (NVIDIA CUDA, Apple Metal, AMD/Vulkan — or CPU if you have no GPU).
- **aihost** sits in front of it and adds API keys, per-key usage/token logging and clean error handling. llama-server stays on `127.0.0.1`; only aihost is exposed.
- Works with any OpenAI client — just change `base_url` and `api_key`.

## 1. Install

```bash
git clone https://github.com/technoadnan/aihost && cd aihost
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install **llama.cpp** with GPU support so `llama-server` is on your `PATH`:

- macOS / Linux: `brew install llama.cpp` (Metal on Mac, CPU/Vulkan elsewhere)
- NVIDIA: download a CUDA build from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), or build it:
  ```bash
  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j
  # binary: build/bin/llama-server  (or set AIHOST_LLAMA_BIN to its path)
  ```

## 2. Get a model

Download any `.gguf` model from Hugging Face into `models/`, e.g. a Q4_K_M quant that fits your VRAM:

| VRAM  | Good starting size |
|-------|--------------------|
| 6–8 GB  | 3B–8B (Q4_K_M) |
| 12–16 GB | 8B–14B (Q4_K_M) |
| 24 GB | up to ~32B (Q4_K_M) |

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models
```

## 3. Run it

```bash
python cli.py doctor                                  # check GPU, llama-server, DB
python cli.py start-model models/qwen2.5-7b-instruct-q4_k_m.gguf   # terminal 1
python cli.py serve                                   # terminal 2  -> http://localhost:8000
python cli.py create my-laptop                        # prints your key ONCE
```

`start-model` options: `--gpu-layers 999` (all on GPU, default; lower it if you run out of VRAM, `0` = CPU only), `--ctx-size 4096`, `--alias my-model`, and `--extra <flag>` (repeatable) for any other llama-server flag.

## 4. Call it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk_local_..." \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk_local_...")
reply = client.chat.completions.create(
    model="local",  # llama-server serves whatever model it loaded
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in reply:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="")
```

## API

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET  | `/health` | no | `{"status": "okay", "model_server": "okay" \| "loading" \| "unavailable"}` |
| GET  | `/v1/models` | key | Model(s) loaded in llama-server |
| POST | `/v1/chat/completions` | key | OpenAI chat completions; supports `stream`, `temperature`, `top_p`, `max_tokens`, `stop` |

Errors: `401` bad/disabled key, `503` model server not running, `504` model server timed out, `502` model server crashed/errored, `4xx` from the model server (e.g. prompt too long) are passed through. Interactive docs at `http://localhost:8000/docs`.

## CLI

| Command | What it does |
|---------|--------------|
| `python cli.py create NAME` | Create an API key (only its SHA-256 hash is stored) |
| `python cli.py list` | List keys and whether they are active |
| `python cli.py disable ID` | Revoke a key |
| `python cli.py usage` | Requests, errors, tokens and avg latency per key |
| `python cli.py start-model MODEL.gguf` | Start llama-server on the GPU |
| `python cli.py serve` | Start the API (`--port`, `--host`, `--reload`) |
| `python cli.py status` | Is the model server up, which model is loaded |
| `python cli.py doctor` | Check everything is set up |

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default |
|----------|---------|
| `AIHOST_DB_PATH` | `aihost.db` |
| `AIHOST_LLAMA_BIN` | `llama-server` |
| `AIHOST_LLAMA_HOST` / `AIHOST_LLAMA_PORT` | `127.0.0.1` / `8080` |
| `AIHOST_LLAMA_URL` | `http://$AIHOST_LLAMA_HOST:$AIHOST_LLAMA_PORT` |
| `AIHOST_REQUEST_TIMEOUT` | `300` seconds |
| `AIHOST_API_HOST` / `AIHOST_API_PORT` | `0.0.0.0` / `8000` |

## Exposing it outside your home network

aihost speaks plain HTTP. To use it from outside, put it behind HTTPS — e.g. [Tailscale](https://tailscale.com) (private, easiest), a Cloudflare Tunnel, or Caddy as a reverse proxy. Don't port-forward plain HTTP: your API key would travel unencrypted.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake model server, so they run without a GPU.
