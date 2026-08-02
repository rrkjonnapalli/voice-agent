# voice-agent

A local, real-time speech-to-speech voice agent — talk to it, it listens,
thinks, and talks back, and you can even interrupt it mid-sentence like a
real conversation. Built from scratch as a learning project: mic capture,
voice activity detection, speech-to-text, an LLM, and text-to-speech, all
wired together into one live pipeline running locally on Apple Silicon via
MLX.

There's also a networked version — a small FastAPI server you can point a
separate client at, so the "brain" doesn't have to live on the same machine
as the mic and speakers.

For the full technical write-up — architecture, design decisions, gotchas,
and the reasoning behind how things are built — see [ref.md](ref.md).

## Setup

**Prerequisites** — install these yourself, nothing else is needed:
- An Apple Silicon Mac — STT and TTS both run on MLX, which requires it.
- [uv](https://docs.astral.sh/uv/) — everything else runs through it.
- [Ollama](https://ollama.com), running locally with a model pulled (see
  `LLM_MODEL` in `core/constants.py`) — the LLM step talks to it directly.

That's it — no PortAudio, espeak, ffmpeg, or Rust toolchain needed. The
audio libraries (`sounddevice`, `soundfile`) bundle their own compiled
binaries inside the Python wheel, and nothing else in the dependency tree
needs a system-level install.

Python itself is pinned to 3.10 (`.python-version`) — `uv` fetches it
automatically if you don't have it. Then:

```
uv sync
```

This creates the virtual environment and installs everything from `uv.lock`.

## Running it

Locally (mic and speakers on the same machine):
```
uv run pipeline.py
```

Over the network — either a Python client, or a browser at `http://<host>:8000/`
(built-in HTML/JS client, no install needed):
```
uv run uvicorn server:app --host 0.0.0.0 --port 8000
uv run client.py
```

Each connection (client.py, a browser tab, another device) gets its own
independent session — see [ref.md](ref.md) for how.

**Serving over HTTPS** — needed if you want to open the browser client from a
*different device* on your network rather than the same machine, since
`getUserMedia` (mic access) requires a secure context and a plain LAN IP over
HTTP doesn't count. Generate a self-signed cert once (valid for `localhost`
and your current LAN IP; re-run if that IP changes):
```
./gen_cert.sh
```
Then run with it:
```
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile=key.pem --ssl-certfile=cert.pem
```
Each new device will show a one-time "not private" browser warning to click
through (expected — it's self-signed, not from a trusted CA).
> - Advanced > Proceed to unsafe
> - Allow microphone

---

One thing is still stuck: getting acoustic echo cancellation working
reliably, so the agent doesn't hear itself through laptop speakers and start
replying to its own voice. A few real approaches got genuinely close, but
not consistently enough to ship — the full story is in ref.md. If you have
ideas here, contributions are very welcome.
