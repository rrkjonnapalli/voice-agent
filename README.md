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

Requires [uv](https://docs.astral.sh/uv/) and Python 3.10 (pinned in
`.python-version` — `uv` will fetch it automatically if you don't have it):

```
uv sync
```

This creates the virtual environment and installs everything from
`uv.lock`. You'll also need [Ollama](https://ollama.com) running locally
with a model pulled (see `LLM_MODEL` in `core/constants.py`) for the LLM
step.

## Running it

Locally (mic and speakers on the same machine):
```
uv run pipeline.py
```

Over the network (a server plus a separate client):
```
uv run uvicorn server:app --host 0.0.0.0 --port 8000
uv run client.py
```

---

One thing is still stuck: getting acoustic echo cancellation working
reliably, so the agent doesn't hear itself through laptop speakers and start
replying to its own voice. A few real approaches got genuinely close, but
not consistently enough to ship — the full story is in ref.md. If you have
ideas here, contributions are very welcome.
