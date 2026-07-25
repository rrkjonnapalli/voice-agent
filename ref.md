# Speech-to-Speech Voice Agent — Implementation Guide

A local, real-time, interruptible voice agent: microphone in, transcription, LLM
reply, synthesized speech out — plus barge-in support and a network-servable
version. Everything runs locally on Apple Silicon via MLX (no cloud STT/TTS),
except the LLM call, which goes to a local Ollama server.

This is written as a build guide, not a diary — it explains the *why* behind
each design choice and includes the actual working code for the non-obvious
parts, so it can be rebuilt from scratch by someone who hasn't seen the
original process.

## To run

```bash
uv run pipeline.py                                        # local mic/speaker only

uv run uvicorn server:app --host 0.0.0.0 --port 8000       # networked: server...
uv run client.py                                           # ...and a client
```

## Architecture at a glance

```
 mic ──▶ InputStream ──▶ vad_q ──▶ [VAD worker] ──▶ stt_q ──▶ [STT worker]
                                                                    │
                                                             (turn tracker
                                                              bumped here)
                                                                    ▼
 speaker ◀── OutputStream ◀── op_q ◀── [TTS worker] ◀── tts_q ◀── [LLM worker]
```

Four independent pieces connected by `queue.Queue`s, each running on its own
`threading.Thread`. Nothing shares mutable state except the queues and one
shared `TurnTracker` — all bundled into a single `Session` object (§10) that
gets passed to every stage, rather than threading five arguments through
every call site. This shape is what makes barge-in (§9) and network serving
(§12) both possible with almost no extra code per stage.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Package manager | `uv` | pins Python version + deps + lockfile, `uv run` handles the venv automatically |
| VAD | Silero (`silero-vad`, `VADIterator`) | small, fast, well-tuned for 16kHz speech |
| STT | `mlx-community/nemotron-3.5-asr-streaming-0.6b` via `mlx-audio` | accepts `mx.array` directly (no temp files), 60-200ms latency on M-series |
| LLM | Ollama's OpenAI-compatible API, `lfm2.5` | modular — same client code would work against any OpenAI-compatible endpoint |
| TTS | `mlx-community/Kokoro-82M-bf16` via `mlx-audio` | streams audio incrementally per `generate()` call, not one blob |
| Networking | FastAPI + WebSocket | full-duplex, needed since audio flows both directions at once |

---

## 1. Project setup

```bash
uv init
uv python pin 3.10
uv add sounddevice torch silero-vad numpy soundfile openai fastapi "uvicorn[standard]" websockets
uv add "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio.git" "misaki[en]"
```

`uv` replaces pyenv + venv + pip: `pyproject.toml` declares what you want,
`uv.lock` pins exactly what got resolved, `uv run <cmd>` executes inside the
project's environment without manual activation.

Kokoro (TTS) needs `misaki[en]` specifically — the `[en]` extra pulls in the
English G2P (text-to-phoneme) backend; the bare package doesn't include it.

**Project layout**: two packages, `core/` and `stages/`. `stages/` holds only
actual pipeline steps (VAD, STT, LLM, TTS, playback, and their network
equivalents) — nothing in it depends on anything else in it defining shared
config or state. `core/` holds the shared infrastructure those steps
depend on (`constants.py`, `turn.py`, `session.py`) — the dependency arrow
only ever points from `stages/` to `core/`, never the reverse.

---

## 2. Configuration — `core/constants.py`

Every constant used by any stage lives in one file. Stage files never define
config values locally — they only import what they actually use. This keeps
magic numbers from drifting out of sync across files that need to agree on
them (e.g. sample rate has to match between the file that opens the mic
stream and the file that constructs the VAD iterator).

```python
import re

# --- ip.py: mic capture + VAD ---
IP_SAMPLE_RATE = 16000  # mic capture rate, Hz -- required by Silero VAD
IP_CHANNELS = 1
IP_DTYPE = 'float32'
IP_BLOCK_SIZE = 512  # samples per mic callback chunk (32ms @ 16kHz) -- also Silero VAD's required window size
BUFFER_CHUNKS = 10  # rolling pre-roll buffer length, in chunks, prepended when speech starts

# --- op.py: speaker playback ---
OP_SAMPLE_RATE = 24000  # Kokoro's native output rate -- independent of the 16kHz input side
OP_CHANNELS = 1
OP_DTYPE = 'float32'
OP_BLOCK_SIZE = 1024  # samples per speaker callback chunk

# --- stt.py ---
STT_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"

# --- llm.py ---
LLM_BASE_URL = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
LLM_API_KEY = "ollama"  # Ollama ignores the actual value, but the SDK requires one
LLM_MODEL = 'lfm2.5'
LLM_SYSTEM_PROMPT = (
    'You are a smart assistant. Respond briefly and conversationally. ...'
)  # seeds every new session's conversation_history -- see core/session.py
SENTENCE_END_RE = re.compile(r'[.!?\n]')  # marks where a streamed LLM reply can be split off to TTS early

# --- tts.py ---
TTS_MODEL = 'mlx-community/Kokoro-82M-bf16'
```

Two sample rates exist by design, not oversight: the mic/VAD/STT side runs at
16kHz (what Silero and the STT model expect), the speaker/TTS side runs at
24kHz (Kokoro's native vocoder rate). They're independent `sounddevice`
streams, so nothing requires them to match.

---

## 3. Audio fundamentals (concepts)

Before any of the code below makes sense:

- **Sample rate**: how many amplitude measurements per second. 16kHz is
  standard for speech models — much lower than music's 44.1kHz because
  speech doesn't need the higher frequencies.
- **dtype**: hardware usually gives 16-bit ints; ML libraries want 32-bit
  float in `[-1.0, 1.0]`. `sounddevice` can be configured to hand you
  `float32` directly.
- **Channels**: mono (1) for speech, not stereo.
- **Chunking**: a live mic never gives you "all the audio" — it hands you
  small chunks continuously via a callback (e.g. every 32ms). Chunk size is
  a real tradeoff: too small adds overhead/jitter, too large adds latency
  before you can react to anything.
- **Streaming vs. file-based**: recording to a `.wav` and processing after
  is trivial. Continuously capturing while simultaneously processing
  already-captured audio is the actual hard part, and everything below is
  built around that constraint.

`sounddevice` (wraps PortAudio) is the capture/playback library. It's used
via `InputStream`/`OutputStream` with a callback — never a one-shot
`record()` call, since that blocks until done and defeats the whole point of
a live pipeline.

---

## 4. Voice Activity Detection — `stages/ip.py`

**Concept**: the raw Silero model only answers "does this one chunk contain
speech, yes/no?" per call. `VADIterator` wraps it with statefulness — it
remembers whether you're currently "in speech" or "in silence," and only
emits an event when that state flips: `None` (no change), `{'start': ...}`,
or `{'end': ...}`.

Key parameters:
- `sampling_rate` — must match capture rate (16000).
- `min_silence_duration_ms` — debounce. A brief mid-sentence pause shouldn't
  count as "stopped talking"; the iterator waits this many ms of continuous
  silence before firing `'end'`. Too low cuts people off mid-pause; too high
  makes the agent feel slow to respond.
- `speech_pad_ms` — pads the *reported* segment boundary slightly, since VAD
  tends to clip the very start of speech. This does **not** recover audio
  that already passed through un-captured — see the pre-roll buffer below.
- Silero's model is specifically tuned around exactly **512-sample chunks at
  16kHz** (32ms) — it won't hard-crash on other sizes, but accuracy is only
  well-defined at that size.

**Gotcha**: `sounddevice` gives NumPy arrays; Silero (PyTorch) wants a
`torch.Tensor`, flattened to 1D (capture gives you `(frames, channels)`,
VAD wants a flat sequence).

**Pre-roll buffer**: `speech_pad_ms` doesn't retroactively recover audio —
by the time `'start'` fires, earlier chunks already passed through the
callback without being enqueued. Fix: keep a small rolling buffer of the
last N chunks *unconditionally*, every callback. When `'start'` fires, seed
the utterance with whatever's already in that buffer.

**Important**: unlike STT/TTS's models (see §5, §7), the VAD model is **not**
a shared global here — each `Session` loads its own (§10). Two `VADIterator`
instances wrapping one shared `vad_model`, driven concurrently from separate
threads, crash the process (confirmed empirically, see §10) — Silero's model
likely keeps recurrent hidden state inside itself, not just in the iterator
wrapper. The model is small, so duplicating it per session is cheap.

```python
import collections

import numpy as np
import torch

from core.constants import BUFFER_CHUNKS


def make_input_callback(session):
    # uses: session.vad_q (write)
    vad_q = session.vad_q

    # sounddevice callbacks have a fixed signature and can't take extra
    # arguments, so the queue is captured via closure instead
    def input_callback(indata, frames, time, status):
        if status:
            print(status)
        vad_q.put(indata.copy())
    return input_callback


def vad_worker(session):
    # uses: session.vad_q (read), session.vad_iterator (per-session VAD
    # state), session.stt_q (write)
    vad_q = session.vad_q
    stt_q = session.stt_q
    vad_iterator = session.vad_iterator

    started = False
    utterance_chunks = []
    rolling_buffer = collections.deque(maxlen=BUFFER_CHUNKS)

    while True:
        chunk = vad_q.get()
        if chunk is None:  # shutdown sentinel
            stt_q.put(None)
            return

        audio_tensor = torch.from_numpy(chunk.flatten()).float()
        is_speech = vad_iterator(audio_tensor)

        if is_speech is None:
            pass
        elif 'start' in is_speech:
            utterance_chunks = list(rolling_buffer)  # seed with pre-roll
            started = True
        elif 'end' in is_speech:
            utterance_chunks.append(chunk)
            stt_q.put(np.concatenate(utterance_chunks, axis=0).flatten())
            started = False

        if started:
            utterance_chunks.append(chunk)
        rolling_buffer.append(chunk)
```

Model inference/heavy work never happens inside the audio callback itself —
the callback only copies data into a queue; a separate thread (`vad_worker`)
consumes it. This rule holds for every stage: the real-time audio callbacks
do the absolute minimum, everything else happens off-thread.

---

## 5. Speech-to-Text — `stages/stt.py`

The model accepts an `mx.array` directly (`model.generate(mx_array,
language='en-US')`) — no temp files needed for live audio, confirmed by
checking `help(model.generate)` rather than assuming file-path-only.

`stt_model` is a **shared global**, loaded once, reused across every
session's `stt_worker` thread — confirmed safe for concurrent calls from
multiple threads at once (see §10's concurrency testing).

```python
import time

import mlx.core as mx
from mlx_audio.stt import load

from core.constants import STT_MODEL

stt_model = load(STT_MODEL)  # shared across sessions -- concurrent calls verified safe


def stt_worker(session):
    # uses: session.stt_q (read), session.llm_q (write), session.turn_tracker (bump)
    stt_q = session.stt_q
    llm_q = session.llm_q
    turn_tracker = session.turn_tracker

    while True:
        audio_chunk = stt_q.get()
        if audio_chunk is None:
            llm_q.put(None)
            return

        result = stt_model.generate(mx.array(audio_chunk), language='en-US')

        if result.text.strip():
            # only now, once we know this is real speech (not a cough, a
            # click, noise VAD mistook for speech), does it count as an
            # actual conversational turn worth possibly interrupting for
            turn_id = turn_tracker.bump()
            llm_q.put((turn_id, result.text))
```

This is also where the turn counter gets bumped — see §9 for why it's here
and not in the VAD stage.

**Known weakness, accepted for now**: short, isolated, context-free
utterances (single words) transcribe noticeably worse than natural sentences
— a known general ASR limitation (no surrounding context to lean on),
likely worse for a smaller ~600M-param model. Natural conversational speech
tests significantly better. Not fixed — model swap and fine-tuning were both
considered and explicitly deferred.

---

## 6. LLM response generation — `stages/llm.py`

**Concept**: the LLM has no memory between calls — you resend the whole
`conversation_history` (`[{"role", "content"}, ...]`) every turn. Streaming
matters because it's what lets TTS start speaking before the full reply
finishes generating. `conversation_history` lives on the `Session` (§10),
not as a local variable here — same lifetime either way (one per session),
but now inspectable/loggable from outside the worker too.

**Why Ollama's `/v1/chat/completions` and not the Responses API**: Ollama's
`/v1/responses` endpoint exists but is non-stateful (no `previous_response_id`
chaining), so it can't maintain multi-turn state on its own —
`chat.completions.create(..., messages=conversation_history)` with manually
managed history is the correct choice here.

**Sentence-boundary chunking** is the key non-obvious piece: don't feed TTS
one streamed token at a time (no prosody context) and don't wait for the
entire reply (kills latency). Instead, accumulate streamed tokens until a
sentence boundary appears, ship that sentence to TTS immediately, and keep
accumulating the next one while it plays.

```python
import time

from openai import OpenAI

from core.constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, SENTENCE_END_RE

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)  # shared across sessions -- lightweight HTTP client, no per-session state


def llm_worker(session):
    # uses: session.llm_q (read), session.tts_q (write), session.turn_tracker
    # (current), session.conversation_history (read + append, mutated in
    # place -- same list object for the life of the session)
    llm_q = session.llm_q
    tts_q = session.tts_q
    turn_tracker = session.turn_tracker
    conversation_history = session.conversation_history

    while True:
        item = llm_q.get()
        if item is None:
            tts_q.put(None)
            return
        my_turn, text = item
        conversation_history.append({'role': 'user', 'content': text})

        if my_turn != turn_tracker.current():
            continue  # superseded before generation even started

        stream = client.chat.completions.create(
            model=LLM_MODEL, stream=True, messages=conversation_history,
        )

        response_text = ""
        sentence_buffer = ""
        interrupted = False

        for chunk in stream:
            if my_turn != turn_tracker.current():
                interrupted = True
                stream.close()  # see note below -- breaking the loop alone isn't enough
                break

            piece = chunk.choices[0].delta.content or ""
            if not piece:
                continue
            response_text += piece
            sentence_buffer += piece

            while True:
                match = SENTENCE_END_RE.search(sentence_buffer)
                if not match:
                    break
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx].strip()
                sentence_buffer = sentence_buffer[end_idx:]
                if sentence:
                    tts_q.put((my_turn, sentence))

        if interrupted:
            partial = response_text.strip()
            marker = "[interrupted by user before finishing]"
            conversation_history.append({
                'role': 'assistant',
                'content': f"{partial} {marker}" if partial else marker,
            })
            continue

        if sentence_buffer.strip():  # trailing fragment, no terminating punctuation
            tts_q.put((my_turn, sentence_buffer.strip()))
        conversation_history.append({'role': 'assistant', 'content': response_text})
```

**Gotcha — closing the stream on interruption**: breaking out of `for chunk
in stream:` early does **not** close the underlying HTTP connection. Checked
the `openai` SDK source directly (`openai/_streaming.py`): `Stream.close()`
exists and is only called automatically if the stream is read to completion
or used as a context manager. Without an explicit `stream.close()`, Ollama
keeps generating tokens into an unread connection for a beat after
"interruption." Call `.close()` explicitly the moment you detect staleness.

**Why interrupted replies are recorded, not discarded**: if a cut-off reply
is simply dropped, the model sees two `user` messages back-to-back with
nothing from the assistant in between — ambiguous, and undermines a system
prompt instruction like "stop on interruptions" since there's no evidence in
the transcript for the model to react to. The hybrid fix: keep whatever text
was actually generated, append an explicit `"[interrupted by user before
finishing]"` marker.

**Known limitation, not a code bug**: verified via a controlled test
(feed `llm_q` directly, bypassing VAD/STT) that `conversation_history` grows
and is sent correctly every call — a simple "remember X, what is X?" probe
recalls correctly. But on longer, more complex histories, `lfm2.5`
sometimes gives a canned "I don't have access to conversation history"
disclaimer, or partially hallucinates a summary of what was said, despite
the real history being right there in its context. Model behavior, not a
history-management bug — `lfm2.5` was chosen for latency on this hardware,
not conversational quality, and this is the tradeoff that predicts.

---

## 7. Text-to-Speech — `stages/tts.py`

Same shape as STT, mirrored: text in, streamed audio out. `model.generate()`
yields an iterable of chunks (`result.audio`), genuinely incremental.
`model` is a shared global here too, same reasoning and same verified-safe
concurrency as `stt_model`.

```python
import numpy as np
from mlx_audio.tts.utils import load_model

from core.constants import TTS_MODEL

model = load_model(TTS_MODEL)  # shared across sessions -- concurrent calls verified safe


def tts_worker(session):
    # uses: session.tts_q (read), session.op_q (write), session.turn_tracker (current)
    tts_q = session.tts_q
    op_q = session.op_q
    turn_tracker = session.turn_tracker

    while True:
        item = tts_q.get()
        if item is None:
            return
        turn_id, sentence = item

        if turn_id != turn_tracker.current():
            continue  # don't spend model time synthesizing speech nobody will hear

        for result in model.generate(sentence, voice=None, lang_code="a"):
            op_q.put((turn_id, np.array(result.audio)))
```

**Gotchas**:
- Voice names follow `[language][gender]_[name]` (e.g. `af_heart`) — this is
  Kokoro's own convention (`VOICES.md`), not portable from another model's
  example voice names.
- Output is fixed at **24kHz**, set by the model's trained vocoder, not a
  runtime parameter — hence the separate `OP_SAMPLE_RATE` constant and a
  fully independent `OutputStream`.
- `sd.play()` without a following `sd.wait()` produces no audible sound —
  `play()` is non-blocking, and the process exits before playback finishes.
  (Not relevant to the streaming approach above, but a real trap when
  testing TTS output in isolation.)

---

## 8. Speaker playback & the leftover-buffer pattern — `stages/op.py`

**The problem**: TTS produces audio chunks of arbitrary length. `sounddevice`'s
`OutputStream` callback demands *exactly* `frames` samples on every call, no
more, no less. A naive "one queue item per callback" breaks immediately —
chunk length almost never equals the requested frame count.

**The fix**: a persistent buffer (`leftover`) that tops itself up from the
queue only as needed to reach `frames` samples, hands out exactly `frames`
each call, and keeps the remainder for next time. Pads with silence on
genuine underrun (nothing left to play).

```python
import queue

import numpy as np

from core.constants import OP_DTYPE


def make_output_callback(session):
    # uses: session.op_q (read), session.turn_tracker (current)
    op_q = session.op_q
    turn_tracker = session.turn_tracker

    leftover = np.array([], dtype=OP_DTYPE)
    leftover_turn = None

    def output_callback(output_data, frames, time, status):
        nonlocal leftover, leftover_turn
        current = turn_tracker.current()

        # audio already in the buffer was valid when queued, but the turn
        # may have moved on while it waited to play -- drop it immediately
        if leftover_turn is not None and leftover_turn != current:
            leftover = np.array([], dtype=OP_DTYPE)
            leftover_turn = None

        while len(leftover) < frames:
            try:
                turn_id, chunk = op_q.get_nowait()
            except queue.Empty:
                break
            if turn_id != current:
                continue  # stale -- discard without ever buffering it
            leftover_turn = turn_id
            leftover = np.concatenate((leftover, chunk), axis=0)

        if len(leftover) >= frames:
            output_data[:, 0] = leftover[:frames]
            leftover = leftover[frames:]
        else:
            output_data[:len(leftover), 0] = leftover
            output_data[len(leftover):, 0] = 0  # underrun -- pad with silence
            leftover = np.array([], dtype=OP_DTYPE)
            leftover_turn = None

    return output_callback
```

Note `output_data[:, 0] = ...`, not `output_data[:] = ...` — `output_data`'s
shape is `(frames, channels)`; assigning a 1D array directly doesn't
broadcast correctly and raises a shape mismatch.

---

## 9. Interruption / barge-in — `core/turn.py`

**Rejected design**: a shared boolean `is_speaking` flag. Breaks down across
threads — races over who clears it and when, especially with a second
interruption arriving while the first is still being handled.

**Design used**: a monotonically increasing turn counter, thread-safe via a
lock. Every piece of data flowing downstream gets tagged with the turn id it
was produced under. Every stage that receives tagged data compares its tag
against `current()` before acting — a mismatch means a newer turn has
superseded it, so the data is discarded. No flag-clearing races, since the
counter only ever increases.

```python
import threading


class TurnTracker:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def bump(self):
        with self._lock:
            self._value += 1
            return self._value

    def current(self):
        with self._lock:
            return self._value
```

Every queue's payload is a `(turn_id, payload)` tuple from `stt_q` onward
(the `None` shutdown sentinel is checked *before* attempting to unpack).

**Where each stage checks, and why the location matters**:

| Stage | Check | Reasoning |
|---|---|---|
| `stt_worker` | Only place that calls `bump()` — but only when transcribed text is non-empty | See below — moved here from VAD deliberately |
| `stt.py` → `llm_q` | No staleness check | Real user speech is always passed through; losing part of what the user said is worse than the alternative |
| `llm_worker` | Twice: before starting generation, and on **every streamed chunk** | The per-chunk check is what actually cuts a reply off mid-sentence, not just skips it before starting |
| `tts_worker` | Before calling `generate()` | Don't spend model time synthesizing speech nobody will hear |
| `output_callback` | On **every callback invocation** — both the queue and the already-buffered `leftover` | Buffered audio was valid *when queued*, but the turn can move on while it's still waiting to play — must re-check every callback or stale audio plays out to completion instead of stopping immediately |

**Why `bump()` lives in STT, not VAD**: the first version bumped on every
VAD `'start'` event — fastest possible response, but any noise/cough
crossing the VAD threshold would falsely interrupt the agent. Moved to
`stt_worker`, firing only once transcribed text confirms real speech (not
full-sentence completeness — a single word like "stop" still counts).
Tradeoff accepted deliberately: since STT only runs after VAD's `'end'`
(which itself waits out `min_silence_duration_ms`) plus transcription time,
the interruption no longer lands the instant speech starts — there's a real
delay before it takes effect, traded for never falsely interrupting on
non-speech noise.

**Over the network, this local mechanism alone isn't enough** — see §12.

---

## 10. Bundling it into a `Session` — `core/session.py`

**Why this exists**: every piece of per-conversation state (the 5 queues,
`TurnTracker`, `conversation_history`, VAD's model+iterator) used to be
separate local variables/arguments threaded through `pipeline.py`/`server.py`
by hand. That gets unworkable the moment you want more than one concurrent
conversation (multiple sessions each need their own copy of all of it).
`Session` bundles all of it into one object; every stage function takes
`session` as its single argument and unpacks exactly what it uses as the
first lines of the function (see §4-§8) — so starting a new conversation is
just `Session()` + `.start()`, not five queues and four `threading.Thread(...)`
calls written out by hand at every call site.

**What's per-session vs. shared globally, and why** — this was verified
empirically, not assumed. A diagnostic script (`test_model_concurrency.py`)
fired multiple threads at each shared model instance simultaneously (using a
`threading.Barrier` to force genuine overlap, not just close timing), each
thread with different, known input, checking whether each thread's output
matched *its own* input:

- `stt_model` and TTS's `model` — **safe to share** as module-level globals
  across every session. All concurrent calls came back correct, no
  cross-contamination, across multiple rounds.
- VAD's model — **not safe to share**. Two `VADIterator` instances wrapping
  one shared `vad_model`, driven concurrently from separate threads, crashed
  the process outright (a native signal, not a catchable Python exception).
  Likely cause: Silero's model keeps recurrent hidden state inside itself,
  not just in the iterator wrapper, so concurrent forward passes corrupt
  each other. Fix: each `Session` loads its own `vad_model`/`VADIterator` —
  cheap, since Silero's VAD model is small (unlike STT/TTS).

```python
import queue
import threading

from silero_vad import VADIterator, load_silero_vad

from core.constants import IP_SAMPLE_RATE, LLM_SYSTEM_PROMPT
from core.turn import TurnTracker
from stages.ip import vad_worker
from stages.llm import llm_worker
from stages.stt import stt_worker
from stages.tts import tts_worker


class Session:
    def __init__(self):
        self.vad_q = queue.Queue()
        self.stt_q = queue.Queue()
        self.llm_q = queue.Queue()
        self.tts_q = queue.Queue()
        self.op_q = queue.Queue()

        self.turn_tracker = TurnTracker()
        self.conversation_history = [
            {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
        ]

        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model, sampling_rate=IP_SAMPLE_RATE,
            min_silence_duration_ms=200, speech_pad_ms=50, threshold=0.5,
        )

        self._threads = []

    def start(self, daemon=False):
        # daemon=False (pipeline.py) expects a clean stop() call. daemon=True
        # (server.py, for now) means threads just die with the process --
        # there's no per-connection stop() being called yet (see §12).
        self._threads = [
            threading.Thread(target=vad_worker, args=(self,), daemon=daemon),
            threading.Thread(target=stt_worker, args=(self,), daemon=daemon),
            threading.Thread(target=llm_worker, args=(self,), daemon=daemon),
            threading.Thread(target=tts_worker, args=(self,), daemon=daemon),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        # cascading shutdown, same pattern as before: one sentinel into the
        # front queue, each stage forwards its own None downstream, then
        # join every thread
        self.vad_q.put(None)
        for t in self._threads:
            t.join()
        self._threads = []
        self.conversation_history = []

        # defensive: release the per-session VAD model/iterator explicitly
        # rather than relying solely on this Session object being garbage
        # collected -- protects against a stray reference elsewhere (e.g. a
        # forgotten entry in a session registry) silently keeping a whole
        # model pinned in memory
        self.vad_iterator = None
        self.vad_model = None
```

`pipeline.py` collapses down to:

```python
import sounddevice as sd

from core.constants import (
    IP_BLOCK_SIZE, IP_CHANNELS, IP_DTYPE, IP_SAMPLE_RATE,
    OP_BLOCK_SIZE, OP_CHANNELS, OP_DTYPE, OP_SAMPLE_RATE,
)
from core.session import Session
from stages.ip import make_input_callback
from stages.op import make_output_callback

session = Session()
session.start()

with sd.InputStream(
    samplerate=IP_SAMPLE_RATE, channels=IP_CHANNELS, dtype=IP_DTYPE,
    blocksize=IP_BLOCK_SIZE, callback=make_input_callback(session),
), sd.OutputStream(
    samplerate=OP_SAMPLE_RATE, channels=OP_CHANNELS, dtype=OP_DTYPE,
    blocksize=OP_BLOCK_SIZE, callback=make_output_callback(session),
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        pass

session.stop()
```

**Known bug, deferred**: `sd.sleep()` is a blocking call into the PortAudio C
library that doesn't check for Python signals while blocked — a real Ctrl+C
(SIGINT) doesn't reliably interrupt it, confirmed by testing in isolation
(decoupled from any Session/queue logic). `except KeyboardInterrupt` — and
therefore `session.stop()` — may never actually fire. Pre-existing since the
original pipeline rebuild; every earlier smoke test in this project used
`timeout` (SIGTERM, a different, uncaught signal) rather than a real
interrupt, so it went unnoticed until specifically tested. Fix would be
replacing the `sd.sleep(1000)` loop with something that checks more often
(e.g. `time.sleep(0.1)` in a loop) — left as-is for now.

---

## 11. Wiring it up locally vs. checking your work

Callback functions (`input_callback`, `output_callback`) have a fixed
signature imposed by `sounddevice` and can't take extra arguments directly —
that's why they're built via factory functions (`make_input_callback(session)`)
that close over whatever state they need. Worker thread functions, by
contrast, just take `session` as a plain argument since there's no such
constraint on `threading.Thread(target=...)`.

**Verifying claims empirically, not by code review alone**, has been a
recurring theme throughout — several diagnostic scripts sit in the project
root, kept around rather than deleted since they document real, verified
findings a rebuild would want to know:
- `test_model_concurrency.py` — the shared-vs-per-session model findings from §10.
- `test_conversation_history.py` — confirms `conversation_history` grows and
  is used correctly across turns (isolates the code from model behavior,
  relevant to §6's "known limitation" note).
- `test_interrupt_signal.py`, `test_playback_flush.py` — confirm the
  network interruption fix in §12 actually works, using fake send/receive
  callables instead of a real socket, so the exact logic under test is
  isolated from any real network flakiness.

---

## 12. Serving over the network — FastAPI + WebSocket

**Key insight that shaped this**: none of the four worker functions
(`vad_worker`, `stt_worker`, `llm_worker`, `tts_worker`) ever touch
`sounddevice` directly — only the callback factories and `pipeline.py`'s
`with sd.InputStream(...), sd.OutputStream(...):` block do. So serving over
a network requires **zero changes** to those workers or to `core/turn.py` —
only new "edges" that replace the sounddevice-specific input/output.

**Scope decisions**: audio capture/playback moves to the *client* — the
server does no direct hardware I/O, just processes bytes over the
connection. Single session by design (no multi-session routing built yet —
see the note at the end of this section), but structured so that wouldn't
require touching the worker functions — per-connection state lives in
`Session`, not a module global.

**`stages/server_ip.py`** — buffers incoming network bytes into fixed
`IP_BLOCK_SIZE` chunks (WebSocket messages don't arrive pre-chunked to match
what Silero VAD expects) before pushing to `session.vad_q`:

```python
import numpy as np

from core.constants import IP_BLOCK_SIZE, IP_DTYPE


class AudioReceiver:
    """One instance per connection -- its own leftover buffer, no shared
    global state. One of these gets created per Session."""

    def __init__(self, session):
        self.vad_q = session.vad_q
        self.leftover = np.array([], dtype=IP_DTYPE)

    def feed(self, raw_bytes):
        incoming = np.frombuffer(raw_bytes, dtype=IP_DTYPE)
        self.leftover = np.concatenate((self.leftover, incoming))

        while len(self.leftover) >= IP_BLOCK_SIZE:
            chunk = self.leftover[:IP_BLOCK_SIZE]
            self.leftover = self.leftover[IP_BLOCK_SIZE:]
            self.vad_q.put(chunk)
```

**`stages/server_op.py`** — drains `session.op_q`, same staleness check as
the local output callback, but with one important addition beyond a direct
port of the local logic:

**The problem this solves**: locally, `op.py`'s callback re-checks
`turn_tracker` on every audio callback (dozens of times/sec), so it can drop
already-buffered audio mid-playback. Over the network, that's not enough —
once bytes are sent over the WebSocket, they can't be unsent. The client
*receiving* those bytes has no turn-awareness of its own, so audio already
in flight when an interruption happens would just play out normally,
ignoring the interruption. (This was a real bug, found and fixed after the
rest of this architecture was already believed complete and tested.)

**The fix**: watch `turn_tracker` every loop iteration, and the moment the
turn changes, send an explicit control message before continuing — a "stop
playing whatever you have buffered, right now" signal, sent as a JSON text
frame, distinct from the binary audio frames. That's what lets the client
stay simple (no turn-tracking of its own): it just reacts to one event by
flushing its playback buffer.

```python
import asyncio
import queue


async def stream_audio_out(session, send_bytes, send_control):
    op_q = session.op_q
    turn_tracker = session.turn_tracker
    last_known_turn = turn_tracker.current()

    while True:
        current = turn_tracker.current()
        if current != last_known_turn:
            await send_control()
            last_known_turn = current

        try:
            turn_id, chunk = op_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if turn_id != current:
            continue  # stale, discard without sending
        await send_bytes(chunk.tobytes())
```

**`server.py`** — one module-level `Session`, threads started with
`daemon=True` (a long-running server doesn't have a clean per-connection
shutdown path yet, so threads just die with the process rather than
expecting `session.stop()` to be called). The WebSocket route runs a
receive-loop and a send-loop concurrently (`asyncio.create_task`), since a
connection is full-duplex:

```python
session = Session()
session.start(daemon=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    receiver = AudioReceiver(session)

    async def send_interrupt():
        await websocket.send_json({"event": "interrupt"})

    send_task = asyncio.create_task(
        stream_audio_out(session, websocket.send_bytes, send_interrupt)
    )
    try:
        while True:
            data = await websocket.receive_bytes()
            receiver.feed(data)
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
```

**`client.py`** — mirrors `pipeline.py`'s shape but over the network:
`sd.InputStream`/`sd.OutputStream` reusing the same constants, bridged to a
`websockets.connect()` session via two concurrent asyncio tasks
(`send_mic_audio`, `receive_audio`). `receive_audio` distinguishes binary
frames (audio bytes → `playback_q`) from text frames (control message → set
a `threading.Event`). The playback callback checks that event on *every*
invocation — its equivalent of `op.py`'s `leftover_turn != current` check —
and if set, drops its `leftover` buffer and drains `playback_q` before
continuing, then clears the flag:

```python
def make_playback_callback(playback_q, flush_event):
    leftover = np.array([], dtype=OP_DTYPE)

    def playback_callback(output_data, frames, time, status):
        nonlocal leftover
        if flush_event.is_set():
            leftover = np.array([], dtype=OP_DTYPE)
            while True:
                try:
                    playback_q.get_nowait()
                except queue.Empty:
                    break
            flush_event.clear()

        while len(leftover) < frames:
            try:
                chunk = playback_q.get_nowait()
            except queue.Empty:
                break
            leftover = np.concatenate((leftover, chunk), axis=0)

        if len(leftover) >= frames:
            output_data[:, 0] = leftover[:frames]
            leftover = leftover[frames:]
        else:
            output_data[:len(leftover), 0] = leftover
            output_data[len(leftover):, 0] = 0
            leftover = np.array([], dtype=OP_DTYPE)

    return playback_callback
```

**Verdict**: tested end-to-end successfully, including interruption working
correctly over the wire (with the control-message fix above) — confirms the
architectural bet (decoupling workers from hardware I/O) paid off, with one
real gap found and closed along the way.

**Considered and ruled out: doing the whole thing in Node.js.** LLM and VAD
are portable (HTTP calls; Silero has an ONNX export). STT/TTS is the real
blocker — checked actual Node MLX projects directly rather than assume:
neither has any audio-model support, just LLM/vision. `whisper.cpp` is a
viable Node-native STT alternative if revisited, but no equally clear TTS
answer was found. Not pursued.

**Multi-session routing — designed, not yet built.** Current server is a
single module-level `Session`, shared by every connection. The proposed
direction: each connection gets its own `Session`, keyed by a session id
(a UUIDv7 was suggested, for roughly-sortable-by-creation-time ids), with a
registry mapping user id → session id → `Session` so a reconnecting user can
resume rather than starting fresh. Not implemented — `AudioReceiver`,
`stream_audio_out`, and `Session` itself are already shaped to make this
additive (they take a `session` object already, not global state), but the
per-connection creation/lookup/cleanup logic in `server.py` doesn't exist
yet.

---

## 13. Known issues & deferred work

**STT accuracy on short utterances** (§5) and **LLM history-recall
reliability on long/self-referential questions** (§6) — both covered where
they're discussed above; both are model-quality limitations, not
architecture bugs, and both were deliberately left as-is rather than
chasing a model swap.

**`sd.sleep()`/Ctrl+C shutdown hang** (§10) — pre-existing, deferred.

**Acoustic Echo Cancellation (AEC) — attempted, reverted, worth resuming
later.** Problem: on laptop speakers (as opposed to earphones), the agent's
own TTS output gets picked up by the mic and misidentified as new user
speech, causing the agent to reply to itself in a loop.

What was tried: a new `aec_worker` stage (`stages/aec.py`, kept on disk but
**not currently wired into `pipeline.py`**, and its imports are now stale
against the `core/`/`stages/` split above — would need updating before
reviving) using `pyaec` (WebRTC/speexdsp-based), fed a copy of the actual
speaker output as a reference signal tapped from `op.py`'s output callback.
Real, measurable progress was made:

- Fixed a real bug: `pyaec`'s `filter_length` needs to be an exact multiple
  of `frame_size` (both of its own examples do this) — our original value
  wasn't, and aligning it produced a large, confirmed jump in cancellation
  quality.
- Diagnosed and eliminated real-time resampling artifacts (the mic path
  runs 16kHz, speaker path ran 24kHz) two ways: first with a continuous/
  stateful resampler, then by unifying both paths to 16kHz entirely
  (resampling Kokoro's output once per sentence in `tts.py`, rather than
  per-block in real time). Neither fully fixed the remaining issue.
- Remaining problem: cancellation quality was **bursty** — sometimes 90%+
  echo reduction, sometimes near zero, inconsistently, even after removing
  every resampling artifact. Leading hypothesis: Python's GIL causes timing
  jitter between the mic-input and speaker-output callback threads
  (competing with STT/TTS/LLM inference for the GIL), and adaptive echo
  cancellers are sensitive to *consistent* timing between reference and
  near-end signals — a deeper architectural fix (moving real-time audio I/O
  off GIL-bound threads) would be needed to fully resolve it.
- A text-similarity-based backstop (comparing new transcriptions against the
  assistant's last reply) was considered and rejected — too easy to produce
  false negatives (suppressing real user speech that happens to echo the
  assistant's wording).

Reverted to the pre-AEC state rather than ship something unreliable. `pyaec`
and `scipy` remain installed dependencies. If resumed, the most promising
next step is a numeric (not textual) backstop: comparing cleaned mic energy
against a rolling estimate of recent reference energy, gating whether a
candidate utterance counts as a genuine new turn — combining the real (if
imperfect) audio-level cancellation with a second, independent signal.

**Multi-session server support**: designed but not built — see the end of §12.

---

## 14. Dependencies

```
fastapi, "uvicorn[standard]", websockets   # networking
numpy, soundfile, sounddevice, torch        # audio I/O + tensors
silero-vad                                  # VAD
mlx-audio (git), "misaki[en]"               # STT + TTS (MLX)
openai                                      # LLM client (against Ollama)
pyaec, scipy                                # installed for AEC, currently unused (see §13)
```

`mlx-audio` is pulled from git directly (`[tool.uv.sources]` in
`pyproject.toml`) rather than PyPI, since it's under active development.
