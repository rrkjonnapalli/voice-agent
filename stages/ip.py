import collections

import numpy as np
import torch

from core.constants import BUFFER_CHUNKS

# note: no module-level vad_model/VADIterator here anymore -- each Session
# loads its own (see stages/session.py). Two VADIterator instances sharing
# one vad_model crash when driven concurrently from separate threads
# (confirmed via test_model_concurrency.py), so VAD can't be a shared global
# the way stt_model/tts's model are.


def make_input_callback(session):
    # uses: session.vad_q (write)
    vad_q = session.vad_q

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

    # note: this stage no longer touches the turn tracker. VAD firing 'start'
    # doesn't by itself mean anything worth interrupting for (could be a
    # cough, a click) -- the turn only bumps once STT confirms there was
    # actually a word in it. See stt_worker.
    started = False
    utterance_chunks = []
    rolling_buffer = collections.deque(maxlen=BUFFER_CHUNKS)

    while True:
        chunk = vad_q.get()
        if chunk is None:
            stt_q.put(None)
            return

        audio_tensor = torch.from_numpy(chunk.flatten()).float()
        is_speech = vad_iterator(audio_tensor)

        if is_speech is None:
            pass
        elif 'start' in is_speech:
            utterance_chunks = list(rolling_buffer)
            started = True
        elif 'end' in is_speech:
            utterance_chunks.append(chunk)
            stt_q.put(np.concatenate(utterance_chunks, axis=0).flatten())
            started = False

        if started:
            utterance_chunks.append(chunk)
        rolling_buffer.append(chunk)
