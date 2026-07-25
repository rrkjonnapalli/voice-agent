import time

import mlx.core as mx
from mlx_audio.stt import load

from core.constants import STT_MODEL

stt_model = load(STT_MODEL)  # shared across sessions -- concurrent calls verified safe, see test_model_concurrency.py
print('STT model loaded successfully!')


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

        start_time = time.time()
        mx_array = mx.array(audio_chunk)
        result = stt_model.generate(mx_array, language='en-US')
        end_time = time.time()
        print(f"Transcription time: {end_time - start_time:.2f} seconds")
        print(f"Transcribed text: {result.text}")

        if result.text.strip():
            # only now, once we know this utterance is real speech (not a
            # cough, a click, noise VAD mistook for speech), does it count
            # as an actual conversational turn worth possibly interrupting for
            turn_id = turn_tracker.bump()
            llm_q.put((turn_id, result.text))
