import numpy as np
from mlx_audio.tts.utils import load_model

from core.constants import TTS_MODEL

model = load_model(TTS_MODEL)  # shared across sessions -- concurrent calls verified safe, see test_model_concurrency.py

print('TTS model loaded successfully!')

def tts_worker(session):
    # uses: session.tts_q (read), session.op_q (write), session.turn_tracker (current)
    tts_q = session.tts_q
    op_q = session.op_q
    turn_tracker = session.turn_tracker

    while True:
        item = tts_q.get()
        if item is None:
            # nothing downstream is blocked on a sentinel from op_q (the output
            # callback drains it with get_nowait(), not a blocking loop), so the
            # cascade stops here
            return
        turn_id, sentence = item

        # skip synthesis entirely for a sentence whose turn has already been
        # superseded -- no point spending model time on speech nobody will hear
        if turn_id != turn_tracker.current():
            continue

        for result in model.generate(sentence, voice=None, lang_code="a"):
            op_q.put((turn_id, np.array(result.audio)))
