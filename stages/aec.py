import queue

import numpy as np
from pyaec import Aec

from stages.ip import BLOCK_SIZE as MIC_BLOCK_SIZE, SAMPLE_RATE as MIC_SAMPLE_RATE
from stages.op import SAMPLE_RATE as REF_SAMPLE_RATE

# tts.py resamples Kokoro's output down to op.py's SAMPLE_RATE once, per
# sentence, specifically so this stage never has to resample the reference
# signal in real time -- resampling small, unevenly-sized chunks on every
# mic frame was a real source of the timing jitter that made cancellation
# quality swing between very good and barely-there
assert MIC_SAMPLE_RATE == REF_SAMPLE_RATE, "mic and reference rates must match -- see tts.py's resample step"

# how long of an echo tail the adaptive filter can cancel -- laptop
# speaker->mic acoustic paths are short (a few ms), but our own
# queue/thread pipeline adds real delay between "audio queued for
# playback" and "it actually comes out the speaker", so this needs to
# cover that too. 0.3s is a starting point; if echo still leaks through,
# raise this first.
#
# speex's underlying echo canceller (MDF) processes the filter in
# frame_size-sized blocks internally -- both of pyaec's own examples keep
# filter_length an exact multiple of frame_size (1600/160=10, 6400/160=40),
# so round up to match rather than passing an arbitrary sample count
FILTER_SECONDS = 0.3
_raw_filter_length = int(MIC_SAMPLE_RATE * FILTER_SECONDS)
FILTER_LENGTH = -(-_raw_filter_length // MIC_BLOCK_SIZE) * MIC_BLOCK_SIZE  # round up to multiple of MIC_BLOCK_SIZE


def _float_to_int16(chunk):
    return np.clip(chunk * 32768.0, -32768, 32767).astype(np.int16)


def _int16_to_float(chunk):
    # cancel_echo returns a plain Python list of ints, not an ndarray
    return np.array(chunk, dtype=np.float32) / 32768.0


def _rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def aec_worker(mic_q, ref_q, vad_q):
    """Cancels TTS echo out of the mic signal before VAD ever sees it.

    mic_q: raw mic audio straight from the input callback (float32, 16kHz,
        MIC_BLOCK_SIZE-sized chunks -- same shape vad_q used to receive
        directly, before this stage existed).
    ref_q: copies of whatever was actually just written to the speaker
        (float32, 16kHz, op.py's BLOCK_SIZE-sized chunks), fed continuously
        by op.py's output callback -- including silence, which matters just
        as much as real audio for the echo canceller to stay aligned.
    vad_q: same queue vad_worker already reads from -- it gets cleaned
        16kHz float32 audio in the exact same shape as before, so nothing
        downstream needs to change.
    """
    aec = Aec(MIC_BLOCK_SIZE, FILTER_LENGTH, MIC_SAMPLE_RATE, True)

    # reference audio, waiting to be handed out MIC_BLOCK_SIZE samples at a
    # time to line up with each mic frame -- no resampling needed here
    # anymore since ref_q already arrives at MIC_SAMPLE_RATE
    ref_buffer = np.array([], dtype=np.float32)

    # --- temporary diagnostics: print mic/ref/cleaned loudness whenever the
    # agent is actually outputting audible audio, so we can see numerically
    # whether cancel_echo is attenuating anything at all. Remove once AEC is
    # confirmed working.
    frame_count = 0
    DEBUG_RMS_THRESHOLD = 0.01

    while True:
        chunk = mic_q.get()
        if chunk is None:
            vad_q.put(None)
            return

        # pull in every reference chunk that's shown up since the last mic
        # frame -- op.py's callback and our input callback are driven by
        # two independent audio-hardware clocks, so there's no guarantee
        # of a 1:1 handoff each loop iteration
        while True:
            try:
                ref_chunk = ref_q.get_nowait()
            except queue.Empty:
                break
            ref_buffer = np.concatenate((ref_buffer, ref_chunk.astype(np.float32)))

        if len(ref_buffer) >= MIC_BLOCK_SIZE:
            ref_frame = ref_buffer[:MIC_BLOCK_SIZE]
            ref_buffer = ref_buffer[MIC_BLOCK_SIZE:]
        else:
            # no reference audio buffered yet (e.g. right at startup) --
            # treat it as silence rather than stalling on the mic
            ref_frame = np.zeros(MIC_BLOCK_SIZE, dtype=np.float32)

        mic_frame = chunk.flatten()
        cleaned_int16 = aec.cancel_echo(_float_to_int16(mic_frame), _float_to_int16(ref_frame))
        cleaned = _int16_to_float(cleaned_int16)
        vad_q.put(cleaned)

        frame_count += 1
        ref_rms = _rms(ref_frame)
        if ref_rms > DEBUG_RMS_THRESHOLD and frame_count % 8 == 0:
            print(
                f"[aec] ref_rms={ref_rms:.4f} mic_rms={_rms(mic_frame):.4f} "
                f"cleaned_rms={_rms(cleaned):.4f}"
            )
