import sounddevice as sd

from core.constants import (
    IP_BLOCK_SIZE,
    IP_CHANNELS,
    IP_DTYPE,
    IP_SAMPLE_RATE,
    OP_BLOCK_SIZE,
    OP_CHANNELS,
    OP_DTYPE,
    OP_SAMPLE_RATE,
)
from core.session import Session
from stages.ip import make_input_callback
from stages.op import make_output_callback

session = Session()
session.start()

print("Started recording. Press Ctrl+C to stop.")

with sd.InputStream(
    samplerate=IP_SAMPLE_RATE,
    channels=IP_CHANNELS,
    dtype=IP_DTYPE,
    blocksize=IP_BLOCK_SIZE,
    callback=make_input_callback(session),
), sd.OutputStream(
    samplerate=OP_SAMPLE_RATE,
    channels=OP_CHANNELS,
    dtype=OP_DTYPE,
    blocksize=OP_BLOCK_SIZE,
    callback=make_output_callback(session),
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Stopped recording.")

session.stop()
