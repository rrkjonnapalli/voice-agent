import numpy as np

from core.constants import IP_BLOCK_SIZE, IP_DTYPE


class AudioReceiver:
    """Buffers raw audio bytes arriving over the network into fixed
    IP_BLOCK_SIZE chunks (matching what vad_worker/Silero VAD expects, same
    512-sample requirement from the local mic path) before pushing them
    onto session.vad_q.

    One instance per connection -- holds its own leftover buffer, no shared
    global state. One of these gets created per Session.
    """

    def __init__(self, session):
        # uses: session.vad_q (write)
        self.vad_q = session.vad_q
        self.leftover = np.array([], dtype=IP_DTYPE)

    def feed(self, raw_bytes):
        incoming = np.frombuffer(raw_bytes, dtype=IP_DTYPE)
        self.leftover = np.concatenate((self.leftover, incoming))

        while len(self.leftover) >= IP_BLOCK_SIZE:
            chunk = self.leftover[:IP_BLOCK_SIZE]
            self.leftover = self.leftover[IP_BLOCK_SIZE:]
            self.vad_q.put(chunk)
