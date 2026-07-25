import threading


class TurnTracker:
    """Shared "which conversation turn is currently active" counter.

    VAD calls bump() every time it detects the user starting to speak --
    this is the only place the counter changes. Every other stage tags the
    data it produces with the turn id it was working on (captured from
    bump()'s return value, or passed through from upstream), and compares
    that tag against current() before doing further work with it. A
    mismatch means a newer turn has started since this data was produced,
    so it belongs to an interrupted turn and should be discarded.
    """

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
