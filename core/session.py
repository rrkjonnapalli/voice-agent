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
    """Everything one conversation needs, bundled into one object that gets
    passed to every stage -- so a new session is just `Session()` +
    `.start()`, instead of hand-threading five queues through every call
    site separately.
    """

    def __init__(self):
        # queues -- one per stage boundary, same shape as pipeline.py today
        self.vad_q = queue.Queue()
        self.stt_q = queue.Queue()
        self.llm_q = queue.Queue()
        self.tts_q = queue.Queue()
        self.op_q = queue.Queue()

        self.turn_tracker = TurnTracker()

        # used to be a local variable inside llm_worker, living and dying
        # with that one function call -- moving it here keeps it exactly as
        # per-session as before (one Session = one history), but now it's
        # inspectable/loggable from outside the worker too
        self.conversation_history = [
            {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
        ]

        # VAD gets its own model + iterator per session, NOT the shared
        # global one stages.ip currently loads at module level. Verified via
        # test_model_concurrency.py: two VADIterator instances sharing one
        # vad_model crash the process when driven concurrently from separate
        # threads (Silero's model likely keeps recurrent hidden state inside
        # itself, not just in the iterator wrapper). The model is small, so
        # duplicating it per session is cheap -- unlike STT/TTS, where
        # concurrent calls into one shared model instance were verified safe,
        # so those stay shared globals (see stages/stt.py, stages/tts.py).
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(
            self.vad_model,
            sampling_rate=IP_SAMPLE_RATE,
            min_silence_duration_ms=200,
            speech_pad_ms=50,
            threshold=0.5,
        )
        print('VAD model loaded for new session')

        self._threads = []

    def start(self, daemon=False):
        """Starts this session's four worker threads, each given `self` as
        its only argument.

        daemon=False (pipeline.py's case) expects a clean stop() call --
        threads block on queue.get() until they see a sentinel. daemon=True
        (server.py's case, for now) means these threads just die with the
        process instead, since there's no per-connection stop() being
        called yet.
        """
        self._threads = [
            threading.Thread(target=vad_worker, args=(self,), daemon=daemon),
            threading.Thread(target=stt_worker, args=(self,), daemon=daemon),
            threading.Thread(target=llm_worker, args=(self,), daemon=daemon),
            threading.Thread(target=tts_worker, args=(self,), daemon=daemon),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        """Cascading shutdown, same pattern as pipeline.py: push exactly one
        sentinel into the front queue -- each stage drains its own backlog
        and forwards its own None downstream -- then join every thread.

        Only once every thread has actually exited (not just been asked to)
        does nothing reference `self` anymore -- so the explicit clearing
        below is a deliberate defensive step, not something Python strictly
        requires: it releases the per-session VAD model/iterator immediately
        rather than only when this Session object itself gets garbage
        collected, which protects against a stray reference elsewhere (e.g.
        a forgotten entry in a session registry) silently keeping a whole
        model pinned in memory.
        """
        self.vad_q.put(None)
        for t in self._threads:
            t.join()
        self._threads = []
        self.conversation_history = []

        self.vad_iterator = None
        self.vad_model = None
