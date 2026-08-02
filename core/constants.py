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
LLM_MODEL = 'laguna-xs-2.1:nvfp4'
# laguna-xs-2.1:nvfp4 / lfm2.5 / north-mini-code-1.0:mlx-nvfp4
# LLM_MODEL = 'lfm2.5'
LLM_SYSTEM_PROMPT = (
    'You are Ria. You are a smart assistant. Respond briefly and conversationally. Clean and simple responses are best. '
    'No emojis or emoticons. STOP on interruptions. Do not repeat yourself. Do not make up information. '
    'If you do not know the answer, say "I don\'t know."'
    'Keep old context in mind, but do not repeat it back to the user.'
)  # seeds every new session's conversation_history -- see stages/session.py
SENTENCE_END_RE = re.compile(r'[.!?\n]')  # marks where a streamed LLM reply can be split off to TTS early

# --- tts.py ---
TTS_MODEL = 'mlx-community/Kokoro-82M-bf16'
