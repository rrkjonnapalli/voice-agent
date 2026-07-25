import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets

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

SERVER_URL = "ws://localhost:8000/ws"

mic_q = queue.Queue()


def mic_callback(indata, frames, time, status):
    if status:
        print(status)
    mic_q.put(indata.copy().tobytes())


def make_playback_callback(playback_q, flush_event):
    # same leftover-buffer shape as stages/op.py, but no per-chunk
    # turn_tracker/staleness check here -- the server already filtered out
    # stale *unsent* audio before it ever went over the wire (see
    # stages/server_op.py). What the server can't do is unsend bytes
    # already in flight when an interruption happens, so it sends an
    # explicit control message instead -- flush_event.is_set() is this
    # callback's equivalent of op.py's "leftover_turn != current" check:
    # re-checked every invocation, drops whatever's buffered the moment it
    # sees the flag, rather than waiting to run out of audio naturally
    leftover = np.array([], dtype=OP_DTYPE)

    def playback_callback(output_data, frames, time, status):
        nonlocal leftover
        if status:
            print(status)

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


async def send_mic_audio(websocket):
    while True:
        try:
            data = mic_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        await websocket.send(data)


async def receive_audio(websocket, playback_q, flush_event):
    # binary frames = audio bytes (as before); anything else = a control
    # message (JSON text frame) -- currently only one kind exists
    # ({"event": "interrupt"}), so no need to parse it, just react
    async for message in websocket:
        if isinstance(message, bytes):
            chunk = np.frombuffer(message, dtype=OP_DTYPE)
            playback_q.put(chunk)
        else:
            flush_event.set()


async def main():
    playback_q = queue.Queue()
    flush_event = threading.Event()

    async with websockets.connect(SERVER_URL) as websocket:
        print(f"Connected to {SERVER_URL}")

        with sd.InputStream(
            samplerate=IP_SAMPLE_RATE,
            channels=IP_CHANNELS,
            dtype=IP_DTYPE,
            blocksize=IP_BLOCK_SIZE,
            callback=mic_callback,
        ), sd.OutputStream(
            samplerate=OP_SAMPLE_RATE,
            channels=OP_CHANNELS,
            dtype=OP_DTYPE,
            blocksize=OP_BLOCK_SIZE,
            callback=make_playback_callback(playback_q, flush_event),
        ):
            print("Talking to the server. Press Ctrl+C to stop.")
            # both directions run concurrently, same reasoning as the
            # server's receive-loop + send-task: mic audio going out and
            # synthesized speech coming back both need to happen at once
            await asyncio.gather(
                send_mic_audio(websocket),
                receive_audio(websocket, playback_q, flush_event),
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
