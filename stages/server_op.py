import asyncio
import queue


async def stream_audio_out(session, send_bytes, send_control):
    """Drains session.op_q and hands raw audio bytes to send_bytes (an async
    callable, e.g. a websocket's .send_bytes) as they become available.

    Same staleness rule as the local op.py output callback -- skip chunks
    whose turn has already been superseded -- but with one important
    difference: bytes already sent over the wire can't be "unsent" the way
    op.py can drop already-buffered *local* audio mid-playback (it just
    re-checks turn_tracker on every callback and stops feeding the
    speaker). Once this coroutine has handed bytes to send_bytes, they're
    gone -- the client may already be playing them by the time a newer
    turn starts.

    So on top of the same filtering, this also watches turn_tracker every
    loop iteration, and the moment it notices the turn has changed since
    the last chunk it sent, it calls send_control() once -- an explicit
    "stop playing whatever you have buffered, right now" signal. That's
    what lets the client stay simple (no turn-tracking of its own): it
    just reacts to this one event by flushing its playback buffer,
    instead of needing to replicate op.py's per-chunk staleness logic.

    Takes send_bytes/send_control as plain async callables (not a
    websocket object directly) so this stage has no FastAPI/Starlette
    import at all -- keeps it usable from any transport that can hand it
    async senders.
    """
    # uses: session.op_q (read), session.turn_tracker (current)
    op_q = session.op_q
    turn_tracker = session.turn_tracker
    last_known_turn = turn_tracker.current()

    while True:
        current = turn_tracker.current()
        if current != last_known_turn:
            await send_control()
            last_known_turn = current

        try:
            turn_id, chunk = op_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue

        if turn_id != current:
            continue  # stale, discard without sending

        await send_bytes(chunk.tobytes())
