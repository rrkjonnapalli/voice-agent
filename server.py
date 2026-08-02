import asyncio
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from core.session import Session
from stages.server_ip import AudioReceiver
from stages.server_op import stream_audio_out

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse("client.html")

# one Session per connection, keyed by a random id -- lets multiple clients
# (e.g. several devices running client.py at once) each get their own
# isolated queues/turn_tracker/conversation_history/VAD state with no
# cross-talk. Anonymous for now: a fresh id every connection, no notion of
# a persistent user identity and no reconnection support yet.
sessions: dict[str, Session] = {}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    session = Session()
    # daemon=True is a safety net, not a substitute for the explicit stop()
    # below -- it only matters if the *server process itself* gets killed
    # while sessions are still active, so those threads don't hang process
    # exit waiting on a sentinel nobody's going to send. The explicit
    # stop() in `finally` below is what actually runs on a normal
    # disconnect, releasing this session's VAD model instead of leaking it.
    session.start(daemon=True)
    sessions[session_id] = session
    print(f"[{session_id}] connected -- {len(sessions)} active session(s)")

    receiver = AudioReceiver(session)

    # sent as a text/JSON frame, distinct from the binary audio frames --
    # tells the client "an interruption happened, stop playing right now"
    async def send_interrupt():
        await websocket.send_json({"event": "interrupt"})

    # the connection is full-duplex: receiving mic audio from the client and
    # sending synthesized speech back both need to happen concurrently, not
    # one after the other -- so the send side runs as its own task alongside
    # the receive loop below
    send_task = asyncio.create_task(
        stream_audio_out(session, websocket.send_bytes, send_interrupt)
    )

    try:
        while True:
            data = await websocket.receive_bytes()
            receiver.feed(data)
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        # session.stop() joins four threads -- a blocking call. Running it
        # directly here would freeze the whole event loop (and therefore
        # every OTHER concurrent session) until this one's threads finish
        # draining, so it's offloaded to a worker thread instead.
        await asyncio.to_thread(session.stop)
        del sessions[session_id]
        print(f"[{session_id}] disconnected -- {len(sessions)} active session(s)")
