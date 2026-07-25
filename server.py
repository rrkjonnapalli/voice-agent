import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from core.session import Session
from stages.server_ip import AudioReceiver
from stages.server_op import stream_audio_out

app = FastAPI()

# single session for the whole server, for now -- matches the previous
# behavior exactly. Routing multiple concurrent sessions (one per
# connection, keyed by a session/user id) is a separate, not-yet-built step.
session = Session()
session.start(daemon=True)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
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
