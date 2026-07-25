import queue

import numpy as np

from core.constants import OP_DTYPE


def make_output_callback(session):
    # uses: session.op_q (read), session.turn_tracker (current)
    op_q = session.op_q
    turn_tracker = session.turn_tracker

    leftover = np.array([], dtype=OP_DTYPE)
    leftover_turn = None

    def output_callback(output_data, frames, time, status):
        nonlocal leftover, leftover_turn
        if status:
            print(status)

        current = turn_tracker.current()

        # audio already sitting in the buffer was valid when it was queued,
        # but the turn may have moved on while it was waiting to be played --
        # drop it immediately so playback actually stops, rather than
        # finishing out whatever's left
        if leftover_turn is not None and leftover_turn != current:
            leftover = np.array([], dtype=OP_DTYPE)
            leftover_turn = None

        while len(leftover) < frames:
            try:
                turn_id, chunk = op_q.get_nowait()
            except queue.Empty:
                break
            if turn_id != current:
                continue  # stale chunk -- discard without ever buffering it
            leftover_turn = turn_id
            leftover = np.concatenate((leftover, chunk), axis=0)

        if len(leftover) >= frames:
            output_data[:, 0] = leftover[:frames]
            leftover = leftover[frames:]
        else:
            output_data[:len(leftover), 0] = leftover
            output_data[len(leftover):, 0] = 0
            leftover = np.array([], dtype=OP_DTYPE)
            leftover_turn = None

    return output_callback
