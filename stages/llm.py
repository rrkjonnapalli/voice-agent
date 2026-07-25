import time

from openai import OpenAI

from core.constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, SENTENCE_END_RE

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)  # shared across sessions -- lightweight HTTP client, no per-session state


def llm_worker(session):
    # uses: session.llm_q (read), session.tts_q (write), session.turn_tracker
    # (current), session.conversation_history (read + append, mutated in
    # place -- same list object for the life of the session)
    llm_q = session.llm_q
    tts_q = session.tts_q
    turn_tracker = session.turn_tracker
    conversation_history = session.conversation_history

    while True:
        item = llm_q.get()
        if item is None:
            tts_q.put(None)
            return
        my_turn, text = item

        # always keep the user's words in context, even if we're about to
        # decide not to generate a spoken reply for them
        conversation_history.append({'role': 'user', 'content': text})

        if my_turn != turn_tracker.current():
            print(f"Skipping turn {my_turn} (superseded before generation started)")
            continue

        start_time = time.time()

        stream = client.chat.completions.create(
            model=LLM_MODEL,
            stream=True,
            messages=conversation_history,
        )

        started = False
        response_text = ""
        sentence_buffer = ""
        interrupted = False

        for chunk in stream:
            # re-checked on every chunk -- this is what lets a reply get cut
            # off mid-generation, not just skipped before it starts
            if my_turn != turn_tracker.current():
                interrupted = True
                # breaking out of this loop alone does NOT close the
                # underlying HTTP connection -- the Stream object would just
                # sit there, still open, until Python happens to garbage
                # collect it. Close it explicitly so Ollama actually stops
                # generating tokens for a reply nobody will hear.
                stream.close()
                break

            piece = chunk.choices[0].delta.content or ""
            if not piece:
                continue

            if not started:
                started = True
                print(f"Response started at: {time.time() - start_time:.2f} seconds")
            print(piece, end="", flush=True)

            response_text += piece
            sentence_buffer += piece

            # flush every complete sentence in the buffer as soon as it appears,
            # so TTS can start on sentence 1 while the LLM is still generating sentence 2
            while True:
                match = SENTENCE_END_RE.search(sentence_buffer)
                if not match:
                    break
                end_idx = match.end()
                sentence = sentence_buffer[:end_idx].strip()
                sentence_buffer = sentence_buffer[end_idx:]
                if sentence:
                    tts_q.put((my_turn, sentence))

        if interrupted:
            # keep whatever was actually generated (real context is useful),
            # but mark it explicitly as cut short -- otherwise the model has
            # no way to tell an interrupted fragment from a finished thought
            partial = response_text.strip()
            marker = "[interrupted by user before finishing]"
            conversation_history.append({
                'role': 'assistant',
                'content': f"{partial} {marker}" if partial else marker,
            })
            print(f"\nInterrupted turn {my_turn} -- recorded partial reply as interrupted")
            continue

        # trailing fragment with no terminating punctuation (e.g. reply didn't end in . ! ?)
        if sentence_buffer.strip():
            tts_q.put((my_turn, sentence_buffer.strip()))

        end_time = time.time()
        print(f"\nResponse time: {end_time - start_time:.2f} seconds")
        conversation_history.append({'role': 'assistant', 'content': response_text})
