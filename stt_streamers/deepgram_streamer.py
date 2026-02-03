import json
import os
import threading
import urllib.parse
import numpy as np
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

DEEPGRAM_WEBSOCKET_URL = "wss://api.deepgram.com/v1/listen"


class DeepgramStreamer:
    def __init__(self, fs_hz: int, stream_name: str, on_update=None) -> None:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPGRAM_API_KEY.")

        self.stream_name = stream_name
        self.api_name = "Deepgram V1 Nova-3"
        self.on_update = on_update
        self.final_tokens: list[dict] = []
        self.lock = threading.Lock()
        self.finished_event = threading.Event()

        # 1. Build the Deepgram URL with query parameters
        config = self.get_config(fs_hz)
        query_string = urllib.parse.urlencode(config)
        url_with_params = f"{DEEPGRAM_WEBSOCKET_URL}?{query_string}"

        print(f"Connecting {stream_name} to Deepgram...")

        # 2. Connect with Authorization header
        # Deepgram requires the API key in the headers
        headers = {"Authorization": f"Token {api_key}"}
        self.ws = connect(url_with_params, additional_headers=headers)

        # 3. Start the receiving thread
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stream_array(self, pcm: np.ndarray, fs_hz: int) -> str:
        """
        Streams audio chunks to Deepgram and waits for the final result.
        """
        chunk_size = 160  # Keeping the same chunk size as the reference
        num_chunks = int(np.ceil(len(pcm) / chunk_size))
        print(f"Streaming {self.stream_name} audio to Deepgram...")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(pcm))
            chunk = pcm[start_idx:end_idx]
            self.process_chunk(chunk)

        # Signal the end of the stream to Deepgram
        self.close()

        # Wait for the 'finished' signal (Metadata) from the receive loop
        self.finished_event.wait()

        # Ensure connection is closed and return final text
        self._ensure_closed()
        with self.lock:
            return self.render_tokens(self.final_tokens, [])

    def get_config(self, fs_hz: int) -> dict:
        """
        Returns parameters for the Deepgram V1 URL query string.
        """
        assert fs_hz == 16000, "Only 16 kHz audio is supported."

        return {
            "model": "nova-3",  # Recommended general model
            "encoding": "linear16",  # Corresponds to pcm_s16le
            "sample_rate": 16000,
            "channels": 1,
            "smart_format": "true",  # handling punctuation/formatting
            "interim_results": "true",  # required for non-final updates
            "endpointing": "500",  # ms silence to trigger finalization
        }

    def process_chunk(self, chunk: np.ndarray) -> None:
        """
        Converts float32 numpy array to int16 bytes and sends to WebSocket.
        """
        chunk = np.clip(chunk, -1.0, 1.0)
        chunk_int16 = (chunk * 32767).astype(np.int16)
        if len(chunk_int16) > 0:
            try:
                self.ws.send(chunk_int16.tobytes())
            except Exception:
                pass

    def render_tokens(
        self, final_tokens: list[dict], non_final_tokens: list[dict]
    ) -> str:
        """
        Renders the list of token dicts into a string.
        Matches Soniox logic: treats certain tokens as punctuation triggers.
        """
        text_parts = []
        for token in final_tokens + non_final_tokens:
            text = token["text"]
            text_parts.append(text)
            # Add newline if the text chunk looks like end-of-sentence punctuation
            # Note: Deepgram 'smart_format' usually attaches punctuation to the word.
            if text.strip() in [".", "?", "!"]:
                text_parts.append("\n")
        return "".join(text_parts)

    def _receive_loop(self):
        """
        Background loop to handle incoming JSON messages from Deepgram.
        """
        try:
            while True:
                message = self.ws.recv()
                res = json.loads(message)

                # Check for metadata indicating stream end
                if res.get("type") == "Metadata":
                    self.finished_event.set()
                    break

                # Deepgram error handling
                if "error" in res:
                    print(f"Deepgram Error: {res['error']}")
                    break

                # Process Transcripts
                # Deepgram V1 structure: result -> channel -> alternatives -> [0] -> transcript
                if "channel" in res:
                    is_final = res.get("is_final", False)
                    alternatives = res["channel"].get("alternatives", [])

                    if alternatives:
                        transcript = alternatives[0].get("transcript", "")

                        if transcript:
                            # Wrap the transcript in a dict to match the
                            # 'render_tokens' expectation of a list[dict]
                            token_data = {
                                "text": transcript + " ",  # Add space for readability
                                "is_final": is_final,
                            }

                            non_final_tokens = []
                            with self.lock:
                                if is_final:
                                    self.final_tokens.append(token_data)
                                else:
                                    non_final_tokens.append(token_data)

                                current_finals = list(self.final_tokens)

                            # Trigger the callback
                            text = self.render_tokens(current_finals, non_final_tokens)
                            if self.on_update:
                                self.on_update(text)

        except (ConnectionClosedOK, ConnectionClosedError):
            pass
        except Exception as e:
            print(f"Deepgram receive loop error: {e}")
        finally:
            self.finished_event.set()

    def close(self) -> str:
        """
        Sends the specific JSON message Deepgram expects to close the stream.
        """
        if hasattr(self, "ws"):
            try:
                # Deepgram V1 expects this specific JSON to close the stream
                self.ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        return self.render_tokens(self.final_tokens, [])

    def _ensure_closed(self) -> None:
        """
        Physical socket closure and thread cleanup.
        """
        if hasattr(self, "ws"):
            try:
                self.ws.close()
            except Exception:
                pass

        if (
            hasattr(self, "thread")
            and self.thread.is_alive()
            and self.thread != threading.current_thread()
        ):
            self.thread.join(timeout=1.0)
