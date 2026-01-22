import json
import os
import threading
import numpy as np
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosedOK

SONIOX_WEBSOCKET_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxStreamer:
    def __init__(self, fs_hz: int, stream_name: str, on_update=None) -> None:
        api_key = os.environ.get("SONIOX_API_KEY")
        if not api_key:
            raise RuntimeError("Missing SONIOX_API_KEY.")
        self.stream_name = stream_name
        self.on_update = on_update

        # Store final tokens as class attribute to persist them
        self.final_tokens: list[dict] = []
        self.lock = threading.Lock()  # Thread safety for the list

        config = self.get_config(api_key, fs_hz)

        print(f"Connecting {stream_name} to Soniox...")
        self.ws = connect(SONIOX_WEBSOCKET_URL)
        self.ws.send(json.dumps(config))

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def get_config(self, api_key: str, fs_hz: int) -> dict:
        config = {
            "api_key": api_key,
            "model": "stt-rt-v3",
            "language_hints": ["en"],
            "language_hints_strict": True,
            "enable_language_identification": True,
            "enable_speaker_diarization": False,
            "enable_endpoint_detection": True,
        }
        assert fs_hz == 16000, "Only 16 kHz audio is supported."
        config["audio_format"] = "pcm_s16le"
        config["sample_rate"] = 16000
        config["num_channels"] = 1
        return config

    def process_chunk(self, chunk: np.ndarray) -> None:
        chunk = np.clip(chunk, -1.0, 1.0)
        chunk_int16 = (chunk * 32767).astype(np.int16)
        if len(chunk_int16) > 0:
            try:
                self.ws.send(chunk_int16.tobytes())
            except Exception:
                pass  # Connection likely closed

    def render_tokens(
        self, final_tokens: list[dict], non_final_tokens: list[dict]
    ) -> str:
        text_parts = []
        for token in final_tokens + non_final_tokens:
            text = token["text"]
            text_parts.append(text)
            # Add a newline after sentence terminators
            if text.strip() in [".", "?", "!"]:
                text_parts.append("\n")
        return "".join(text_parts)

    def _receive_loop(self):
        try:
            while True:
                message = self.ws.recv()
                res = json.loads(message)

                if res.get("error_code") is not None:
                    break

                non_final_tokens: list[dict] = []

                # Protect access to self.final_tokens
                with self.lock:
                    for token in res.get("tokens", []):
                        if token.get("text"):
                            if token.get("is_final"):
                                self.final_tokens.append(token)
                            else:
                                non_final_tokens.append(token)

                    # Create snapshot for rendering
                    current_finals = list(self.final_tokens)

                text = self.render_tokens(current_finals, non_final_tokens)

                if self.on_update:
                    self.on_update(text)

                if res.get("finished"):
                    break

        except ConnectionClosedOK:
            pass
        except Exception:
            pass

    def close(self) -> str:
        """Closes the connection and returns the full final transcript."""
        if hasattr(self, "ws"):
            try:
                # Send empty message to signal end of stream if supported
                self.ws.send("")
            except Exception:
                pass
            self.ws.close()

        # Join the thread to ensure processing is done (optional, prevents race conditions)
        if (
            hasattr(self, "thread")
            and self.thread.is_alive()
            and self.thread != threading.current_thread()
        ):
            self.thread.join(timeout=1.0)

        with self.lock:
            return self.render_tokens(self.final_tokens, [])
