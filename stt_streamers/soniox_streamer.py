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
        self.api_name = "Soniox RT"
        self.on_update = on_update
        self.final_tokens: list[dict] = []
        self.lock = threading.Lock()
        self.finished_event = threading.Event()
        config = self.get_config(api_key, fs_hz)
        print(f"Connecting {stream_name} to Soniox...")
        self.ws = connect(SONIOX_WEBSOCKET_URL)
        self.ws.send(json.dumps(config))

        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()

    def stream_array(self, pcm: np.ndarray, fs_hz: int) -> str:
        chunk_size = 160
        num_chunks = int(np.ceil(len(pcm) / chunk_size))
        print(f"Streaming {self.stream_name} audio to Soniox...")
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(pcm))
            chunk = pcm[start_idx:end_idx]
            self.process_chunk(chunk)
        try:
            self.ws.send("")
        except Exception:
            pass

        # 3. Wait for the 'finished' message from the receive loop
        self.finished_event.wait()
        self.close()
        with self.lock:
            return self.render_tokens(self.final_tokens, [])

    def get_config(self, api_key: str, fs_hz: int) -> dict:
        config = {
            "api_key": api_key,
            "model": "stt-rt-v3",
            "language_hints": ["en", "de"],
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
                pass

    def render_tokens(
        self, final_tokens: list[dict], non_final_tokens: list[dict]
    ) -> str:
        text_parts = []
        for token in final_tokens + non_final_tokens:
            text = token["text"]
            text_parts.append(text)
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

                with self.lock:
                    for token in res.get("tokens", []):
                        if token.get("text"):
                            if token.get("is_final"):
                                self.final_tokens.append(token)
                            else:
                                non_final_tokens.append(token)

                    current_finals = list(self.final_tokens)

                text = self.render_tokens(current_finals, non_final_tokens)

                if self.on_update:
                    self.on_update(text)

                if res.get("finished"):
                    # Signal stream_array to stop waiting
                    self.finished_event.set()
                    break

        except ConnectionClosedOK:
            pass
        except Exception:
            pass
        finally:
            self.finished_event.set()

    def close(self) -> str:
        """Closes the connection."""
        if hasattr(self, "ws"):
            try:
                self.ws.send("")
            except Exception:
                pass
            self.ws.close()

        if (
            hasattr(self, "thread")
            and self.thread.is_alive()
            and self.thread != threading.current_thread()
        ):
            self.thread.join(timeout=1.0)

        with self.lock:
            return self.render_tokens(self.final_tokens, [])
