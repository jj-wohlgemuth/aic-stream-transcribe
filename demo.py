import argparse
import sounddevice as sd
import numpy as np
import os
import sys
import threading
import select
from aic_sdk import (
    Model,
    Processor,
    ProcessorConfig,
    ProcessorParameter,
    VadParameter,
)
from stt_streamers import DeepgramStreamer, SonioxStreamer
import shutil
import textwrap
import soundfile as sf
from html_generator import create_html_report

FILE_PATH_RAW = "stream_raw.wav"
FILE_PATH_ENHANCED = "stream_enhanced.wav"

try:
    import termios
    import tty

    UNIX_AVAILABLE = True
except ImportError:
    UNIX_AVAILABLE = False

try:
    import msvcrt

    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False


def get_license_key() -> str | None:
    """Get license key from environment variable or .env file."""
    try:
        license_key = os.getenv("AIC_SDK_LICENSE")
    except Exception:
        license_key = None
    return license_key


class SDKParams:
    def __init__(
        self,
        enhancement_level: float,
        license_key: str,
        model_name: str,
        voice_gain_factor: float = 1,
    ):
        self.enhancement_level: float = enhancement_level
        self.license_key: str = license_key
        self.model_name: str = model_name
        self.voice_gain_factor: float = voice_gain_factor


class AudioProcessor:
    """
    Handles buffering and processing of audio frames for real-time processing.
    """

    def __init__(
        self,
        processor: Processor,
        vad_context,
        latency: int,
        chunk_size: int,
        num_channels: int,
        dtype,
        fs_hz: int,
        vad_status_callback=None,
        bypass_callback=None,
        transcriber_mix=None,
        transcriber_pred=None,
    ):
        self.processor = processor
        self.vad_context = vad_context
        self.latency = latency
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.fs_hz = fs_hz
        self.dtype = dtype
        self.last_speech_state = False
        self.vad_status_callback = vad_status_callback
        self.bypass_callback = bypass_callback
        self.transcriber_mix = transcriber_mix
        self.transcriber_pred = transcriber_pred
        self.last_bypass_state = False
        self.input_frames = []
        self.output_frames = []

    def process_frame(self, indata: np.ndarray) -> np.ndarray:
        """
        Process a frame of input audio.
        """
        self.input_frames.append(indata.copy())

        indata_copy = indata.copy()
        bypass_enabled = self.bypass_callback() if self.bypass_callback else False

        if bypass_enabled != self.last_bypass_state:
            self.last_bypass_state = bypass_enabled

        # If bypass is enabled, return input directly without processing
        if bypass_enabled:
            self.output_frames.append(indata_copy)  # Save unprocessed output
            return indata_copy

        input_ndim = indata_copy.ndim

        # Convert sounddevice format to SDK format: (frames, channels) -> (channels, frames)
        if indata.ndim == 1:
            indata_copy = indata_copy.reshape(1, -1)
        else:
            indata_copy = indata.T

        # Process the chunk (modifies chunk in-place)
        outdata = self.processor.process(indata_copy)

        # Check for speech activity via VAD Context
        is_speech = self.vad_context.is_speech_detected()

        # Update VAD status indicator
        if is_speech != self.last_speech_state:
            self.last_speech_state = is_speech
            vad_char = "●" if is_speech else "○"
            if self.vad_status_callback:
                self.vad_status_callback(vad_char)

        if self.transcriber_mix:
            self.transcriber_mix.process_chunk(indata_copy)
        if self.transcriber_pred:
            self.transcriber_pred.process_chunk(outdata)

        # Reshape output back to sounddevice format
        if input_ndim == 1:
            final_out = outdata.reshape(-1)
        else:
            final_out = outdata.T

        # 2. Capture Processed Output
        self.output_frames.append(final_out.copy())

        return final_out

    def save_files(self, raw_transcript: str, enhanced_transcript: str):
        """
        Merges buffered frames, writes them to WAV files, and
        triggers spek-cli to plot spectrograms in the terminal.
        """
        if not self.input_frames:
            print("No audio data to save.")
            return

        print("\nSaving audio recordings...")

        try:
            full_input = np.concatenate(self.input_frames, axis=0)
            full_output = np.concatenate(self.output_frames, axis=0)
            sf.write(FILE_PATH_RAW, full_input, self.fs_hz)
            sf.write(FILE_PATH_ENHANCED, full_output, self.fs_hz)
            print(f"Saved: {FILE_PATH_RAW} and {FILE_PATH_ENHANCED}")
        except Exception as e:
            print(f"Error processing files: {e}")
        # ---- Create HTML visualization ----
        try:
            create_html_report(
                raw_transcript,
                enhanced_transcript,
                FILE_PATH_RAW,
                FILE_PATH_ENHANCED,
                output_html="recordings.html",
            )
        except Exception as e:
            print(f"Error creating HTML report: {e}")


class AudioHandler:
    """
    Main audio handler class that manages the SDK Processor and V2 pipeline.
    """

    def __init__(
        self,
        params: SDKParams,
        fs_hz: int,
        num_channels: int,
        dtype: np.dtype,
        transcribe: bool = True,
        stt_api: str = "soniox",
    ):
        print("Initializing SDK model...")
        self.transcribe = transcribe

        if os.path.exists(params.model_name):
            model_path = params.model_name
        else:
            print(
                f"Model file '{params.model_name}' not found locally. Attempting download..."
            )
            cache_dir = "models"
            os.makedirs(cache_dir, exist_ok=True)
            try:
                model_path = Model.download(params.model_name, cache_dir)
                print(f"Downloaded model to: {model_path}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download model '{params.model_name}': {e}"
                )

        self.model = Model.from_file(model_path)
        self.processor = Processor(self.model, params.license_key)
        self.optimal_sample_rate = self.model.get_optimal_sample_rate()
        self.optimal_num_frames = self.model.get_optimal_num_frames(fs_hz)

        if fs_hz == self.optimal_sample_rate:
            config = ProcessorConfig.optimal(self.model, None, num_channels)
        else:
            config = ProcessorConfig(fs_hz, self.optimal_num_frames, num_channels)

        self.processor.initialize(config)
        print("Processor initialized successfully!")

        # --- UI STATE & CONFIG ---
        self.vad_status = "○"
        self.raw_text = ""
        self.enhanced_text = ""
        self.display_lock = threading.Lock()

        # UI Constants
        self.MAX_ROWS = 20  # Height of the rolling window (history)

        # Dynamic Column Width Calculation
        # We check the terminal width and subtract 3 chars for the separator " | "
        term_width = shutil.get_terminal_size((100, 20)).columns
        self.COL_WIDTH = max(30, (term_width - 5) // 2)

        self.proc_ctx = self.processor.get_processor_context()
        self.vad_ctx = self.processor.get_vad_context()

        # Configure Parameters
        try:
            self.proc_ctx.set_parameter(
                ProcessorParameter.EnhancementLevel, params.enhancement_level
            )
        except Exception as e:
            print(
                f"Warning: Failed to set Enhancement Level to {params.enhancement_level}: {e}"
            )
        try:
            self.proc_ctx.set_parameter(
                ProcessorParameter.VoiceGain, params.voice_gain_factor
            )
        except Exception as e:
            print(
                f"Warning: Failed to set Voice Gain to {params.voice_gain_factor}: {e}"
            )
        self.vad_ctx.set_parameter(VadParameter.Sensitivity, 15.0)

        processing_latency_samples = self.proc_ctx.get_output_delay()

        window_length_ms = int(
            round((self.optimal_num_frames / self.optimal_sample_rate) * 1000.0)
        )
        model_delay_ms = int(
            round((processing_latency_samples / self.optimal_sample_rate) * 1000.0)
        )
        output_delay_ms = int(round((processing_latency_samples / fs_hz) * 1000.0))

        self.params = params
        self.fs_hz = fs_hz
        self.window_length_ms = window_length_ms
        self.model_delay_ms = model_delay_ms
        self.output_delay_ms = output_delay_ms
        self.latency = processing_latency_samples
        self.chunk_size = self.optimal_num_frames
        self.status_printed = False

        self._bypass_enabled = threading.Event()
        self._bypass_lock = threading.Lock()

        # --- DISPLAY REFRESH LOGIC ---
        def process_text_to_lines(text, width):
            """
            Splits text by <end> and wraps lines that are too long.
            """
            segments = text.replace("<end>", "\n").split("\n")

            final_lines = []
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                wrapped = textwrap.wrap(segment, width=width)
                final_lines.extend(wrapped)

            return final_lines

        def refresh_ui():
            """Repaints the VAD status and the Multi-Row table."""
            if not self.status_printed:
                return

            with self.display_lock:
                r_lines = process_text_to_lines(self.raw_text, self.COL_WIDTH)
                e_lines = process_text_to_lines(self.enhanced_text, self.COL_WIDTH)

                # Slice to get the most recent history
                r_view = (
                    r_lines[-self.MAX_ROWS :]
                    if len(r_lines) > self.MAX_ROWS
                    else r_lines
                )
                e_view = (
                    e_lines[-self.MAX_ROWS :]
                    if len(e_lines) > self.MAX_ROWS
                    else e_lines
                )

                # Pad with empty lines
                while len(r_view) < self.MAX_ROWS:
                    r_view.append("")
                while len(e_view) < self.MAX_ROWS:
                    e_view.append("")

                # Move cursor UP to the start of the UI block
                total_height = 2 + self.MAX_ROWS
                sys.stdout.write(f"\033[{total_height}A")

                # 1. VAD Status
                sys.stdout.write(
                    f"\r\033[K➤ Voice Activity Detection: {self.vad_status}\n"
                )

                # 2. Header
                if self.transcriber_mix and self.transcriber_pred:
                    header = f"{'RAW Audio into ' + self.transcriber_mix.api_name:<{self.COL_WIDTH}}\
                             | {'ENHANCED Audio into ' + self.transcriber_pred.api_name:<{self.COL_WIDTH}}"
                    sys.stdout.write(f"\r\033[K{header}\n")

                # 3. Data Rows
                for i in range(self.MAX_ROWS):
                    row_str = (
                        f"{r_view[i]:<{self.COL_WIDTH}} | {e_view[i]:<{self.COL_WIDTH}}"
                    )
                    # Strict truncation to prevent wrapping glitches
                    max_w = (self.COL_WIDTH * 2) + 3
                    if len(row_str) > max_w:
                        row_str = row_str[:max_w]

                    sys.stdout.write(f"\r\033[K{row_str}\n")

                sys.stdout.flush()

        # --- CALLBACKS ---
        def update_vad_status(status):
            self.vad_status = status
            refresh_ui()

        def update_raw_transcript(text):
            self.raw_text = text
            refresh_ui()

        def update_enhanced_transcript(text):
            self.enhanced_text = text
            refresh_ui()

        def get_bypass_state():
            with self._bypass_lock:
                return self._bypass_enabled.is_set()

        if self.transcribe:
            Streamer = (
                DeepgramStreamer if stt_api.lower() == "deepgram" else SonioxStreamer
            )
            self.transcriber_mix = Streamer(
                fs_hz, "RAW", on_update=update_raw_transcript
            )
            self.transcriber_pred = Streamer(
                fs_hz, "ENHANCED", on_update=update_enhanced_transcript
            )
        else:
            self.transcriber_mix = None
            self.transcriber_pred = None

        self.processor_wrapper = AudioProcessor(
            processor=self.processor,
            vad_context=self.vad_ctx,
            latency=self.latency,
            chunk_size=self.chunk_size,
            num_channels=num_channels,
            fs_hz=fs_hz,
            dtype=dtype,
            vad_status_callback=update_vad_status,
            bypass_callback=get_bypass_state,
            transcriber_mix=self.transcriber_mix,
            transcriber_pred=self.transcriber_pred,
        )

    def toggle_bypass(self):
        with self._bypass_lock:
            if self._bypass_enabled.is_set():
                self._bypass_enabled.clear()
            else:
                self._bypass_enabled.set()

    def is_bypass_enabled(self):
        with self._bypass_lock:
            return self._bypass_enabled.is_set()

    def print_status(self):
        print("➤ Model:", self.params.model_name)
        print("➤ Optimal Sample Rate:", f"{self.optimal_sample_rate} Hz")
        print("➤ Processing Block Size:", self.optimal_num_frames)
        print("➤ Window Length:", f"{self.window_length_ms} ms")
        print("➤ Total Output Delay:", f"{self.output_delay_ms} ms")
        print()
        print("➤ Current Sampling Rate:", f"{self.fs_hz} Hz")
        print("➤ Enhancement Level:", self.params.enhancement_level)
        print("➤ Voice Gain:", self.params.voice_gain_factor)

        bypass_state = "ON" if self.is_bypass_enabled() else "OFF"
        print("➤ Bypass Mode:", bypass_state)

        # --- PREPARE THE LIVE DISPLAY AREA ---
        print(f"➤ Voice Activity Detection: {self.vad_status}")

        header = f"{'RAW':<{self.COL_WIDTH}} | {'ENHANCED':<{self.COL_WIDTH}}"
        print(header)

        # Fix: Print simple empty lines to reserve space.
        # This prevents auto-wrapping issues if the terminal is narrow.
        for _ in range(self.MAX_ROWS):
            print()

        self.status_printed = True

    def update_bypass_status(self):
        if self.status_printed:
            bypass_state = "ON" if self.is_bypass_enabled() else "OFF"
            # Move UP past the table to the bypass line
            # Jump = MAX_ROWS + Header(1) + VAD(1) + Bypass(1)
            jump = self.MAX_ROWS + 3
            sys.stdout.write(
                f"\033[{jump}A\r\033[K➤ Bypass Mode: {bypass_state}\033[{jump}B"
            )
            sys.stdout.flush()

    def callback(self, indata, outdata, frames, time, status):
        # Delegate to the processor wrapper
        processed = self.processor_wrapper.process_frame(indata)
        outdata[:] = processed

    def close(self):
        self.processor = None
        self.model = None
        mix_transcript = ""
        enhanced_transcript = ""
        if hasattr(self, "transcriber_mix") and self.transcriber_mix:
            mix_transcript = self.transcriber_mix.close()
        if hasattr(self, "transcriber_pred") and self.transcriber_pred:
            enhanced_transcript = self.transcriber_pred.close()
        if hasattr(self, "processor_wrapper"):
            self.processor_wrapper.save_files(mix_transcript, enhanced_transcript)


def int_or_str(text):
    try:
        return int(text)
    except ValueError:
        return text


def str_to_bool(text):
    if isinstance(text, bool):
        return text
    if text.lower() in ("true", "1", "yes", "on"):
        return True
    elif text.lower() in ("false", "0", "no", "off"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {text}")


# ... [KeyboardMonitor class remains unchanged] ...
class KeyboardMonitor:
    def __init__(self, callback, quit_event):
        self.callback = callback
        self.quit_event = quit_event
        self.running = False
        self.thread = None
        self.old_settings = None

    def _unix_monitor(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == " ":
                        self.callback()
                    elif char == "\n" or char == "\r":
                        self.quit_event.set()
                        break
        except Exception as e:
            print(f"\nKeyboard monitor error: {e}")
        finally:
            if self.old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except:
                    pass

    def _windows_monitor(self):
        try:
            while self.running:
                if msvcrt.kbhit():
                    char = msvcrt.getch()
                    if char == b" ":
                        self.callback()
                    elif char == b"\r":
                        self.quit_event.set()
                        break
        except Exception as e:
            print(f"\nKeyboard monitor error: {e}")

    def start(self):
        self.running = True
        if UNIX_AVAILABLE:
            self.thread = threading.Thread(target=self._unix_monitor, daemon=True)
        elif WINDOWS_AVAILABLE:
            self.thread = threading.Thread(target=self._windows_monitor, daemon=True)
        else:
            return False
        if self.thread:
            self.thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        if UNIX_AVAILABLE and self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except:
                pass


# Parse command line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser],
)
parser.add_argument(
    "-i",
    "--input-device",
    type=int_or_str,
    help="input device (numeric ID or substring)",
)
parser.add_argument(
    "-o",
    "--output-device",
    type=int_or_str,
    help="output device (numeric ID or substring)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="quail-vf-l-16khz",
    help="model name (e.g. quail-vf-l-16khz, sparrow-xxs-48khz) or path to .aicmodel file",
)
parser.add_argument(
    "-sr",
    "--samplerate",
    type=float,
    help="sampling rate (optional, overrides model optimal rate if provided)",
)
parser.add_argument(
    "-c",
    "--channels",
    type=int,
    default=1,
    help="number of channels (default: 1)",
)
parser.add_argument(
    "-el",
    "--enhancement-level",
    type=float,
    default=1.0,
    help="audio enhancement intensity (0.0 to 1.0, default: 1.0)",
)
parser.add_argument(
    "-vg",
    "--voice-gain",
    type=float,
    default=1.0,
    help="gain factor applied to voice signal (default: 1.0)",
)
parser.add_argument(
    "-t",
    "--transcribe",
    type=str_to_bool,
    default=True,
    help="enable transcription (true/false, default: true)",
)
parser.add_argument(
    "-s",
    "--stt-api",
    type=str,
    default="soniox",
    help="STT API for transcription (soniox or deepgram, default: soniox)",
)
args = parser.parse_args(remaining)

license_key = get_license_key()

if args.samplerate:
    sample_rate = int(args.samplerate)
    print(f"Using provided sample rate: {sample_rate} Hz")
else:
    # We will determine it inside AudioHandler based on the model
    sample_rate = 0  # Signal to discover optimal

assert license_key is not None, "AIC_SDK_LICENSE environment variable not set."

# Create SDK parameters
params = SDKParams(
    enhancement_level=args.enhancement_level,
    license_key=license_key,
    model_name=args.model,
    voice_gain_factor=args.voice_gain,
)


# Create audio handler
print("Setting up audio handler...")
# try:
# If sample_rate is 0, AudioHandler will set it to optimal
audio_handler = AudioHandler(
    params=params,
    fs_hz=sample_rate
    if sample_rate > 0
    else 48000,  # Init with safe default if unknown
    num_channels=args.channels,
    dtype=np.dtype("float32"),
    transcribe=args.transcribe,
    stt_api=args.stt_api if hasattr(args, "stt_api") else "soniox",
)

# If we auto-detected the rate, update our local variable for the Stream
if sample_rate == 0:
    sample_rate = audio_handler.optimal_sample_rate
    print(f"Using model optimal sample rate: {sample_rate} Hz")

# except Exception as e:
#     print(f"Failed to initialize SDK: {e}")
#     sys.exit(1)

# Start the stream
print("Starting audio stream...")
try:
    with sd.Stream(
        device=(args.input_device, args.output_device),
        samplerate=sample_rate,
        blocksize=audio_handler.chunk_size,
        dtype="float32",
        latency="low",
        channels=args.channels,
        callback=audio_handler.callback,
    ):
        print("Audio stream started successfully!")
        print()
        keyboard_supported = UNIX_AVAILABLE or WINDOWS_AVAILABLE
        if keyboard_supported:
            print("Press SPACE to toggle bypass, RETURN to quit...")
        else:
            print("Press RETURN to quit...")
        print()

        audio_handler.print_status()

        keyboard_monitor = None
        quit_event = threading.Event()

        try:
            if keyboard_supported:

                def on_spacebar():
                    audio_handler.toggle_bypass()
                    audio_handler.update_bypass_status()

                keyboard_monitor = KeyboardMonitor(on_spacebar, quit_event)
                keyboard_monitor.start()

                try:
                    while not quit_event.is_set():
                        quit_event.wait(timeout=0.1)
                except KeyboardInterrupt:
                    pass
            else:
                try:
                    input()
                except KeyboardInterrupt:
                    pass
        finally:
            if keyboard_monitor:
                keyboard_monitor.stop()
except KeyboardInterrupt:
    print("\nInterrupted by user.")
    parser.exit(0)
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    parser.exit(1)
finally:
    print()
    audio_handler.close()
