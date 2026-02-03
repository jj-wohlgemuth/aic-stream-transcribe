import argparse
import sys
import soundfile as sf
from pathlib import Path
from stt_streamers import SonioxStreamer, DeepgramStreamer
from html_generator import create_html_report
from aic_sdk_enhancer import process_single_file


def main():
    parser = argparse.ArgumentParser(
        description="Process a WAV file: Enhance with AIC SDK, Transcribe with Soniox, and generate an HTML report."
    )

    # 1. Positional Argument: Input File
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input WAV file (e.g., stream_raw.wav)",
    )

    # 2. Optional Argument: Model Name
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="quail-vf-l-16khz",
        help="AIC model name or path (default: quail-vf-l-16khz)",
    )

    # 3. Optional Argument: Output Filename
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="prerecorded_enhanced.wav",
        help="Path for the enhanced output WAV file (default: prerecorded_enhanced.wav)",
    )
    parser.add_argument(
        "-s",
        "--stt-api",
        type=str,
        default="soniox",
        help="STT API for transcription (soniox or deepgram, default: soniox)",
    )

    args = parser.parse_args()

    # --- Execution Logic ---
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Processing: {args.input_file}")
    print(f"Model:      {args.model}")
    print(f"Output:     {args.output}")
    print(f"STT API:    {args.stt_api}")
    print("-" * 40)

    try:
        # 1. Load Audio
        print("➤ Loading audio...")
        raw_pcm, fs_hz = sf.read(args.input_file, dtype="float32")

        # 2. Process Audio (AIC SDK)
        print("➤ Enhancing audio with AIC SDK...")
        enhanced_pcm = process_single_file(
            args.input_file,
            model_name=args.model,
            enhancement_level=None,  # Use model default
        )

        api_map = {
            "soniox": SonioxStreamer,
            "deepgram": DeepgramStreamer,
        }

        stt_api = args.stt_api.lower()
        if stt_api not in api_map:
            raise ValueError(
                f"Invalid STT API. Choose from: {', '.join(api_map.keys())}"
            )

        print(f"➤ Transcribing audio with {stt_api.capitalize()}...")
        Streamer = api_map[stt_api]

        transcript_raw = Streamer(fs_hz, "RAW").stream_array(raw_pcm, fs_hz)
        transcript_enhanced = Streamer(fs_hz, "ENHANCED").stream_array(
            enhanced_pcm, fs_hz
        )

        # 5. Save Enhanced Audio
        sf.write(args.output, enhanced_pcm.T, fs_hz)
        print(f"➤ Saved enhanced audio to: {args.output}")

        # 6. Generate HTML Report
        print("➤ Generating HTML report...")
        import os

        input_file_basename = os.path.basename(args.input_file)
        create_html_report(
            transcript_raw,
            transcript_enhanced,
            args.input_file,
            args.output,
            output_html=f"report_{input_file_basename}.html",
        )

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
