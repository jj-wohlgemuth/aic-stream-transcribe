import argparse
import sys
import tempfile
import numpy as np
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
    parser.add_argument(
        "-el",
        "--enhancement-level",
        type=float,
        default=0.8,
        help="audio enhancement intensity (0.0 to 1.0, default: 0.8)",
    )
    parser.add_argument(
        "-a",
        "--amplify",
        type=float,
        default=0.0,
        help="Amplification in decibels applied to the input before enhancement (default: 0.0 dB). Clips to [-1, 1].",
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
    print(f"Amplify:    {args.amplify:+.1f} dB")
    print("-" * 40)

    try:
        # 1. Load Audio
        print("➤ Loading audio...")
        raw_pcm, fs_hz = sf.read(args.input_file, dtype="float32")

        # 2. Apply amplification and clip before enhancement
        enhance_input = args.input_file
        if args.amplify != 0.0:
            gain = 10 ** (args.amplify / 20.0)
            amplified = raw_pcm * gain
            if np.any(np.abs(amplified) > 1.0):
                clipped_pct = np.mean(np.abs(amplified) > 1.0) * 100
                print(
                    f"Warning: clipping detected in {clipped_pct:.1f}% of samples. "
                    f"Consider reducing --amplify."
                )
            raw_pcm = np.clip(amplified, -1.0, 1.0)
            print(f"➤ Applied {args.amplify:+.1f} dB amplification with clipping")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                enhance_input = tmp.name
            sf.write(enhance_input, raw_pcm, fs_hz)

        # 3. Process Audio (AIC SDK)
        print("➤ Enhancing audio with AIC SDK...")
        enhanced_pcm = process_single_file(
            enhance_input,
            model_name=args.model,
            enhancement_level=args.enhancement_level,
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
