import argparse
import sys
import soundfile as sf
from pathlib import Path
from soniox_streamer import SonioxStreamer
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

    args = parser.parse_args()

    # --- Execution Logic ---
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Processing: {args.input_file}")
    print(f"Model:      {args.model}")
    print(f"Output:     {args.output}")
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

        # 3. Transcribe Raw
        print("➤ Transcribing RAW audio...")
        # Note: We pass on_update=None to keep the console clean
        transcript_raw = SonioxStreamer(fs_hz, "RAW", on_update=None).stream_array(
            raw_pcm, fs_hz
        )

        # 4. Transcribe Enhanced
        print("➤ Transcribing ENHANCED audio...")
        transcript_enhanced = SonioxStreamer(
            fs_hz, "ENHANCED", on_update=None
        ).stream_array(enhanced_pcm, fs_hz)

        # 5. Save Enhanced Audio
        sf.write(args.output, enhanced_pcm.T, fs_hz)
        print(f"➤ Saved enhanced audio to: {args.output}")

        # 6. Generate HTML Report
        print("➤ Generating HTML report...")
        create_html_report(
            transcript_raw, transcript_enhanced, args.input_file, args.output
        )

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
