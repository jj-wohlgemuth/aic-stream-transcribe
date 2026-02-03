import argparse
import sys
import soundfile as sf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from stt_streamers import SonioxStreamer, DeepgramStreamer
from html_generator import create_html_report
from aic_sdk_enhancer import process_single_file

# Define a list of colors to cycle through
COLORS = ["green", "yellow", "blue", "magenta", "cyan", "red"]


def process_file(
    file_path: Path, output_dir: Path, model_name: str, stt_api: str, position: int
):
    """
    Process a single WAV file with a dedicated progress bar.
    """
    stem = file_path.stem
    output_wav_path = output_dir / f"{stem}_enhanced.wav"
    output_html_path = output_dir / f"report_{stem}.html"

    # Pick a color based on the position
    bar_color = COLORS[position % len(COLORS)]

    # Create a progress bar for this specific file
    # total=5 steps: Load -> Enhance -> Transcribe Raw -> Transcribe Enh -> Save/Report
    with tqdm(
        total=5,
        position=position,
        desc=f"{file_path.name[:15]:<15}",
        leave=False,  # Clear bar when done to keep UI clean
        colour=bar_color,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
    ) as pbar:
        try:
            pbar.set_postfix_str("Loading...")
            raw_pcm, fs_hz = sf.read(str(file_path), dtype="float32")
            pbar.update(1)

            # 1. Process Audio
            pbar.set_postfix_str("Enhancing...")
            enhanced_pcm = process_single_file(
                str(file_path),
                model_name=model_name,
                enhancement_level=None,
            )
            pbar.update(1)

            Streamer = (
                DeepgramStreamer if stt_api.lower() == "deepgram" else SonioxStreamer
            )
            pbar.set_postfix_str("Transcribing Raw...")
            transcript_raw = Streamer(fs_hz, "RAW").stream_array(raw_pcm, fs_hz)
            pbar.update(1)

            pbar.set_postfix_str("Transcribing Enh...")
            transcript_enhanced = Streamer(fs_hz, "ENHANCED").stream_array(
                enhanced_pcm, fs_hz
            )
            pbar.update(1)

            pbar.set_postfix_str("Saving...")
            if enhanced_pcm.ndim > 1 and enhanced_pcm.shape[0] < enhanced_pcm.shape[1]:
                enhanced_pcm = enhanced_pcm.T

            sf.write(str(output_wav_path), enhanced_pcm, fs_hz)

            create_html_report(
                transcript_raw,
                transcript_enhanced,
                str(file_path),
                str(output_wav_path),
                output_html=str(output_html_path),
            )
            pbar.update(1)

            return True

        except Exception:
            pbar.colour = "red"  # Turn bar red on error
            pbar.set_postfix_str("Error!")
            return False


def main():
    parser = argparse.ArgumentParser(description="Batch process WAV files in parallel.")
    parser.add_argument("input_folder", type=str, help="Path to input folder")
    parser.add_argument("-m", "--model", type=str, default="quail-vf-l-16khz")
    parser.add_argument("-w", "--workers", type=int, default=2, help="Parallel threads")
    parser.add_argument(
        "-s",
        "--stt-api",
        type=str,
        default="soniox",
        help="STT API for transcription (soniox or deepgram, default: soniox)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_folder)

    if not input_dir.exists():
        print("Input folder not found.")
        sys.exit(1)

    wav_files = list(input_dir.glob("*.wav"))

    if not wav_files:
        print("No .wav files found.")
        sys.exit(0)

    print(f"Processing {len(wav_files)} files with {args.workers} workers...\n")

    # Reserve space for the progress bars (move cursor down)
    sys.stdout.write("\n" * args.workers)

    success_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        # Assign a fixed position index (0 to workers-1) to each file task
        # Ideally, we'd map workers to positions, but for simple batching,
        # we can just pass an index. To avoid collision in a simple way,
        # we can submit chunks or just let tqdm handle the visual stack overlap
        # (it might flicker if queue > workers).
        # A cleaner way for 'tqdm' parallel is to use position=index % workers.

        for i, wav_file in enumerate(wav_files):
            # position=0 is the bottom, so we add 1 to leave room for the main bar if we wanted one
            # Here we just stack them.
            pos = i % args.workers
            futures.append(
                executor.submit(
                    process_file, wav_file, input_dir, args.model, args.stt_api, pos
                )
            )

        for future in as_completed(futures):
            if future.result():
                success_count += 1

    print(f"\n\nDone! Successfully processed {success_count}/{len(wav_files)} files.")


if __name__ == "__main__":
    main()
