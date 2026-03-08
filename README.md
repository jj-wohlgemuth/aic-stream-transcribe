# AIC Stream & Transcribe

A real-time audio enhancement and transcription tool using the **AIC SDK** for speech enhancement and **Soniox** for high-accuracy speech-to-text.

This tool captures audio from your microphone, processes it using the AIC SDK, and streams both the raw and enhanced audio to Soniox for real-time transcription. At the end of the session, it generates an interactive HTML report comparing the waveforms, spectrograms, and transcripts of the raw vs. enhanced audio.

## Acknowledgments

Special thanks to GitHub users [@user4-33](https://github.com/user4-33) and [@steckes](https://github.com/steckes) for creating the code that made this demo possible.

# Getting started

## Install UV on macOS
```bash
brew install uv
```

## Set API and SDK Keys
```bash
echo 'export AIC_SDK_LICENSE="your_actual_aic_license_key"' >> ~/.zshrc
echo 'export SONIOX_API_KEY="your_actual_soniox_api_key"' >> ~/.zshrc
echo 'export DEEPGRAM_API_KEY="your_actual_deepgram_api_key"' >> ~/.zshrc
```
Run this to reload the shell
```bash
source ~/.zshrc
```

## Install dependencies, activate venv and run demo

1.  **Install dependencies:**
This command creates a virtual environment and installs all packages defined in `pyproject.toml`.
```bash
uv sync
```

2.  **Run the demo:**
`uv run` automatically uses the correct virtual environment, so you don't need to manually activate it.
```bash
uv run demo.py -h
```

### Command Line Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `-h`, `--help` | Show this help message and exit. | |
| `-l`, `--list-devices` | Show list of available audio devices and exit. | |
| `-i`, `--input-device` | Input device (numeric ID or substring). | System Default |
| `-o`, `--output-device` | Output device (numeric ID or substring). | System Default |
| `-m`, `--model` | Model name (e.g. `quail-vf-l-16khz`) or path to `.aicmodel`. | `quail-vf-l-16khz` |
| `-sr`, `--samplerate` | Sampling rate (overrides model optimal rate). | Model Optimal |
| `-c`, `--channels` | Number of channels. | `1` |
| `-el`, `--enhancement-level` | Audio enhancement intensity (0.0 to 1.0). | `0.8` |
| `-a`, `--amplify` | Pre-enhancement input amplification in dB. Clips signal to [-1, 1] before processing. A clipping warning is printed if the amplified signal exceeds the range. | `0.0` |
| `-t`, `--transcribe` | Enable transcription (`true`/`false`). | `true` |
| `-s`, `--stt-api` | Specify which STT API to use (`soniox`/`deepgram`). | `soniox` |

### Enhancement Level

`--enhancement-level` (0.0–1.0) controls how aggressively the model suppresses noise and enhances speech. A value of `1.0` applies the full enhancement; `0.0` passes audio through with minimal processing. Some models have a fixed enhancement level and will ignore this parameter with a warning.

### Amplification and Clipping

`--amplify` applies a gain (in dB) to the input signal **before** it is passed to the enhancement model. This is useful when the microphone input is too quiet for the model to process effectively.

- Positive values (e.g. `6.0`) boost the signal; negative values attenuate it.
- After amplification the signal is **hard-clipped** to the range `[-1.0, 1.0]` to prevent overflow. If clipping occurs, a yellow warning is printed to the terminal (throttled to once per second).
- Use the minimum amplification needed — excessive gain will introduce clipping distortion before enhancement.

Here are a few example parameter sets you can add to the usage section to help users get started quickly with different configurations:

### Example Scenarios

16 kHz Voice Focus on Motu M2 Audio Interface

```bash
uv run demo.py -i M2 -o M2 -sr 16000 -m quail-vf-2.0-l-16khz
```

16 kHz Voice Focus on Mac Book

```bash
uv run demo.py -i "MacBook Pro Microphone" -o "MacBook Pro Speakers" -sr 16000 -m quail-vf-2.0-l-16khz -el 0.8 -a 9.0
```

48kHz perceptual speech enhancement Processing without transcription

```bash
uv run demo.py -i M2 -o M2 -sr 48000 -m sparrow-l-48khz -t false
```

## File Processing (Offline Mode)

In addition to real-time streaming, this project includes a script (`demo_prerecorded.py`) to process existing `.wav` files. This tool enhances a pre-recorded audio file using the AIC SDK, transcribes both the original and enhanced versions using Soniox, and generates a comparative HTML report.

### Usage

```bash
uv run process.py [INPUT_FILE] [OPTIONS]
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `input_file` | (Required) Path to the input WAV file. | |
| `-m`, `--model` | AIC Model name (e.g., `quail-vf-l-16khz`) or path to a `.aicmodel` file. | `quail-vf-l-16khz` |
| `-o`, `--output` | Path where the enhanced audio WAV will be saved. | `prerecorded_enhanced.wav` |
| `-h`, `--help` | Show the help message and exit. | |