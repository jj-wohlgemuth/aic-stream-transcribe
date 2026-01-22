# AIC Stream & Transcribe

A real-time audio enhancement and transcription tool using the **AIC SDK** for speech enhancement and **Soniox** for high-accuracy speech-to-text.

This tool captures audio from your microphone, processes it using the AIC SDK, and streams both the raw and enhanced audio to Soniox for real-time transcription. At the end of the session, it generates an interactive HTML report comparing the waveforms, spectrograms, and transcripts of the raw vs. enhanced audio.

# Getting started

## Install UV on macOS
```bash
brew install uv
```

## Set API and SDK Keys
```bash
echo 'export AIC_SDK_LICENSE="your_actual_aic_license_key"' >> ~/.zshrc
echo 'export SONIOX_API_KEY="your_actual_soniox_api_key"' >> ~/.zshrc
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
| `-el`, `--enhancement-level` | Audio enhancement intensity (0.0 to 1.0). | `1.0` |
| `-vg`, `--voice-gain` | Gain factor applied to voice signal. | `1.0` |
| `-t`, `--transcribe` | Enable transcription (`true`/`false`). | `true` |

Here are a few example parameter sets you can add to the usage section to help users get started quickly with different configurations:

### Example Scenarios

16 kHz Voice Focus

```bash
uv run demo.py -i M2 -o M2 -sr 16000 -m quail-vf-l-16khz
```

48kHz perceptual speech enhancement Processing without transcription

```bash
uv run demo.py -i M2 -o M2 -sr 48000 -m sparrow-l-48khz -t false
```
