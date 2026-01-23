from scipy.signal import spectrogram
from holoviews.operation.datashader import rasterize
import holoviews as hv
import panel as pn
import numpy as np
import soundfile as sf
from pathlib import Path

# --- GLOBAL SETTINGS ---
# Fixed dimensions for 1920x1080 screens
PLOT_WIDTH = 1100
WAVEFORM_HEIGHT = 80
SPECTROGRAM_HEIGHT = 190
TRANSCRIPT_HEIGHT = 120


def audio_with_spectrogram(signal, fs_hz, title, transcript=None):
    hv.extension("bokeh", logo=False)

    # 1. Generate Waveform Data
    audio_data = signal.flatten()
    duration = len(audio_data) / fs_hz
    # times = np.linspace(0, duration, len(audio_data))

    # 2. Fixed Waveform Plot
    # waveform = hv.Curve((times, audio_data), "Time (s)", "Amplitude").opts(
    #     title=title,
    #     width=PLOT_WIDTH,
    #     height=WAVEFORM_HEIGHT,
    #     color="#812481",
    #     tools=["hover", "save"],
    #     toolbar="above",
    #     xaxis=None,  # Hide x-axis labels on top plot to reduce clutter
    # )

    # 3. Fixed Spectrogram Plot
    f, t, sxx = spectrogram(audio_data, fs_hz, nfft=256)
    log_sxx = np.log10(sxx + 1e-10)

    spec_img = rasterize(
        hv.Image((t, f, log_sxx), ["Time (s)", "Frequency (Hz)"])
    ).opts(
        width=PLOT_WIDTH,
        height=SPECTROGRAM_HEIGHT,
        cmap="magma",
        xlabel="Time (s)",
        toolbar="above",
    )

    # 4. Fixed Width Audio Player
    audio = pn.pane.Audio(
        audio_data,
        sample_rate=fs_hz,
        name="Audio",
        throttle=50,
        width=PLOT_WIDTH,  # FIXED WIDTH
        styles={
            "background": "#FFFFFF",
            "border-radius": "5px",
            "margin-top": "5px",
            "margin-bottom": "10px",
        },
        stylesheets=[
            """
                audio {
                    width: 100%;
                    filter: invert(90%) sepia(20%) saturate(300%) hue-rotate(275deg);
                    outline: none;
                }
            """
        ],
    )

    # 5. Fixed Width Transcript
    if transcript:
        formatted_transcript = transcript.replace("<end>", "<br>").replace("\n", "<br>")
        markdown_text = f"**Transcript:**\n\n{formatted_transcript}"
    else:
        markdown_text = "_No transcript provided_"

    transcript_pane = pn.pane.Markdown(
        markdown_text,
        width=PLOT_WIDTH,  # FIXED WIDTH
        height=TRANSCRIPT_HEIGHT,  # FIXED HEIGHT
        styles={
            "overflow-y": "auto",
            "border": "1px solid #ddd",
            "padding": "15px",
            "background": "#fafafa",
            "border-radius": "4px",
        },
    )

    # 6. Return Fixed Column
    return pn.Column(
        # waveform,
        spec_img,
        audio,
        transcript_pane,
        width=PLOT_WIDTH,  # Lock the container width
        styles={"margin-bottom": "40px"},  # Spacing between different audio files
    )


def create_html_report(
    raw_transcript: str,
    enhanced_transcript: str,
    raw_wav_path: str,
    enhanced_wav_path: str,
    output_html: str = "recordings.html",
):
    hv.extension("bokeh", logo=False)
    pn.extension()

    raw_audio, fs_raw = sf.read(raw_wav_path)
    enh_audio, fs_enh = sf.read(enhanced_wav_path)

    raw_panel = audio_with_spectrogram(raw_audio, fs_raw, "Raw Audio", raw_transcript)
    enh_panel = audio_with_spectrogram(
        enh_audio, fs_enh, "Enhanced Audio", enhanced_transcript
    )

    # Main Layout - Centered on screen
    layout = pn.Column(
        raw_panel,
        pn.layout.Divider(width=PLOT_WIDTH),
        enh_panel,
        # Center the column on the screen
        styles={"margin-left": "auto", "margin-right": "auto", "padding-top": "20px"},
    )

    output_path = Path(output_html)
    layout.save(output_path)

    print("\nHTML report created:")
    absolute_path = output_path.resolve()
    print(f"file://{absolute_path}")
