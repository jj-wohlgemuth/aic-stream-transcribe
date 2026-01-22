from scipy.signal import spectrogram
from holoviews.operation.datashader import rasterize
import holoviews as hv
import panel as pn
import numpy as np
import soundfile as sf
from pathlib import Path


def audio_with_spectrogram(signal, fs_hz, title, transcript=None, height=380):
    hv.extension("bokeh", logo=False)

    # 1. Generate Waveform
    audio_data = signal.flatten()
    duration = len(audio_data) / fs_hz
    times = np.linspace(0, duration, len(audio_data))

    waveform = hv.Curve((times, audio_data), "Time (s)", "Amplitude").opts(
        title=title,
        responsive=True,
        height=int(height / 2),
        color="#812481",
        tools=["hover", "save"],  # Removed 'tap' as we aren't using it
    )

    # 2. Generate Spectrogram
    f, t, sxx = spectrogram(audio_data, fs_hz, nfft=256)
    log_sxx = np.log10(sxx + 1e-10)
    spec_img = rasterize(
        hv.Image((t, f, log_sxx), ["Time (s)", "Frequency (Hz)"]), precompute=True
    ).opts(
        responsive=True,
        height=int(height / 2),
        cmap="magma",
        xlabel="",
    )

    # 3. Create Magma-Themed Player
    audio = pn.pane.Audio(
        audio_data,
        sample_rate=fs_hz,
        name="Audio",
        throttle=50,
        sizing_mode="stretch_width",
        styles={
            "background": "#FFFFFF",
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

    # 4. Format Transcript
    if transcript:
        # Standardize newlines to HTML breaks
        formatted_transcript = transcript.replace("<end>", "<br>").replace("\n", "<br>")
        markdown_text = f"**Transcript:**\n\n{formatted_transcript}"
    else:
        markdown_text = "_No transcript provided_"

    transcript_pane = pn.pane.Markdown(
        markdown_text,
        sizing_mode="stretch_width",
        height=150,
        styles={"overflow-y": "auto", "border": "1px solid #eee", "padding": "10px"},
    )

    # 5. Return Full Stack (Waveform + Spectrogram + Audio + Text)
    return pn.Column(
        waveform, spec_img, audio, transcript_pane, sizing_mode="stretch_width"
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

    layout = pn.Column(
        pn.pane.Markdown("## Audio Comparison"),
        raw_panel,
        pn.layout.Divider(),
        enh_panel,
        sizing_mode="stretch_width",
    )

    output_path = Path(output_html)

    # NOTE: embed=True removed to prevent Audio serialization errors.
    # .save() still produces a standalone HTML file with all data included.
    layout.save(output_path)

    return output_path
