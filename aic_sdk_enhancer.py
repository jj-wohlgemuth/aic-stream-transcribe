import os
import numpy as np
import soundfile as sf
from tqdm import tqdm
import aic_sdk as aic
import warnings


def _load_audio_original(input_wav: str) -> tuple[np.ndarray, int, int]:
    """Load audio with original sample rate and channels, return numpy ndarray, sample_rate, and num_channels."""
    # Use soundfile to preserve original properties
    audio, sample_rate = sf.read(input_wav, dtype="float32")

    # audio is (frames,) for mono or (frames, channels) for multi-channel
    if audio.ndim == 1:
        # Mono audio: reshape to (1, frames)
        audio = audio.reshape(1, -1)
        num_channels = 1
    else:
        # Multi-channel: transpose from (frames, channels) to (channels, frames)
        audio = audio.T
        num_channels = audio.shape[0]

    return audio, sample_rate, num_channels


def process_chunk(
    processor: aic.Processor,
    chunk: np.ndarray,
    buffer_size: int,
    num_channels: int,
) -> np.ndarray:
    """Process a single audio chunk with the given processor."""
    valid_samples = chunk.shape[1]

    # Create and zero-initialize process buffer
    process_buffer = np.zeros((num_channels, buffer_size), dtype=np.float32)

    # Copy input data into the buffer
    process_buffer[:, :valid_samples] = chunk

    # Process the chunk
    processed_chunk = processor.process(process_buffer)

    # Return only the valid part
    return processed_chunk[:, :valid_samples]


def get_license_key() -> str | None:
    """Get license key from environment variable or .env file."""
    try:
        license_key = os.getenv("AIC_SDK_LICENSE")
    except Exception:
        license_key = None
    return license_key


def process_single_file(
    input_wav: str,
    enhancement_level: float | None,
    model_name: str,
) -> np.ndarray:
    license_key = get_license_key()
    if os.path.exists(model_name):
        model_path = model_name
    else:
        print(f"Model file '{model_name}' not found locally. Attempting download...")
        cache_dir = "models"
        os.makedirs(cache_dir, exist_ok=True)
        try:
            model_path = aic.Model.download(model_name, cache_dir)
            print(f"Downloaded model to: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_name}': {e}")
    model = aic.Model.from_file(model_path)
    assert license_key is not None, "AIC_SDK_LICENSE environment variable not set."
    processor = aic.Processor(
        model,
        license_key,
    )
    audio_input, sample_rate, num_channels = _load_audio_original(input_wav)

    # Create optimal config using original number of channels and sample rate
    config = aic.ProcessorConfig.optimal(
        model, sample_rate=sample_rate, num_channels=num_channels
    )

    # Re-initialize the processor with the new config for this file
    processor.initialize(config)
    proc_ctx = processor.get_processor_context()

    # Reset processor state to clear any previous file's data
    proc_ctx.reset()

    latency_samples = proc_ctx.get_output_delay()

    # Pad the input audio with zeros at the beginning to compensate for algorithmic delay
    padding = np.zeros((num_channels, latency_samples), dtype=np.float32)
    audio_input = np.concatenate([padding, audio_input], axis=1)

    num_frames_model = config.num_frames
    num_frames_audio_input = audio_input.shape[1]

    # Set Enhancement Parameter if provided
    if enhancement_level is not None:
        try:
            proc_ctx.set_parameter(
                aic.ProcessorParameter.EnhancementLevel, enhancement_level
            )
        except aic.ParameterFixedError:
            warnings.warn(
                "Enhancement level cannot be adjusted for this model. "
                "This model has a fixed enhancement level. Please run without specifying --enhancement-level.",
                UserWarning,
            )
    else:
        # Use model's default enhancement level
        enhancement_level = proc_ctx.get_parameter(
            aic.ProcessorParameter.EnhancementLevel
        )

    # Initialize output array
    output = np.zeros_like(audio_input)

    # Process the entire file sequentially with this processor
    num_chunks = (num_frames_audio_input + num_frames_model - 1) // num_frames_model

    with tqdm(
        total=num_chunks,
        desc=f"Processing {os.path.basename(input_wav)}",
    ) as pbar:
        for chunk_start in range(0, num_frames_audio_input, num_frames_model):
            chunk_end = min(chunk_start + num_frames_model, num_frames_audio_input)
            chunk = audio_input[:, chunk_start:chunk_end]

            # Process the chunk
            processed = process_chunk(
                processor,
                chunk,
                num_frames_model,
                config.num_channels,
            )

            output[:, chunk_start : chunk_start + processed.shape[1]] = processed
            pbar.update(1)
    output = output[:, latency_samples:]
    return output
