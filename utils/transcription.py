# Utility functions for audio transcription
import whisper
import numpy as np
import tempfile
from scipy.io.wavfile import write
import config

# Cache models so they don’t reload every call
_loaded_models = {}

def load_model(model_size=config.DEFAULT_MODEL_SIZE):
    """
    Load and cache Whisper model by size.
    """
    if model_size not in _loaded_models:
        _loaded_models[model_size] = whisper.load_model(model_size)
    return _loaded_models[model_size]

def transcribe_audio(file_path, model_size):
    """
    Transcribe short audio using Whisper tiny model (lightweight).
    Options: tiny, base, small, medium, large
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"]


def transcribe_audio_array(audio_array, model_size=config.DEFAULT_MODEL_SIZE, samplerate=config.AUDIO_SAMPLE_RATE):
    """
    Transcribe directly from a NumPy audio array.
    Whisper expects a file, so we save to temp first.
    """
    model = load_model(model_size)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        write(tmpfile.name, samplerate, audio_array.astype(np.int16))
        result = model.transcribe(tmpfile.name)
    return result["text"]