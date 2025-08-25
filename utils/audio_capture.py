# Utility functions for audio capture
import sounddevice as sd
import tempfile
import config

def record_audio_test(duration=config.RECORD_DURATION, samplerate=config.AUDIO_SAMPLE_RATE):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        from scipy.io.wavfile import write
        write(tmpfile.name, samplerate, recording)
        return tmpfile.name
    
import queue
# Global queue to hold audio chunks
audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback called automatically when new audio arrives."""
    if status:
        print("Audio status:", status)
    audio_q.put(indata.copy())

def start_stream(samplerate=16000, blocksize=4000):
    """Start microphone stream and push audio chunks into queue."""
    return sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=samplerate,
        blocksize=blocksize
    )
