import streamlit as st
import sounddevice as sd
import queue
import tempfile
import os
from utils.transcription import transcribe_audio
from utils.translation import translate_text

st.set_page_config(page_title="AI Live Caption Translator", layout="wide")

st.title("🎙 AI Live Caption Translator")
st.write("Real-time captions with on-the-fly translation")

# Config
source_lang = st.selectbox("Source Language", ["en", "es", "fr", "hi"])
target_lang = st.selectbox("Target Language", ["es", "en", "fr", "hi"])

audio_q = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def record_audio(duration=15, samplerate=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        return tmpfile.name

if st.button("🎤 Record 5 sec audio"):
    audio_file_path = record_audio()
    st.audio(audio_file_path)

    st.write("Transcribing...")
    transcript = transcribe_audio(audio_file_path, model_size="small")
    st.write(f"**Original ({source_lang}):** {transcript}")

    st.write("Translating...")
    translated = translate_text(transcript, src_lang=source_lang, tgt_lang=target_lang)
    st.write(f"**Translated ({target_lang}):** {translated}")

    os.remove(audio_file_path)
