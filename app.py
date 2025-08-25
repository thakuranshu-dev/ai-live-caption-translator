import streamlit as st
import os
import threading
import numpy as np
from utils.transcription import transcribe_audio, transcribe_audio_array
from utils.translation import translate_text
from utils.audio_capture import record_audio_test, audio_q, start_stream
import config

st.set_page_config(page_title="AI Live Caption Translator", layout="wide")

st.title("🎙 AI Live Caption Translator")
st.write("Real-time captions with on-the-fly translation")

source_lang = st.selectbox("Source Language", ["en", "es", "fr", "hi"])
target_lang = st.selectbox("Target Language", ["es", "en", "fr", "hi"])

caption_placeholder = st.empty()
translation_placeholder = st.empty()

def process_audio_stream():
    """
    Continuously fetch audio from queue and process in chunks.
    """
    buffer = np.zeros((0, 1))
    samplerate = 16000

    while True:
        chunk = audio_q.get()
        buffer = np.concatenate([buffer, chunk])

        # Process every 5 seconds worth of audio
        if len(buffer) >= 5 * samplerate:
            audio_data = buffer[:5 * samplerate].flatten()
            buffer = buffer[5 * samplerate:]

            transcript = transcribe_audio_array(audio_data, model_size=config.DEFAULT_MODEL_SIZE)

            if transcript.strip():
                caption_placeholder.markdown(f"📝 **Transcript:** {transcript}")

                translated = translate_text(transcript, src_lang=source_lang, tgt_lang=target_lang)
                translation_placeholder.markdown(f"🌍 **Translation ({target_lang}):** {translated}")

if st.button("🎤 Start Live Captions"):
    threading.Thread(target=process_audio_stream, daemon=True).start()

    with start_stream():  # microphone stream
        st.success("Listening... Close the app to stop.")
        if st.button("🛑 Stop"):
            threading.Thread(target=process_audio_stream, daemon=True).stop()
            st.info("Stopped listening.")
        while True:
            pass

# if st.button("🎤 Record 10 sec audio"):
#     info = st.info("Recording... speak now")
#     audio_file_path = record_audio_test()
#     st.audio(audio_file_path)
#     info.empty()

#     st.write("Transcribing...")
#     transcript = transcribe_audio(audio_file_path, model_size=config.DEFAULT_MODEL_SIZE)
#     st.write(f"**Original ({source_lang}):** {transcript}")

#     st.write("Translating...")
#     translated = translate_text(transcript, src_lang=source_lang, tgt_lang=target_lang)
#     st.write(f"**Translated ({target_lang}):** {translated}")

#     os.remove(audio_file_path)
