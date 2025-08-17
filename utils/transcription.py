import whisper

# def transcribe_audio(file_path, model_size="small"):
#     model = whisper.load_model(model_size)
#     result = model.transcribe(file_path)
#     return result["text"]

# preffered model size for cpu
def transcribe_audio(file_path, model_size="tiny"):
    """
    Transcribe short audio using Whisper tiny model (lightweight).
    Options: tiny, base, small, medium, large
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"]
