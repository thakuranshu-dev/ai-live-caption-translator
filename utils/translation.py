# Utility functions for text translation
from transformers import MarianMTModel, MarianTokenizer
import config

# Cache models to avoid reloading every call
_loaded_models = {}

def load_model(model_name):
    """
    Load and cache MarianMT model by name.
    """
    if model_name not in _loaded_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _loaded_models[model_name] = (tokenizer, model)
    return _loaded_models[model_name]

def translate_text(text, src_lang=config.DEFAULT_SOURCE_LANG, tgt_lang=config.DEFAULT_TARGET_LANG):
    """
    Translate text using MarianMT (lightweight translation models).
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    if model_name not in _loaded_models:
        tokenizer, model = load_model(model_name)

    tokens = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
