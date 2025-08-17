from transformers import MarianMTModel, MarianTokenizer

# Cache models to avoid reloading every call
_loaded_models = {}

# def translate_text(text, src_lang="en", tgt_lang="es"):
#     model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
#     tokenizer = MarianTokenizer.from_pretrained(model_name)
#     model = MarianMTModel.from_pretrained(model_name)

#     tokens = tokenizer(text, return_tensors="pt", padding=True)
#     translated = model.generate(**tokens)
#     return tokenizer.decode(translated[0], skip_special_tokens=True)

# optimized method
def translate_text(text, src_lang="en", tgt_lang="es"):
    """
    Translate text using MarianMT (lightweight translation models).
    """
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"

    if model_name not in _loaded_models:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _loaded_models[model_name] = (tokenizer, model)
    else:
        tokenizer, model = _loaded_models[model_name]

    tokens = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
