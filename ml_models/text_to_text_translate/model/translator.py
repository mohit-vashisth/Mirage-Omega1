from translate import Translator

from api.schemas.language_translate import LanguageRequest

languages = ["hi", "en"]

def translate_to_hi(request: LanguageRequest) -> str:
    translator = Translator(from_lang="hi", to_lang=request.dest)
    translated_text = translator.translate(request.text)
    
    return translated_text