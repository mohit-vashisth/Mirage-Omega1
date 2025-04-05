from translate import Translator

from api.schemas.language_translate import LanguageRequest

languages = ["hi", "en"]

def translate_to_hi(request: LanguageRequest):
    for language  in languages:
        translator = Translator(
            to_lang=request.dest,
            from_lang="hi"
        )
        translated_text = translator.translate(request.text)
        
        print(translated_text)