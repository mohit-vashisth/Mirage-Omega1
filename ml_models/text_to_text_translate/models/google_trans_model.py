from googletrans import Translator

from api.schemas.language_translate import Language_response
from api.logging.logger import init_logger

def google_Trans_transltor(response: Language_response):
    try:
        translated_text = Translator().translate(text=response.text, dest=response.dest, src="auto").text
        init_logger(message=f"final text: {translated_text}")
        return translated_text
    except Exception as e:
        init_logger(message=f"an error occured: {str(e)}", level="error")
        raise RuntimeError("Unable to translate")