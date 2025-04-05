from fastapi import HTTPException, status
import googletrans as gt

from api.logging.logger import init_logger
from api.schemas.language_translate import LanguageRequest

async def google_translate(request: LanguageRequest) -> str:
    translator = gt.Translator()
    language_detect = await translator.detect(text=request.text)

    if language_detect.confidence < 0.8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        )
    translated = await translator.translate(text=request.text, dest=request.dest)
    init_logger(message=f"text: {translated}")

    return translated.text