from api.core import config
from api.logging.logger import init_logger
from api.schemas.language_translate import LanguageRequest

from fastapi import APIRouter, Request, status

from ml_models.text_to_text_translate.model.model import en_to_hi_translator
from ml_models.text_to_text_translate.model.transformers_model import google_translate

translate_route = APIRouter()

@translate_route.post(config.TRANSLATE_EP, status_code=status.HTTP_200_OK)
async def translate(req: LanguageRequest, request: Request):
    init_logger(message=f"text: {req.text} | destination: {req.dest}", request=request)

    translated_text = en_to_hi_translator(request=req)

    return {
        "translated_text": translated_text
    }