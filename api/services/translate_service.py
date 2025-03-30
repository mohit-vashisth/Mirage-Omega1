from api.core import config
from api.logging.logger import init_logger
from api.schemas.language_translate import LanguageRequest

from fastapi import APIRouter, Request, status

translate_route = APIRouter()

@translate_route.post(config.TRANSLATE_EP, status_code=status.HTTP_200_OK)
async def translate(req: LanguageRequest, request: Request):
    init_logger(message=f"text: {req.text} | destination: {req.dest}", request=request)
    return {
        "final_text": "mera naam mohit hai"
    }