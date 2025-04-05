from api.logging.logger import init_logger
from api.schemas.language_translate import LanguageRequest

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

os.environ["CUDA_VISIBLE_DEVICES"] = ""
model_path = "ml_models/text_to_text_translate/model/nllb_model_600M"
device = torch.device("cpu")

def en_to_hi_translator(request: LanguageRequest) -> str:
    text = request.text

    target_lang_map = {
        "hi": "hin_Deva",
        "en": "eng_Latn"
    }

    if request.dest not in target_lang_map:
        init_logger(message=f"dest: {request.dest} Not found")
        raise ValueError(f"Unsupported destination language: {request.dest}")

    dest = target_lang_map[request.dest]


    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    if tokenizer.lang_code_to_id is None:
        tokenizer.lang_code_to_id = {
            "eng_Latn": tokenizer.convert_tokens_to_ids(">>eng_Latn<<"),
            "hin_Deva": tokenizer.convert_tokens_to_ids(">>hin_Deva<<"),
        }

    inputs = tokenizer(text, return_tensors="pt")

    forced_bos_token_id = tokenizer.lang_code_to_id
    if forced_bos_token_id is None:
        raise ValueError(f"Unsupported target language code: {dest}")

    inputs["forced_bos_token_id"] = forced_bos_token_id

    with torch.no_grad():
        output_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)

    translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
    print(f"Translated Text: {translated_text}")

    return translated_text