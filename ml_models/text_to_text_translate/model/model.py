import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "ml_models/text_to_text_translate/model/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

src_text = "मेरा नाम मोहित है तू कौन हो भाई?"
dest = "eng_Latn"
src = "hin_deva"

inputs = tokenizer(src_text, return_tensors="pt")
tokenizer.lang_code_to_id = {
    "eng_Latn": tokenizer.convert_tokens_to_ids(">>eng_Latn<<"),
    "hin_Deva": tokenizer.convert_tokens_to_ids(">>hin_Deva<<"),
}
inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id.get(dest)

# Generate translation
with torch.no_grad():
    output_tokens = model.generate(**inputs)

# Decode and print translation
translated_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
print(f"Translated Text: {translated_text}")
