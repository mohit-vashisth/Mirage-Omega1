from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
model_store_dir = "ml_models/ete-synthesis"

print("downloading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(model_store_dir)
model.save_pretrained(model_store_dir)
print("✅ Model saved successfully!")
