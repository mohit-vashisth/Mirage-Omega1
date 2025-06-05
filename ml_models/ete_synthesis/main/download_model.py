from TTS.utils.manage import ModelManager
import shutil
import os


destination_dir = r"C:\Users\Admin\Documents\Mirage-Omega1\ml_models\ete_synthesis\tts_model"
model_download_path = ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")

os.makedirs(destination_dir, exist_ok=True)

shutil.copytree(str(model_download_path), destination_dir, dirs_exist_ok=True)

print(f"Model copied to: {destination_dir}")

source = r"C:\Users\Admin\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"

print(f"Now Moving Model to: {destination_dir}")
shutil.move(source, destination_dir)

print("✅ Model moved successfully to:", destination_dir)

